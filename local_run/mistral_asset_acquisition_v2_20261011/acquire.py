#!/usr/bin/env python3
"""Acquire exactly the frozen public HF snapshot files; never load a model."""
import argparse, datetime, hashlib, json, os, platform, resource, socket, subprocess, sys, time
from pathlib import Path

def require(value, message):
    if not value: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser()
    for name in ('expectations','cache','out','package-sha256'): ap.add_argument('--'+name,required=True)
    a=ap.parse_args();out=Path(a.out);cache=Path(a.cache);expect=Path(a.expectations)
    require(platform.system()=='Linux' and os.environ.get('SLURM_JOB_ID'),'Linux Slurm required')
    require(socket.gethostname().split('.')[:2] != ['csis','mn1'],'Login node excluded')
    require(not out.exists(),'New output directory required');os.umask(0o077);out.mkdir(parents=True)
    start=time.perf_counter();report={'schema':'third-mistral-asset-acquisition-v2','passed':False,'model_loaded':False,'model_forwards':0,'human_responses_created':0,'confirmatory_study_ready':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'downloaded_files':[]}
    try:
        from package_check import verify
        verify(Path(__file__).resolve().parent,a.package_sha256)
        import quota_grants as q
        require(sha(expect)==q.EXPECTATIONS_SHA,'Expectations hash mismatch')
        pins=json.loads(expect.read_text()); files=pins['files']
        require(pins['model_id']=='mistralai/Mistral-Nemo-Instruct-2407' and pins['revision']=='04d8a90549d23fc6bd7f642064003592df51e9b3','Unexpected snapshot')
        require(len(files)==13 and sum(f['size_bytes'] for f in files)==24510683806,'Unexpected acquisition size')
        require(all(Path(f['relative_path']).name==f['relative_path'] for f in files),'Unexpected file path')
        require(not cache.exists(),'New isolated cache required');cache.mkdir(parents=True)
        mount = q.mount_identity(cache)
        require(mount == q.WORKER_MOUNT, 'Compute NFS mapping differs from observed mapping')
        (out / 'quota_requests').mkdir()
        ordered = q.ordered_files(pins)
        grant, grant_sha = q.wait_for_grant(out, -1, ordered, cache, a.package_sha256, os.environ['SLURM_JOB_ID'])
        report['quota_grants'] = [{'index': -1, 'grant_sha256': grant_sha, 'grant': grant}]
        report['compute_mount'] = mount
        from importlib.metadata import version
        require(version('huggingface_hub')=='0.36.2','HF CLI package changed')
        env=dict(os.environ)
        for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'): env.pop(key,None)
        env.update({'HF_HUB_DISABLE_XET':'1','HF_XET_CHUNK_CACHE_SIZE_BYTES':'0','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DOWNLOAD_TIMEOUT':'120','PYTHONDONTWRITEBYTECODE':'1'})
        snapshot=cache/('models--'+pins['model_id'].replace('/','--'))/'snapshots'/pins['revision']
        report.update({'cache':str(cache),'snapshot':str(snapshot),'expectations_sha256':q.EXPECTATIONS_SHA,'package_sha256':a.package_sha256,'python':sys.version,'hf_version':version('huggingface_hub'),'reserve_bytes':q.RESERVE_BYTES,'download_implementation':'Installed hf console entry point via huggingface_hub.cli.hf.main; binary wrapper absent','download_strategy':'One exact file at a time into single isolated cache; native duplicate weight file excluded; Xet chunk cache disabled'})
        for index,spec in enumerate(ordered):
            require(q.mount_identity(cache) == q.WORKER_MOUNT, 'Compute mount changed')
            grant,grant_sha=q.wait_for_grant(out,index,ordered,cache,a.package_sha256,os.environ['SLURM_JOB_ID'])
            report['quota_grants'].append({'index':index,'grant_sha256':grant_sha,'grant':grant})
            name=spec['relative_path'];cmd=[sys.executable,'-c','from huggingface_hub.cli.hf import main; main()','download',pins['model_id'],name,'--revision',pins['revision'],'--cache-dir',str(cache),'--max-workers','1']
            tick=time.perf_counter()
            with (out/(name+'.download.log')).open('x') as log:
                p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=False,timeout=3600)
            require(p.returncode==0,'HF download failed: '+name)
            target=snapshot/name;resolved=target.resolve(strict=True)
            require(resolved.is_relative_to(cache.resolve()) and resolved.is_file(),'Snapshot file outside isolated cache')
            require(target.stat().st_size==spec['size_bytes'],'Downloaded size mismatch: '+name)
            digest=sha(target)
            if spec['identity_kind']=='hub_lfs_sha256': require(digest==spec['sha256'],'Published LFS SHA256 mismatch: '+name)
            else:
                raw=target.read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                require(blob==spec['hub_git_blob_sha1'],'Published Git blob mismatch: '+name)
            row={'relative_path':name,'bytes':spec['size_bytes'],'sha256':digest,'published_identity_verified':spec['identity_kind'],'seconds_download_and_hash':time.perf_counter()-tick}
            report['downloaded_files'].append(row);write(out/(name+'.verified.json'),row)
        require(sorted(p.name for p in snapshot.iterdir())==sorted(f['relative_path'] for f in files),'Snapshot inventory mismatch')
        require(sha(expect)==q.EXPECTATIONS_SHA,'Acquisition expectation changed')
        verify(Path(__file__).resolve().parent,a.package_sha256)
        for row in report['downloaded_files']: require(sha(snapshot/row['relative_path'])==row['sha256'],'Downloaded asset changed')
        require(len(report['quota_grants']) == 14, 'Incomplete quota grant ledger')
        for item in report['quota_grants']:
            prefix = f'{item["index"] + 1:02d}'
            require(q.sha(out/'quota_requests'/(prefix+'.grant.json')) == item['grant_sha256'], 'Quota grant changed')
            require(q.sha(out/'quota_requests'/(prefix+'.request.json')) == item['grant']['request_sha256'], 'Quota request changed')
            require(q.sha(out/'quota_requests'/item['grant']['quota_raw_file']) == item['grant']['quota_raw_sha256'], 'Raw quota evidence changed')
        report.update({'passed':True,'quota_grants_verified':len(report['quota_grants']),'quota_observation_is_reservation':False,'final_quota_observed':False,'total_verified_bytes':sum(f['bytes'] for f in report['downloaded_files'])})
    except Exception as error:
        report['error']=str(error);report['error_type']=type(error).__name__
    report['quota_scope']='Fresh login-node user-quota snapshot before initial allocation and each file, not a reservation against concurrent writers; final quota is not measured.'
    report['seconds']=time.perf_counter()-start;report['peak_rss_kib_linux']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(out/'result.json',report);print(json.dumps({k:report.get(k) for k in ['passed','error','seconds','total_verified_bytes']}))
    return 0 if report['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
