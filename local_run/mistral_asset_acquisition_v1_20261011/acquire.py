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

def headroom():
    raw = subprocess.check_output(['quota', '-v', '-w'], text=True)
    rows = [line.split() for line in raw.splitlines() if line.strip().startswith('/dev/sdb1 ')]
    require(len(rows) == 1, 'Expected exactly one /dev/sdb1 quota row')
    fields = rows[0]; used, soft, hard = [int(x.rstrip('*')) for x in fields[1:4]]
    require(soft > 0 and hard >= soft, 'Unexpected quota limits')
    return {'used_kib': used, 'soft_kib': soft, 'hard_kib': hard, 'available_soft_bytes': (soft-used)*1024}

def main():
    ap=argparse.ArgumentParser()
    for name in ('expectations','expectations-sha256','cache','out','script-sha256'): ap.add_argument('--'+name,required=True)
    a=ap.parse_args();out=Path(a.out);cache=Path(a.cache);expect=Path(a.expectations)
    require(platform.system()=='Linux' and os.environ.get('SLURM_JOB_ID'),'Linux Slurm required')
    require(socket.gethostname().split('.')[:2] != ['csis','mn1'],'Login node excluded')
    require(not out.exists(),'New output directory required');os.umask(0o077);out.mkdir(parents=True)
    start=time.perf_counter();report={'schema':'third-mistral-asset-acquisition-v1','passed':False,'model_loaded':False,'model_forwards':0,'human_responses_created':0,'confirmatory_study_ready':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'downloaded_files':[]}
    try:
        require(sha(__file__)==a.script_sha256,'Acquisition script hash mismatch')
        require(sha(expect)==a.expectations_sha256,'Expectations hash mismatch')
        pins=json.loads(expect.read_text()); files=pins['files']
        require(pins['model_id']=='mistralai/Mistral-Nemo-Instruct-2407' and pins['revision']=='04d8a90549d23fc6bd7f642064003592df51e9b3','Unexpected snapshot')
        require(len(files)==13 and sum(f['size_bytes'] for f in files)==24510683806,'Unexpected acquisition size')
        require(all(Path(f['relative_path']).name==f['relative_path'] for f in files),'Unexpected file path')
        require(not cache.exists(),'New isolated cache required');cache.mkdir(parents=True)
        filesystem=subprocess.check_output(['df','-Pk',str(cache)],text=True).splitlines()[-1].split()[0]
        require(filesystem=='/dev/sdb1','Cache filesystem does not match quota device')
        reserve=8*1024**3;quota=headroom();report['initial_quota']=quota
        require(quota['available_soft_bytes'] >= pins['total_bytes']+reserve,'Insufficient quota for exact files plus 8 GiB reserve')
        from importlib.metadata import version
        require(version('huggingface_hub')=='0.36.2','HF CLI package changed')
        env=dict(os.environ)
        for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'): env.pop(key,None)
        env.update({'HF_HUB_DISABLE_XET':'1','HF_XET_CHUNK_CACHE_SIZE_BYTES':'0','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DOWNLOAD_TIMEOUT':'120','PYTHONDONTWRITEBYTECODE':'1'})
        snapshot=cache/('models--'+pins['model_id'].replace('/','--'))/'snapshots'/pins['revision']
        report.update({'cache':str(cache),'snapshot':str(snapshot),'expectations_sha256':a.expectations_sha256,'script_sha256':a.script_sha256,'python':sys.version,'hf_version':version('huggingface_hub'),'reserve_bytes':reserve,'download_implementation':'Installed hf console entry point via huggingface_hub.cli.hf.main; binary wrapper absent','download_strategy':'One exact file at a time into single isolated cache; native duplicate weight file excluded; Xet chunk cache disabled'})
        ordered=sorted(files,key=lambda f:(f['relative_path'].endswith('.safetensors'),f['relative_path']))
        for spec in ordered:
            require(headroom()['available_soft_bytes'] >= spec['size_bytes']+reserve,'Quota reserve would be violated')
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
        require(sha(expect)==a.expectations_sha256 and sha(__file__)==a.script_sha256,'Acquisition bindings changed')
        for row in report['downloaded_files']: require(sha(snapshot/row['relative_path'])==row['sha256'],'Downloaded asset changed')
        report.update({'passed':True,'final_quota':headroom(),'total_verified_bytes':sum(f['bytes'] for f in report['downloaded_files'])})
    except Exception as error:
        report['error']=str(error);report['error_type']=type(error).__name__
    report['seconds']=time.perf_counter()-start;report['peak_rss_kib_linux']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(out/'result.json',report);print(json.dumps({k:report.get(k) for k in ['passed','error','seconds','total_verified_bytes']}))
    return 0 if report['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
