"""One submission; metadata-only login preflight; immutable stage intent."""
import argparse
from datetime import datetime, timezone
import json
import os
import time
import census as x

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package-sha256',required=True)
    args = p.parse_args()
    x.verify(x.FOLDER,args.package_sha256)
    x.verify(x.CONTROL,x.CONTROL_SHA)
    c = x.module('tail_launch_prior_control',x.CONTROL/'control.py')
    h = c.helpers()
    x.require(not os.path.lexists(x.ROOT), 'existing_stage_reconcile_without_retry')
    x.require(not h.command(['git','status','--porcelain'],cwd=x.REPO).strip(), 'dirty_cluster_checkout')
    x.require(h.command(['git','rev-parse','HEAD'],cwd=x.REPO).strip() ==
              'de9de39a1a72db40485878452561affe2fbd24aa', 'cluster_commit')
    h.verify_package(x.SAMPLER,x.SAMPLER_SHA)
    h.verify_file(x.PREVIOUS,x.PREVIOUS_SHA)
    accounting = h.command(['sacct','-j','13571,13630','-n','-P','-o',
                            'JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    for job in ('13571','13630'):
        h.terminal_parent(accounting,job)
    qpath=x.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qpath,h.QUOTA_HELPER_SHA)
    q=x.module('tail_launch_quota',qpath)
    mount=q.mount_identity(x.W)
    started=time.monotonic()
    raw=h.command(['quota','-v','-w'],timeout=30)
    quota=q.parse_login_quota(raw)
    # Reuse the conservative 128 MiB output allowance and 8 GiB reserve.
    c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j'])
    x.require(all(not line.split('|')[1].startswith('third-') for line in queue.splitlines()),
              'project_job_active_reconcile')
    x.verify(x.FOLDER,args.package_sha256)
    c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-tail-census-v1',
        '--time=60','--output',str(x.ROOT/'slurm-%j.out'),'--error',str(x.ROOT/'slurm-%j.err'),
        str(x.FOLDER/'census.sbatch'),str(x.FOLDER),args.package_sha256]
    intent={'schema':'third-tail-census-launch-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'package_sha256':args.package_sha256,'root':str(x.ROOT),'command':command,
        'fresh_accounting':accounting,'fresh_queue':queue,'quota':quota,'quota_raw':raw,'mount':mount,
        'reserve_bytes':c.RESERVE,'allowance_bytes':c.ALLOWANCE,'quota_is_not_reservation':True,
        'model_inference_calls':0,'human_annotation_calls':0,'automatic_retry':False}
    os.umask(0o077)
    x.ROOT.mkdir(mode=0o700,exist_ok=False)
    print(json.dumps(c.submit_once(h,x.ROOT,command,intent),sort_keys=True))

if __name__ == '__main__':
    main()
