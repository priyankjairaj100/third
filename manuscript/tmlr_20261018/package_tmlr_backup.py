"""Build a versioned development backup, distinct from anonymous supplement."""
import argparse
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path

def digest(data):return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',required=True,type=Path)
 p.add_argument('--version',required=True)
 a=p.parse_args();root=a.root.resolve()
 assert a.version.isascii() and a.version.replace('_','').isalnum()
 archive=root/'manuscript'/('release_'+a.version+'.zip')
 manifestpath=root/'manuscript'/('release_manifest_'+a.version+'.json')
 assert not archive.exists() and not manifestpath.exists(),'immutable version occupied'
 prior=root/'manuscript/release_20261018_v1.zip'
 assert digest(prior.read_bytes())['sha256']=='d9567681911dc8f7bcef27daf915e4c943ae8d42869ab4e348ffb3e2ab5f2443'
 with zipfile.ZipFile(prior) as z:
  assert z.testzip() is None
  files={n:z.read(n) for n in z.namelist()}
 overlays=['manuscript/third_working_draft.tex','manuscript/BUILD_STATUS.json','manuscript/README.txt','manuscript/REPRODUCIBILITY.txt','manuscript/release_20261018_v1.zip']
 overlays += [p.relative_to(root).as_posix() for p in (root/'manuscript/tmlr_20261018').iterdir() if p.is_file()]
 for name in overlays:files[name]=(root/name).read_bytes()
 manifest={'schema':'third-tmlr-development-backup-v1','created_utc':datetime.now(timezone.utc).isoformat(),
  'version':a.version,'anonymous_submission_package':False,
  'scope':'TMLR source, venue/delta review, anonymous supplement and historical evidence closure. Development backup only; never upload wholesale for blind review.',
  'prior_archive':{'path':prior.relative_to(root).as_posix(),**digest(prior.read_bytes())},
  'files':{n:digest(b) for n,b in sorted(files.items())},
  'private_source_text_or_pair_observations_included':False,'compiled_pdf_included':False,
  'visual_review_verified':False,'original_empirical_program_complete':False,'submitted':False}
 manifestpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 files[manifestpath.relative_to(root).as_posix()]=manifestpath.read_bytes()
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for name,b in sorted(files.items()):
   info=zipfile.ZipInfo(name,(2026,10,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
   z.writestr(info,b)
 with tempfile.TemporaryDirectory(prefix='third-tmlr-backup-check-') as temp:
  temp=Path(temp)
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None and set(z.namelist())==set(files)
   z.extractall(temp)
  for name,b in files.items():assert (temp/name).read_bytes()==b,name
  run=subprocess.run([sys.executable,'-B','manuscript/tmlr_20261018/check_tmlr.py','--root',str(temp),'--output',str(temp/'replay.json'),'--replay-historical'],cwd=temp,text=True,capture_output=True,timeout=120)
  assert run.returncode==0,run.stderr
  replay=json.loads((temp/'replay.json').read_text())
  original=json.loads(files['manuscript/tmlr_20261018/evidence.json'])
  for x in [replay,original]:x.pop('created_utc')
  assert replay==original,'new evidence differs on clean extraction'
 status={'schema':'third-tmlr-backup-build-status-v1','observed_utc':datetime.now(timezone.utc).isoformat(),
  'archive':{'path':archive.relative_to(root).as_posix(),**digest(archive.read_bytes())},
  'manifest':{'path':manifestpath.relative_to(root).as_posix(),**digest(manifestpath.read_bytes())},
  'archive_members':len(files),'exact_member_and_crc_checks_passed':True,
  'clean_extraction_tmlr_and_historical_replay_passed':True,'anonymous_submission_package':False,
  'compiled_pdf_included':False,'visual_review_verified':False,'submission_ready':False}
 (root/'manuscript/RELEASE_BUILD_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status))
if __name__=='__main__':main()
