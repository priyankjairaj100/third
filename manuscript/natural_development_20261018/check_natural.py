"""Check the bounded manuscript delta and replay source-bound aggregate evidence."""
import argparse,hashlib,json,pathlib,subprocess,sys,tempfile,zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(command,cwd):
 r=subprocess.run(command,cwd=cwd,text=True,capture_output=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr or r.stdout)
 return json.loads(r.stdout)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
 root=a.root.resolve();here=root/'manuscript/natural_development_20261018'
 delta=json.loads((here/'delta.json').read_text())
 source=root/'manuscript/third_working_draft.tex'
 assert sha(source)==delta['new_source_sha256']
 archive=root/'manuscript/release_20261018_tmlr_v2.zip'
 assert sha(archive)=='17b17e3388a24a09805ad675cc7b9297247e7b8ed0dda9abbe140a0295b5e676'
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  old=z.read('manuscript/third_working_draft.tex')
  assert hashlib.sha256(old).hexdigest()==delta['prior_source_sha256']
  revised=old.decode()
  for change in delta['changes']:
   assert revised.count(change['old'])==1
   revised=revised.replace(change['old'],change['new'])
  assert revised.encode()==source.read_bytes()
  with tempfile.TemporaryDirectory(prefix='third-natural-history-') as tmp:
   z.extractall(tmp)
   historical=run([sys.executable,'-B','manuscript/tmlr_layout_20261018/check_layout.py','--root',tmp,'--output',str(pathlib.Path(tmp)/'layout_replay.json')],tmp)
 bundle=root/'local_run/postnegative_structure_results_v1_20261018'
 assert sha(bundle/'MANIFEST.json')==delta['stage1_public_manifest_sha256']
 stage1=run([sys.executable,'-B',str(bundle/'replay_aggregates.py')],root)
 assert stage1['passed'] and stage1['all51cuts']
 review=json.loads((here/'delta_review_stage1.json').read_text())
 assert review['passed'] and not review['findings']
 assert review['bindings']['current_manuscript']['sha256']==sha(source)
 assert review['bindings']['delta']['sha256']==sha(here/'delta.json')
 delivery=json.loads((here/'supplement_delivery.json').read_text())
 supplement=here/delivery['archive']['file'];assert sha(supplement)==delivery['archive']['sha256']
 with zipfile.ZipFile(supplement) as z:
  assert z.testzip() is None
  with tempfile.TemporaryDirectory(prefix='third-natural-supplement-') as tmp:
   z.extractall(tmp)
   anonymous=run([sys.executable,'-B',str(pathlib.Path(tmp)/'check_all.py')],root)
   assert anonymous['passed'] and anonymous['manuscript_source_sha256']==sha(source)
 result={'schema':'third-natural-development-source-check-v1','passed':True,'source_sha256':sha(source),'exact_author_delta_reproduced':True,'changes':len(delta['changes']),'historical_layout_tmlr_and_theory_replay_passed':historical['passed'],'stage1_aggregate_replay':stage1,'review_report_sha256':sha(here/'delta_review_stage1.json'),'scientific_scope':'Natural development structure only; no semantic validation, learner outcome, comparative efficiency or original empirical completion claim.','private_observations_authenticated':False,'stage2_results_included':delta['stage2_results_included'],'historical_reviews_rebound':False}
 result['anonymous_supplement_replay_passed']=anonymous['passed']
 result['anonymous_supplement_sha256']=sha(supplement)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
