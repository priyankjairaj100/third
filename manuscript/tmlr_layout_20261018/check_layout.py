"""Verify layout-only source delta and replay the unchanged historical TMLR package."""
import argparse,hashlib,json,pathlib,subprocess,sys,tempfile,zipfile
OLD='ecd1f6690fc1d384060073284f9477b401944059082972776cc1b3fc8928404c'
NEW='5c40003b70b626e995b7075ac4ff1d6ae178fe13b06ea8e085262ab7c90b75c9'
ARCHIVE='5059b6aa4a8f66e7aae56a8b9b9c18a608470a7aa5f3744106408330f1721dc9'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
 source=(a.root/'manuscript/third_working_draft.tex').read_bytes();assert hashlib.sha256(source).hexdigest()==NEW
 archive=a.root/'manuscript/release_20261018_tmlr_v1.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;old=z.read('manuscript/third_working_draft.tex');assert hashlib.sha256(old).hexdigest()==OLD
  before=old.decode();after=source.decode();left='The active representation is \\texttt{intfloat/multilingual-e5-base}';right='bind tokenizer files'
  b0=before.index(left);b1=before.index(right,b0);a0=after.index(left);a1=after.index(right,a0)
  assert before[:b0]==after[:a0] and before[b1:]==after[a1:]
  hashes=['d128750597153bb5987e10b1c3493a34e5a4502a','a09a35458c702b33eeacc393d103063234e8bc28','04d8a90549d23fc6bd7f642064003592df51e9b3']
  for h in hashes:assert before[b0:b1].count(h)==after[a0:a1].count(h)==1
  with tempfile.TemporaryDirectory(prefix='third-layout-historical-') as td:
   z.extractall(td)
   r=subprocess.run([sys.executable,'-B','manuscript/tmlr_20261018/check_tmlr.py','--root',td,'--output',str(pathlib.Path(td)/'replay.json'),'--replay-historical'],cwd=td,capture_output=True,text=True,timeout=120)
   assert r.returncode==0,r.stderr
   historical=json.loads((pathlib.Path(td)/'replay.json').read_text())
 report={'schema':'third-tmlr-layout-delta-check-v1','passed':True,'source_sha256':NEW,'prior_source_sha256':OLD,'change_scope':'Only revision-attribution paragraph reformatted into three model revision lines; all other source bytes identical','model_revision_hashes_preserved':True,'historical_tmlr_archive_replay_passed':True,'historical_bibliography_entries':historical['bibliography_entries'],'historical_theorem_proof_blocks_preserved':historical['theorem_proof_display_blocks_preserved'],'native_pdf_equivalence_claim':False,'scientific_review_rebound':False}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
