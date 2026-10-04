#!/usr/bin/env python3
"""Portable Civil-only empirical development package with integrity checks."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
DEST=PROJECT/'output/counterfactual_curation_empirical_phase2.zip'

def main():
    paths=[p for p in (ROOT/'phase2').rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name not in ['PACKAGE_MANIFEST.json','SHA256SUMS.txt']]
    paths+=list((ROOT/'ccu').glob('*.py'))+list((ROOT/'ccu/native').iterdir())
    paths+=[ROOT/p for p in ['requirements.txt','CURRENT_STATUS.json','DATA_ACCESS_NOTE.txt',
      'preview_acquisition_manifest.json','data/civil_comments_engineering_preview.jsonl',
      'data/civil_comments_preview_raw.json','canonical_theory_addendum.tex','certified_ridge_contract.txt']]
    paths+=[PROJECT/p for p in ['output/empirical_program/study_design.json',
      'output/empirical_program/claim_ledger.csv','output/empirical_program/validate_design.py',
      'output/pdf/counterfactual_curation_empirical_protocol.pdf',
      'output/pdf/counterfactual_curation_empirical_phase2.tex',
      'output/pdf/counterfactual_curation_empirical_phase2.pdf']]
    mapping={str(p.relative_to(PROJECT)):p for p in paths if p.is_file()}
    for p in mapping.values():
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
    hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sorted(mapping.items())}
    manifest={'scope':'completed local Civil-only task-consequence development; primary study not started',
      'no_synthetic_empirical_data':True,'no_remote_compute':True,'files':hashes,
      'omitted':['CC-News article text','Python caches','previous experiment packages','temporary TeX runtime'],
      'portable_replay':'main experiment and result audit need only included Civil data; inventory will record news as unavailable',
      'native_backend':'included Linux LP64 GMP6.3.0 helper and source, with exact Python fallback'}
    m=ROOT/'phase2/PACKAGE_MANIFEST.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
    h=ROOT/'phase2/SHA256SUMS.txt';h.write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
    mapping[str(m.relative_to(PROJECT))]=m;mapping[str(h.relative_to(PROJECT))]=h
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for k,v in sorted(mapping.items()):z.write(v,k)
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        for k,v in hashes.items():assert hashlib.sha256(z.read(k)).hexdigest()==v
        assert not any('/data/cc_news' in k for k in z.namelist())
    print(json.dumps({'file':str(DEST),'files':len(mapping),'bytes':DEST.stat().st_size,
      'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
