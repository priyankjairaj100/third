#!/usr/bin/env python3
"""Package only strict-mode code, natural fixture, proofs and executed evidence."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
DEST=PROJECT/'output/counterfactual_curation_strict_mode.zip'

def main():
    names=['README_strict_mode.txt','requirements.txt','run_canonical_validation.py',
      'run_engineering_pilot.py','audit_certified_ridge.py','audit_certificates_independent.py',
      'audit_exact_chart_independent.py','audit_canonical_loader_independent.py',
      'check_exact_algebra.py','canonical_theory_addendum.tex','exact_chart_algorithm_notes.txt',
      'certified_ridge_contract.txt','build_strict_report.py','package_strict_mode.py',
      'DATA_ACCESS_NOTE.txt','preview_acquisition_manifest.json',
      'data/civil_comments_engineering_preview.jsonl','data/civil_comments_preview_raw.json',
      'results/certified_ridge_audit.json']
    paths=[ROOT/name for name in names]
    paths+=sorted((ROOT/'ccu').glob('*.py'))
    paths+=sorted((ROOT/'ccu/native').iterdir())
    paths+=sorted((ROOT/'canonical_results').iterdir())
    required=ROOT/'canonical_results/final_run_review.txt'
    if required not in paths or not required.is_file():raise RuntimeError('final audit missing')
    mapping={str(p.relative_to(PROJECT)):p for p in paths if p.is_file()}
    for ext in ['tex','pdf']:
        p=PROJECT/f'output/pdf/counterfactual_curation_strict_mode.{ext}'
        mapping[str(p.relative_to(PROJECT))]=p
    for p in mapping.values():
        if not p.is_file():raise FileNotFoundError(p)
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(mapping.items())}
    manifest={'scope':'strict canonical exact mode; nonconfirmatory natural-text engineering checks',
      'all_execution_local':True,'no_synthetic_empirical_data':True,
      'native_platform':'Linux LP64; verified GMP 6.3.0 ABI; Python Fraction fallback provided',
      'omitted':['CC-News text','earlier floating experiment outputs','Python caches','temporary TeX runtime'],
      'files':hashes}
    man=ROOT/'STRICT_PACKAGE_MANIFEST.json';man.write_text(json.dumps(manifest,indent=2)+'\n')
    sums=ROOT/'STRICT_SHA256SUMS.txt';sums.write_text(''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    mapping[str(man.relative_to(PROJECT))]=man;mapping[str(sums.relative_to(PROJECT))]=sums
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,path in sorted(mapping.items()):z.write(path,name)
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    print(json.dumps({'file':str(DEST),'bytes':DEST.stat().st_size,'files':len(mapping),
      'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
