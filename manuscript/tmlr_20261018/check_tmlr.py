"""Author checks for the TMLR revision; no experiment or model inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

PREVIOUS_ARCHIVE_SHA = 'd9567681911dc8f7bcef27daf915e4c943ae8d42869ab4e348ffb3e2ab5f2443'
STYLE_SHA = '816214ff5919aa457b6b443bee52b15d9561421417b7f8a50cc84651519f0002'

def bind(path, root):
    b=path.read_bytes()
    return {'path':path.relative_to(root).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def normalize(text):
    return re.sub(r'\s+',' ',text.replace('\\citep{','\\cite{')).strip()

def math_blocks(text):
    envs=r'equation\*?|align\*?|theorem|proposition|corollary|proof'
    found=re.findall(r'\\begin\{('+envs+r')\}(.*?)\\end\{\1\}',text,re.S)
    return Counter((kind,normalize(body)) for kind,body in found)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--replay-historical',action='store_true')
    a=ap.parse_args(); root=a.root.resolve()
    path=root/'manuscript/third_working_draft.tex'
    source=path.read_text()
    archive=root/'manuscript/release_20261018_v1.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==PREVIOUS_ARCHIVE_SHA
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        before=z.read('manuscript/third_working_draft.tex').decode()
        old_evidence=json.loads(z.read('manuscript/evidence_20261016.json'))
        old_build=json.loads(z.read('manuscript/BUILD_STATUS.json'))
        assert old_build['source_binding']['sha256']==hashlib.sha256(before.encode()).hexdigest()
        assert old_evidence['manuscript']['sha256']==hashlib.sha256(before.encode()).hexdigest()
    embedded=re.search(r'\\begin\{filecontents\*\}\[overwrite\]\{tmlr.sty\}\n(.*?)\\end\{filecontents\*\}',source,re.S)
    assert embedded and hashlib.sha256(embedded[1].encode()).hexdigest()==STYLE_SHA
    content=source[embedded.end():]
    assert '\\documentclass[10pt]{article}' in content and '\\usepackage{tmlr}' in content
    assert '\\usepackage[accepted]{tmlr}' not in content and '\\usepackage[preprint]{tmlr}' not in content
    assert 'twocolumn' not in content and '{geometry}' not in content
    assert '\\hypersetup{pdfauthor={}' in content
    assert not re.search(r'/Users/|/nfs_home/|priyank|poonam|csis\.|github\.com/priyank',content,re.I)
    bib=re.findall(r'\\bibitem\[([^\]]+)\]\{([^}]+)\}',content)
    keys=[k for _,k in bib]
    citations={key for group in re.findall(r'\\cite(?:p|t)?\{([^}]+)\}',content) for key in group.split(',')}
    labels=re.findall(r'\\label\{([^}]+)\}',content)
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',content)
    assert len(keys)==len(set(keys))==16 and set(keys)==citations
    assert [v.split('(')[0].lower() for v,k in bib]==sorted(v.split('(')[0].lower() for v,k in bib)
    assert len(labels)==len(set(labels)) and set(refs)<=set(labels)
    assert content.index('\\end{thebibliography}')<content.index('\\appendix')
    assert content.index('\\label{sec:design}')>content.index('\\appendix')
    tab=content[content.index('\\begin{table}'):content.index('\\end{table}')]
    assert tab.index('\\caption')<tab.index('\\begin{tabular}')
    assert math_blocks(before)==math_blocks(content),'mathematical statement/proof change needs substantive review'
    for text in ['remaining budget','strict','full-ID lookup','O(Q+\\Phi_0+sM_0)',
                 'No retuning follows a failure','no human-ground-truth claim',
                 'their six-way categories']:
        # The capitalized sentence is preserved in the current results section.
        assert text.lower() in content.lower(),text
    historical_sources=[]
    for b in old_evidence['source_bindings']:
        assert bind(root/b['path'],root)==b,'historical source drift: '+b['path']
        historical_sources.append(b)
    n=json.loads((root/'local_run/local_llm_execution_20261014/llm_v4_negative_diagnostics_20261016.json').read_text())
    cuts={Fraction(r['tau']['numerator'],r['tau']['denominator']):r for r in n['grid_ledger']}
    for tau in ['.90','.95','.98','.99','.995']:
        row=cuts[Fraction(tau)]
        vals=[100*Fraction(row[k]['numerator'],row[k]['denominator']) for k in ['conservative_precision','missing_uncertain_sensitivity_upper']]
        assert f'{tau} & {row["sample_support"]} & {float(vals[0]):.2f} & {float(vals[1]):.2f}' in content
    for version in ['v3','v4']:
        d=json.loads((root/f'local_run/local_llm_execution_20261014/llm_selection_{version}_terminal_result.json').read_text())
        assert d['execution_complete'] and not d['selection_passed'] and d['selected_tau'] is None
        assert not d['independent_validation_performed']
    replay=None
    if a.replay_historical:
        with tempfile.TemporaryDirectory(prefix='third-historical-replay-') as temp:
            with zipfile.ZipFile(archive) as z: z.extractall(temp)
            result=subprocess.run([sys.executable,'-B','manuscript/build_evidence.py'],cwd=temp,text=True,capture_output=True,timeout=90)
            assert result.returncode==0,result.stderr
            replay=json.loads(result.stdout)
    current_review_path=root/'manuscript/tmlr_20261018/delta_review.json'
    current_review=json.loads(current_review_path.read_text())
    assert current_review['reviewed_manuscript']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert current_review['status']=='no_unresolved_confirmed_defect_within_reviewed_delta'
    assert all(f['status']=='resolved' for f in current_review['findings'])
    supplement=root/'manuscript/tmlr_20261018/tmlr_anonymous_supplement_v1.zip'
    assert hashlib.sha256(supplement.read_bytes()).hexdigest()=='c6ab12aa23da1563d5f659f2406ce2a6985085a0c55b7de531119e9d421dff26'
    report={
        'schema':'third-tmlr-source-check-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Author source/arithmetic/version checks, not a new scientific or visual review.',
        'manuscript':bind(path,root),'historical_archive':bind(archive,root),
        'official_style_embedded_sha256':STYLE_SHA,'official_style_unmodified':True,
        'delta_review':bind(current_review_path,root),'delta_review_findings_resolved':True,
        'anonymous_supplement':bind(supplement,root),
        'historical_review_rebound':False,'historical_sources_verified':historical_sources,
        'historical_checker_replay':replay,'bibliography_entries':len(keys),
        'references_checked':len(refs),'theorem_proof_display_blocks_preserved':sum(math_blocks(before).values()),
        'anonymity_source_scan_passed':True,'calibration_table_exact_arithmetic_passed':True,
        'both_negative_calibrations_preserved':True,'frozen_experiments_launched':False,
        'annotation_mode':'fully LLM; zero human pair annotation or pair review',
        'original_human_and_empirical_gates_unmet':True,
        'visual_review_verified':False,'full_text_comparison_gaps_closed':False,
        'submission_ready':False,'author_attestations_supplied':False,
        'source_check_passed':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['historical_sources_verified']}))

if __name__=='__main__':main()
