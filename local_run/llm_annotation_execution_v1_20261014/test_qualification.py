"""CPU admission helper regressions; no real labels, tokenizers or model calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import qualify as Q


def manifest():
    names=['runtime_manifest.json','token_eligibility.json','effective_constraint_config.json',
      'calls/call-0000/response.json']+['calls/call-0000/attempt-0.'+x for x in
      ('input_tokens.private.json','output_tokens.private.json','raw.private.txt','guard_steps.private.json','metrics.json')]
    return {'schema':'third-historical-output-engineering-replay-v1',
      'selection_rule':'Fixed call-0000 from each already exposed pilot; no new generation or empirical labels',
      'source_role':'historical engineering only','families':{a:{'root':'/fixture/'+a,'call':'call-0000',
      'files':{n:{'bytes':1,'sha256':'a'*64} for n in names}} for a in ('qwen','mistral')}}


class QualificationTests(unittest.TestCase):
    def test_nine_bound_historical_artifacts_and_fixed_call(self):
        m=manifest();files=Q.engineering_files(m)
        self.assertEqual(len(files['qwen']),9)
        self.assertEqual(files['mistral']['guard_steps']['path'],'/fixture/mistral/calls/call-0000/attempt-0.guard_steps.private.json')
        for field,value in [('call','call-0001'),('root','../foreign')]:
            bad=copy.deepcopy(m);bad['families']['qwen'][field]=value
            with self.assertRaises(ValueError):Q.engineering_files(bad)
        for name in ('calls/call-0000/attempt-0.metrics.json','calls/call-0000/attempt-0.guard_steps.private.json'):
            bad=copy.deepcopy(m);del bad['families']['mistral']['files'][name]
            with self.assertRaisesRegex(ValueError,'file_set'):Q.engineering_files(bad)
    def test_suite_rejects_zero_skips_failures_and_missing_summary(self):
        for code,stderr in [(0,'Ran 0 tests in 0.001s\n\nOK\n'),(0,'Ran 3 tests in 0.1s\n\nOK (skipped=1)\n'),
                            (1,'Ran 3 tests in 0.1s\n\nOK\n'),(0,'OK\n')]:
            with patch.object(Q.subprocess,'run',return_value=NS(returncode=code,stdout='',stderr=stderr)):
                with self.assertRaisesRegex(ValueError,'suite_failure'):Q.test_suite(Path('/fixture'),{})
        with patch.object(Q.subprocess,'run',return_value=NS(returncode=0,stdout='',stderr='Ran 3 tests in 0.1s\n\nOK\n')):
            receipt,_,_=Q.test_suite(Path('/fixture'),{});self.assertEqual(receipt['tests'],3)
    def test_statistics_test_only_source_root_relocation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);package=root/'tools'/'statistics';package.mkdir(parents=True);repo=root/'actual_repo';repo.mkdir()
            (repo/'sentinel').write_text('frozen source fixture')
            (package/'statistics_core.py').write_text('VERSION="unchanged"\n')
            (package/'test_statistics.py').write_text('from pathlib import Path\nimport unittest\nimport statistics_core as s\nREPO=Path(__file__).parents[2]\nclass T(unittest.TestCase):\n def test_source(self):\n  self.assertEqual((REPO/"sentinel").read_text(),"frozen source fixture")\n  self.assertEqual(s.VERSION,"unchanged")\n')
            files={p.name:Q.R.binding(p) for p in package.iterdir()};(package/'package_manifest.json').write_bytes(Q.R.canonical({'files':files}))
            before={p.name:Q.R.binding(p) for p in package.iterdir()};pin=Q.R.binding(package/'package_manifest.json')['sha256']
            cmd=[sys.executable,'-B',str(Q.R.ROOT/'run_statistics_tests.py'),'--statistics-package',str(package),
                 '--statistics-package-sha256',pin,'--repo',str(repo)]
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr);self.assertIn('Ran 1 test',run.stderr)
            self.assertEqual(before,{p.name:Q.R.binding(p) for p in package.iterdir()})


if __name__=='__main__':unittest.main()
