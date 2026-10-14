import argparse,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tested_postprocess_control',HERE/'control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
with patch.dict(sys.modules,{'control':c}):
    spec=importlib.util.spec_from_file_location('tested_postprocess_runner',HERE/'runner.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class PostprocessTests(unittest.TestCase):
    def args(self,mode='qwen'):
        return argparse.Namespace(mode=mode,package_sha256='a'*64,resource_plan_sha256='b'*64,admission_sha256='c'*64,activation_sha256='d'*64,result_sha256='e'*64 if mode!='statistics' else None,qwen_result_sha256='e'*64 if mode=='statistics' else None,mistral_result_sha256='f'*64 if mode=='statistics' else None,qwen_audit_sha256='1'*64 if mode=='statistics' else None,mistral_audit_sha256='2'*64 if mode=='statistics' else None)
    def ec(self):
        return argparse.Namespace(PYTHON=Path('/python'),PACKAGES={'integration':Path('/integration')},ADMISSION=Path('/admission'),ACTIVATION=Path('/activation'),MODELS={'qwen':Path('/qwen'),'mistral':Path('/mistral')},root_for=lambda alias:Path('/families')/alias)
    def test_exact_modes_and_pins(self):
        for mode in ('qwen','mistral','statistics'):c.check_args(self.args(mode))
        a=self.args();a.qwen_audit_sha256='a'*64
        with self.assertRaises(ValueError):c.check_args(a)
        a=self.args('statistics');a.mistral_audit_sha256=None
        with self.assertRaises(ValueError):c.check_args(a)
    def test_absent_only_family_result(self):
        a=self.args();a.result_sha256='absent';c.check_args(a)
        a.admission_sha256='absent'
        with self.assertRaises(ValueError):c.check_args(a)
    def test_roots_separate_and_fixed(self):
        roots=[c.root_for(x) for x in ('qwen','mistral','statistics')];self.assertEqual(len(set(roots)),3)
        self.assertTrue(all('/run/output' not in str(x) for x in roots))
        with self.assertRaises(ValueError):c.root_for('validation')
    def test_audit_command_no_weights_or_generation(self):
        for alias in ('qwen','mistral'):
            cmd=c.child_command(self.ec(),self.args(alias));d=dict(zip(cmd[3::2],cmd[4::2]));self.assertEqual(cmd[2],'/integration/audit.py');self.assertEqual(d['--tokenizer-path'],'/'+alias);self.assertEqual(d['--root'],'/families/'+alias+'/run/output');self.assertNotIn('--weights-manifest',cmd);self.assertNotIn('--retry',cmd)
    def test_statistics_exact_two_audits(self):
        cmd=c.child_command(self.ec(),self.args('statistics'));d=dict(zip(cmd[3::2],cmd[4::2]));self.assertEqual(cmd[2],'/integration/ingest.py')
        self.assertEqual(d['--qwen-audit-sha256'],'1'*64);self.assertEqual(d['--mistral-audit-sha256'],'2'*64);self.assertNotIn('--tokenizer-path',cmd)
    def test_terminal_success_and_ordinary_failure(self):
        for state,code,rc in (('COMPLETED','0:0',0),('FAILED','2:0',2)):
            row=c.terminal_parent('123|'+state+'|'+code+'|10|12K|csis.cn1\n123.batch|'+state+'|'+code+'|10|12K|csis.cn1\n','123');self.assertTrue(c.accepted_terminal(row,{'returncode':rc,'console_cap_exceeded':False}))
    def test_timeout_missing_and_integrity_failures_not_admitted(self):
        for state,code,rc in (('TIMEOUT','0:15',-15),('FAILED','1:0',1),('FAILED','2:0',0)):
            row=c.terminal_parent('123|'+state+'|'+code+'|10||csis.cn1\n','123');self.assertFalse(c.accepted_terminal(row,{'returncode':rc,'console_cap_exceeded':False}));self.assertFalse(c.accepted_terminal(row,None))
    def test_nonterminal_duplicate_and_foreign_accounting_refused(self):
        good='123|COMPLETED|0:0|10||csis.cn1\n'
        for raw in (good+good,good.replace('COMPLETED','RUNNING'),good+'124|COMPLETED|0:0|10||csis.cn1\n',good.replace('cn1','mn1')):
            with self.assertRaises(ValueError):c.terminal_parent(raw,'123')
    def test_exact_process_command_and_finite_timing(self):
        p={'schema':'third-llm-execution-process-v1','command':['x'],'slurm_job_id':'123','mode':'qwen','child_wall_seconds_through_exit':1.,'retry_attempts':0,'human_annotation_calls':0};c.process_check(p,['x'],'123','qwen')
        for q in (dict(p,command=['y']),dict(p,child_wall_seconds_through_exit=float('nan')),dict(p,retry_attempts=1)):
            with self.assertRaises(ValueError):c.process_check(q,['x'],'123','qwen')
    def test_report_internal_seal(self):
        x={'passed':True,'text':'é'};x['sha256']=hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest();c.check_seal(x);x['passed']=False
        with self.assertRaises(ValueError):c.check_seal(x)
    def test_never_promote_failed_wrapper_or_audit(self):
        self.assertTrue(r.accepted_result(0,False,True,{'passed':True}))
        for args in ((2,False,True,{'passed':True}),(0,True,True,{'passed':True}),(0,False,False,{'passed':True}),(0,False,True,None),(0,False,True,{'passed':False})):
            self.assertFalse(r.accepted_result(*args))
    def test_bound_integration_and_quota_policy(self):
        self.assertEqual(c.INTEGRATION_SHA,'6baf3ffaa5d64304293e382913db8b2943ea2de17fb142a50c54659697d2fedb');self.assertEqual(c.RESERVE,8*1024**3);self.assertEqual(c.ALLOWANCE,32*1024**2);self.assertEqual(r.EXPECTED_TESTS,12)
if __name__=='__main__':unittest.main()
