import importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
s=importlib.util.spec_from_file_location('tested_annotation_control',Path(__file__).with_name('control.py'));c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
class ControlTests(unittest.TestCase):
    def args(self,mode):return SimpleNamespace(mode=mode,control_package_sha256='a'*64,resource_plan_sha256='b'*64,admission_sha256=None if mode=='qualification' else 'c'*64,activation_sha256=None if mode=='qualification' else 'd'*64)
    def resources(self):return {'packages':{'runner':{'sha256':'e'*64}},'request_manifest':{'sha256':'f'*64}}
    def test_unique_roots(self):
        roots=[c.root_for(m) for m in ('qualification','qwen','mistral')];self.assertEqual(len(set(roots)),3)
        with self.assertRaises(ValueError):c.root_for('validation')
    def test_no_activation_in_qualification(self):
        a=self.args('qualification');c.check_args(a);a.activation_sha256='e'*64
        with self.assertRaises(ValueError):c.check_args(a)
    def test_requires_production_activation(self):
        a=self.args('qwen');c.check_args(a);a.activation_sha256=None
        with self.assertRaises(ValueError):c.check_args(a)
    def test_sha_argument_validation(self):
        a=self.args('qualification');a.control_package_sha256='x'*64
        with self.assertRaises(ValueError):c.check_args(a)
    def test_queue_refuses_any_project_job(self):
        c.queue_clear('123|unrelated\n')
        for raw in ('124|third-llm-infer-qwen','125|third-e5-other','invalid'):
            with self.assertRaises(ValueError):c.queue_clear(raw)
    def test_quota_exact_boundary_and_age(self):
        mount={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'};allowance=2*c.FAMILY_ALLOWANCE
        c.quota_gate(c.RESERVE+allowance,120,mount,allowance)
        for available,age in ((c.RESERVE+allowance-1,0),(c.RESERVE+allowance,120.001)):
            with self.assertRaises(ValueError):c.quota_gate(available,age,mount,allowance)
        with self.assertRaises(ValueError):c.quota_gate(c.RESERVE+allowance,0,{},allowance)
    def test_qualification_child_no_activation(self):
        cmd=c.child_command(self.args('qualification'),self.resources());self.assertEqual(cmd[2],str(c.PACKAGES['runner']/'qualify.py'));self.assertNotIn('--activation-dir',cmd);self.assertIn('--mistral-weights-manifest',cmd)
    def test_family_children_exact_fixed_paths(self):
        for alias in ('qwen','mistral'):
            cmd=c.child_command(self.args(alias),self.resources());d=dict(zip(cmd[3::2],cmd[4::2]));self.assertEqual(d['--model-dir'],str(c.MODELS[alias]));self.assertEqual(d['--output-dir'],str(c.root_for(alias)/'run/output'));self.assertEqual(d['--activation-dir'],str(c.ACTIVATION));self.assertNotIn('--retry',cmd)
if __name__=='__main__':unittest.main()
