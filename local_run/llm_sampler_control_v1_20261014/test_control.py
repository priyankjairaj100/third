"""Software fixtures only: no Slurm, entropy draws or dataset access."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import control as c

def args(mode='qualification'):
    return SimpleNamespace(mode=mode,expected_tests=1,time_limit_minutes=20,
      sampler_package_sha256='a'*64,lock_package_sha256='b'*64,control_package_sha256='c'*64,
      sampling_lock_sha256=None,qualification_result_sha256=None,qualification_review_sha256=None)

class ControllerTests(unittest.TestCase):
    def test_qualification_cannot_consume_production_lock(self):
        a=args();c.validate_args(a)
        a.sampling_lock_sha256='d'*64
        with self.assertRaises(ValueError):c.validate_args(a)
    def test_selection_requires_three_external_bindings(self):
        a=args('selection');a.time_limit_minutes=120
        with self.assertRaises(ValueError):c.validate_args(a)
        a.sampling_lock_sha256='d'*64;a.qualification_result_sha256='e'*64;a.qualification_review_sha256='f'*64
        c.validate_args(a)
        a.qualification_review_sha256='bad'
        with self.assertRaises(ValueError):c.validate_args(a)
    def test_mode_time_limits_not_interchangeable(self):
        a=args();a.time_limit_minutes=240
        with self.assertRaises(ValueError):c.validate_args(a)
        a=args('selection');a.time_limit_minutes=20
        with self.assertRaises(ValueError):c.validate_args(a)
    def test_quota_boundary_staleness_and_mount(self):
        mount={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'}
        c.quota_gate(c.RESERVE+c.ALLOWANCE,120,mount)
        for amount,age,m in ((c.RESERVE+c.ALLOWANCE-1,0,mount),(c.RESERVE+c.ALLOWANCE,121,mount),(c.RESERVE+c.ALLOWANCE,0,{})):
            with self.assertRaises(ValueError):c.quota_gate(amount,age,m)
    def test_queue_refuses_own_pending_or_previous_stage(self):
        for row in ('13556|anything','999|third-llm-sampler-qualification','999|third-llm-sampler-selection'):
            with self.assertRaises(ValueError):c.queue_clear(row)
        c.queue_clear('999|unrelated-project')
    def test_command_separates_evidence_and_executing_source(self):
        cmd=c.child_command(args())
        self.assertEqual(cmd[cmd.index('--repo')+1],str(c.REPO))
        self.assertEqual(cmd[cmd.index('--protocol-evidence')+1],str(c.EVIDENCE))
        self.assertNotIn('--sampling-lock',cmd)
        a=args('selection');a.sampling_lock_sha256='d'*64;a.qualification_result_sha256='e'*64;a.qualification_review_sha256='f'*64
        cmd=c.child_command(a)
        self.assertEqual(cmd[cmd.index('--sampling-lock')+1],str(c.LOCK))
        self.assertEqual(cmd[-1],a.qualification_review_sha256)
    def test_submission_failure_is_single_and_retained(self):
        class H:
            def __init__(self):self.records=[]
            def write_exclusive(self,path,value):self.records.append(dict(value))
        h=H()
        with patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='failed')) as run:
            with self.assertRaises(ValueError):c.submit_once(h,c.QUAL_ROOT,['sbatch'],{})
        self.assertEqual(run.call_count,1);self.assertEqual(len(h.records),2)
    def test_uncertain_submission_never_retried(self):
        class H:
            def __init__(self):self.records=[]
            def write_exclusive(self,path,value):self.records.append(dict(value))
        h=H()
        with patch.object(c.subprocess,'run',side_effect=TimeoutError('uncertain')) as run:
            with self.assertRaises(TimeoutError):c.submit_once(h,c.QUAL_ROOT,['sbatch'],{})
        self.assertEqual(run.call_count,1)
        self.assertEqual(h.records[-1]['submission_outcome'],'unknown_reconcile_without_retry')

if __name__=='__main__':unittest.main()
