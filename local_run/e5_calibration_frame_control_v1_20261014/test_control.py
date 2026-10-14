"""Software fixtures for single-submission, quota and bounded-output safety."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import launch as c
import runner


class Controls(unittest.TestCase):
    def test_prior_jobs_and_duplicate_stage_block(self):
        for row in ('13498_31|other','13546|other','13547|other','999|third-e5-frame-v1'):
            with self.assertRaises(ValueError):c.queue_clear(row)
        c.queue_clear('999|unrelated-project')

    def test_malformed_queue_blocks(self):
        with self.assertRaises(ValueError):c.queue_clear('not-valid')

    def test_exact_quota_and_age_boundary(self):
        mount={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'}
        c.quota_gate(c.RESERVE+c.ALLOWANCE,120,mount)
        for avail,age,m in ((c.RESERVE+c.ALLOWANCE-1,0,mount),(c.RESERVE+c.ALLOWANCE,121,mount),
                            (c.RESERVE+c.ALLOWANCE,0,{})):
            with self.assertRaises(ValueError):c.quota_gate(avail,age,m)

    def test_command_has_full_frame_no_sampling(self):
        cmd=c.frame_command('a'*64,Path('/new/frame'))
        self.assertIn('--assembly-root',cmd);self.assertIn('--qualification-sha256',cmd)
        self.assertEqual(cmd[-2:],['--out','/new/frame'])
        self.assertNotIn('selection',cmd)

    def test_test_console_capped_by_bytes(self):
        log=runner.CappedText(io.StringIO(),3)
        log.write('é')
        with self.assertRaises(ValueError):log.write('é')

    def test_child_capture_retains_both_streams(self):
        out,err=io.BytesIO(),io.BytesIO()
        rc,size,exceeded=runner.child_capture([sys.executable,'-c','import sys;print("ok");print("err",file=sys.stderr)'],out,err,100)
        self.assertEqual((rc,size,exceeded),(0,7,False))
        self.assertEqual(out.getvalue(),b'ok\n');self.assertEqual(err.getvalue(),b'err\n')

    def test_child_capture_aborts_over_cap(self):
        out,err=io.BytesIO(),io.BytesIO()
        rc,size,exceeded=runner.child_capture([sys.executable,'-c','print("x"*100000)'],out,err,100)
        self.assertTrue(exceeded);self.assertEqual(size,100)

    def test_failed_submission_not_retried(self):
        class Helper:
            def __init__(self):self.records=[]
            def write_exclusive(self,path,value):self.records.append((path,dict(value)))
        helper=Helper()
        with patch.object(c.subprocess,'run',return_value=type('R',(),dict(returncode=1,stdout='',stderr='failed'))()) as run:
            with self.assertRaises(ValueError):c.submit_once(helper,['sbatch'],{'intent':True})
        self.assertEqual(run.call_count,1);self.assertEqual(len(helper.records),2)

    def test_unknown_submission_retained_not_retried(self):
        class Helper:
            def __init__(self):self.records=[]
            def write_exclusive(self,path,value):self.records.append(dict(value))
        helper=Helper()
        with patch.object(c.subprocess,'run',side_effect=TimeoutError('uncertain')) as run:
            with self.assertRaises(TimeoutError):c.submit_once(helper,['sbatch'],{})
        self.assertEqual(run.call_count,1)
        self.assertEqual(helper.records[-1]['submission_outcome'],'unknown_reconcile_without_retry')


if __name__=='__main__':unittest.main()
