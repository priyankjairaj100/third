"""Resumption guards only; no remote actions or model execution."""
import sys, unittest
from pathlib import Path
from unittest.mock import patch

ACTIVE = Path('/Users/priyankjairaj/Documents/Codex/2026-10-11/see-i-am-working-on-a/work/resume-20261014/sampling_v1')
sys.path.append(str(ACTIVE))
import run_storage_resume_v2 as s

class Gates(unittest.TestCase):
    def base(self):
        return {'project_queue':'', 'next_stage_slots':{'qwen':{'exists':False,'intent_exists':False,'receipt_exists':False}},
                'quota':{'available_soft_bytes':12<<30},'mistral_required_bytes':(8<<30)+(96<<20),
                'qwen_accounting':'13574|COMPLETED|0:0|2634|csis.cn3'}
    def test_clear(self):
        s.validate_resumption(self.base())
    def test_occupied(self):
        p=self.base();p['next_stage_slots']['qwen']['intent_exists']=True
        with self.assertRaises(ValueError):s.validate_resumption(p)
    def test_quota(self):
        p=self.base();p['quota']['available_soft_bytes']=8<<30
        with self.assertRaises(ValueError):s.validate_resumption(p)
    def test_active(self):
        p=self.base();p['project_queue']='1|third-job|RUNNING'
        with self.assertRaises(ValueError):s.validate_resumption(p)
    def test_incomplete(self):
        p=self.base();p['qwen_accounting']='13574|RUNNING|0:0|200|csis.cn3'
        with self.assertRaises(ValueError):s.validate_resumption(p)
    def test_reuses_qwen_without_model_call(self):
        runner=s.StorageResume(Path('/unused'));runner.event=lambda *a,**k:None
        def disallowed(*a,**k):raise AssertionError('unexpected execution')
        runner.local=runner.remote=disallowed
        value={'passed':True,'execution_passed':True,'job_id':'13574','status_counts':{'valid':99}}
        with patch.object(s,'read',return_value=value), patch.object(s,'bind',return_value={'sha256':s.OLD_REVIEW_SHA}):
            self.assertEqual(runner.family('qwen'),value)

if __name__=='__main__':unittest.main()
