import unittest
from types import SimpleNamespace
import launch as c

class AdapterTests(unittest.TestCase):
    def good(self):return '13574|COMPLETED|0:0|2634||csis.cn3\n13574.batch|COMPLETED|0:0|2634|2307420K|csis.cn3\n'
    def test_completed_gpu_accepted(self):self.assertEqual(c.gpu_qwen_terminal(self.good(),'13574')['NodeList'],'csis.cn3')
    def test_cpu_node_rejected(self):
        with self.assertRaises(ValueError):c.gpu_qwen_terminal(self.good().replace('csis.cn3','csis.cn1'),'13574')
    def test_failure_running_exit_rejected(self):
        for before,after in [('COMPLETED','FAILED'),('COMPLETED','RUNNING'),('0:0','1:0')]:
            with self.subTest(after=after), self.assertRaises(ValueError):c.gpu_qwen_terminal(self.good().replace(before,after),'13574')
    def test_duplicate_unrelated_missing_rejected(self):
        for raw in [self.good()+self.good(),self.good()+'2|COMPLETED|0:0|1||csis.cn3\n','']:
            with self.assertRaises(ValueError):c.gpu_qwen_terminal(raw,'13574')
    def test_exact_job_runtime_required(self):
        for raw,job in [(self.good(),'13575'),(self.good().replace('2634','2635'),'13574')]:
            with self.assertRaises(ValueError):c.gpu_qwen_terminal(raw,job)
    def test_cpu_checks_delegated_unchanged(self):
        calls=[]
        def original(raw,job,success):calls.append((raw,job,success));return 'unchanged'
        p=c.RoleAwareHelpers(SimpleNamespace(terminal_parent=original))
        for job in ['13572','13573','13623']:
            self.assertEqual(p.terminal_parent('original-accounting',job),'unchanged')
        self.assertEqual(len(calls),3);self.assertEqual(p.qwen_checks,0)
    def test_no_failure_override(self):
        p=c.RoleAwareHelpers(SimpleNamespace())
        with self.assertRaises(ValueError):p.terminal_parent(self.good(),'13574',success=False)
        self.assertEqual(p.qwen_checks,0)

if __name__=='__main__':unittest.main()
