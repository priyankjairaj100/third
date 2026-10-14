import unittest
import node_policy as p
class NodePolicyTests(unittest.TestCase):
    def test_only_reviewed_compute_nodes(self):
        for node in p.ALLOWED:self.assertTrue(p.acceptable_node('gpu',node))
        for node in ('csis.mn1','csis.cn2','csis.cn5','None assigned','csis.cn1,csis.cn4'):
            self.assertFalse(p.acceptable_node('gpu',node))
        self.assertTrue(p.acceptable_node('csis.cn1','csis.cn1'))
        self.assertFalse(p.acceptable_node('csis.cn1','csis.cn4'))
    def test_single_gpu_without_required_node_list(self):
        f=p.gpu_flags({'allowed_nodes':list(p.ALLOWED),'host_memory_bytes':48*1024**3,'slurm_time_seconds':21600})
        self.assertIn('--nodes=1',f);self.assertIn('--gres=gpu:a100-80gb:1',f)
        self.assertFalse(any(x.startswith('--nodelist') for x in f))
        self.assertIn('--exclude=csis.mn1,csis.cn2',f)
    def test_foreign_node_pool_is_rejected(self):
        with self.assertRaises(ValueError):p.gpu_flags({'allowed_nodes':['csis.mn1']})
