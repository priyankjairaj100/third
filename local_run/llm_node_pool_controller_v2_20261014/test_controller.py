"""Software fixtures for finite controller dispatch; no remote calls or models."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).parent

def load():
    spec=importlib.util.spec_from_file_location('pipeline_fixture',HERE/'continue_pipeline.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.m=load();self.m.OUT=Path(self.tmp.name)/'run';self.m.p.OUT=self.m.OUT
    def review(self,passed):
        return {'selection_passed':passed,'bindings':{'statistics':{'sha256':'selection-sha'}}}
    def test_negative_selection_never_dispatches_validation(self):
        m=self.m
        with patch.object(m,'reconcile'),patch.object(m,'model_sequence') as seq,patch.object(m.p,'independent_statistics_review',return_value=self.review(False)),patch.object(m.p,'run_cli') as launch,patch.object(m.v,'independent_review') as review:
            m.main();self.assertEqual(seq.call_count,1);launch.assert_not_called();review.assert_not_called()
        self.assertFalse(json.loads((m.OUT/'completion.json').read_text())['validation_run'])
    def test_reconciliation_error_stops_without_retry(self):
        m=self.m
        with patch.object(m,'reconcile',side_effect=RuntimeError('fixture-failure')) as rec,patch.object(m,'model_sequence') as seq:
            with self.assertRaises(RuntimeError):m.main()
            self.assertEqual(rec.call_count,1);seq.assert_not_called()
        self.assertFalse(json.loads((m.OUT/'stopped.json').read_text())['automatic_retry'])
    def test_positive_selection_dispatches_fresh_validation_once_in_order(self):
        m=self.m;order=[]
        def seq(label,*args):order.append(label)
        def cli(label,*args):order.append(label);return {'job_id':'fixture'}
        def wait(label,*args):return {'result':{'planned_requests':440},'result_binding':{'sha256':label+'-sha'}}
        with patch.object(m,'reconcile'),patch.object(m,'model_sequence',side_effect=seq),patch.object(m.p,'independent_statistics_review',return_value=self.review(True)),patch.object(m.p,'run_cli',side_effect=cli),patch.object(m,'await_success',side_effect=wait),patch.object(m.v,'independent_review',return_value={'model_validation_passed':True}) as reviewed:
            m.main();reviewed.assert_called_once()
        self.assertEqual(order,['selection','validation_sampling_launch','validation_requests_launch','validation'])
        self.assertFalse(json.loads((m.OUT/'completion.json').read_text())['project_complete'])
    def test_validation_call_cap_refuses_model_launch(self):
        m=self.m
        with patch.object(m,'reconcile'),patch.object(m,'model_sequence') as seq,patch.object(m.p,'independent_statistics_review',return_value=self.review(True)),patch.object(m.p,'run_cli',return_value={'job_id':'fixture'}),patch.object(m,'await_success',return_value={'result':{'planned_requests':441},'result_binding':{'sha256':'fixture'}}):
            with self.assertRaisesRegex(RuntimeError,'validation_call_cap'):m.main()
            self.assertEqual(seq.call_count,1)
    def test_gpu_nodes_accepted_cpu_node_remains_exact(self):
        m=self.m;m.OUT.mkdir()
        for i,node in enumerate(('csis.cn1','csis.cn3','csis.cn4')):
            status={'state':'COMPLETED','exit':'0:0','node':node,'result':{'passed':True},'result_binding':{}}
            with patch.object(m.p,'status',return_value=status):m.await_success('fixture'+str(i),'root','1','r','gpu')
        with patch.object(m.p,'status',return_value=status):
            with self.assertRaisesRegex(RuntimeError,'stage_failed'):m.await_success('cpu','root','1','r','csis.cn1')
    def test_gpu_master_and_unknown_nodes_fail(self):
        m=self.m;m.OUT.mkdir()
        for node in ('csis.mn1','csis.cn2','csis.cn5','csis.cn[1,3]'):
            with patch.object(m.p,'status',return_value={'state':'COMPLETED','exit':'0:0','node':node,'result':{'passed':True}}):
                with self.assertRaisesRegex(RuntimeError,'stage_failed'):m.await_success('bad','root','1','r','gpu')
    def test_gpu_failure_and_missing_result_fail(self):
        m=self.m;m.OUT.mkdir()
        for state,exitcode,result in [('FAILED','1:0',None),('COMPLETED','0:0',None),('COMPLETED','0:0',{'passed':False})]:
            with patch.object(m.p,'status',return_value={'state':state,'exit':exitcode,'node':'csis.cn3','result':result}):
                with self.assertRaises(RuntimeError):m.await_success('bad','root','1','r','gpu')
    def test_model_stage_order_and_activation_binding(self):
        m=self.m;calls=[]
        def cli(label,folder,pin,script,args):
            calls.append((label,args));return {'job_id':'fixture','activation':{'sha256':'activation-fixture'},'resource_plan':{'sha256':'resources-fixture'}}
        with patch.object(m.p,'run_cli',side_effect=cli),patch.object(m,'await_success',return_value={'result_binding':{'sha256':'qualification-fixture'}}):
            m.model_sequence('selection','folder','pin','root','qual','request')
        self.assertEqual([c[0] for c in calls],['selection_qualification_launch','selection_activation','selection_qwen_launch','selection_audit_qwen_launch','selection_mistral_launch','selection_audit_mistral_launch','selection_statistics_launch'])
        for _,args in calls[2:]:self.assertIn('activation-fixture',args);self.assertIn('resources-fixture',args)
    def test_exclusive_controller_root(self):
        m=self.m;m.OUT.mkdir()
        with patch.object(m,'reconcile') as rec:
            with self.assertRaises(FileExistsError):m.main()
            rec.assert_not_called()

if __name__=='__main__':unittest.main()
