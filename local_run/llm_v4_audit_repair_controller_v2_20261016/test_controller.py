"""Software-only tests: never connect, submit jobs, or call models."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).parent
def load():
 s=importlib.util.spec_from_file_location('repair_controller_fixture',HERE/'continue_pipeline.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class ControlTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.m=load();self.m.OUT=Path(self.tmp.name)/'run';self.m.p.OUT=self.m.OUT
 def test_adopted_qwen_audit_never_resubmits(self):
  m=self.m
  with patch.object(m.n,'await_success') as wait,patch.object(m,'rcli',return_value={'published':True}) as cli:
   m.audit('selection','qwen',m.SEL_BASE,'existing-job');wait.assert_called_once();self.assertEqual(cli.call_count,1);self.assertEqual(cli.call_args.args[2],'publish_bridge.py')
 def test_failed_audit_does_not_publish_or_continue(self):
  m=self.m
  with patch.object(m.n,'await_success',side_effect=RuntimeError('fixture-failed')),patch.object(m,'rcli') as cli:
   with self.assertRaises(RuntimeError):m.audit('selection','qwen',m.SEL_BASE,'existing-job')
   cli.assert_not_called()
 def test_negative_selection_never_launches_validation(self):
  m=self.m
  with patch.object(m,'adopt_qwen_audit',return_value='existing'),patch.object(m,'audit') as audit,patch.object(m,'remaining'),patch.object(m.p,'independent_statistics_review',return_value={'selection_passed':False}),patch.object(m,'validation') as val:
   m.main();audit.assert_called_once_with('selection','qwen',m.SEL_BASE,'existing');val.assert_not_called()
  self.assertFalse(json.loads((m.OUT/'completion.json').read_text())['validation_run'])
 def test_failed_adoption_stops_without_dispatch(self):
  m=self.m
  with patch.object(m,'adopt_qwen_audit',side_effect=RuntimeError('fixture')),patch.object(m,'audit') as a,patch.object(m,'remaining') as r:
   with self.assertRaises(RuntimeError):m.main()
   a.assert_not_called();r.assert_not_called()
  self.assertFalse(json.loads((m.OUT/'stopped.json').read_text())['automatic_retry'])
 def test_positive_selection_dispatches_validation_once(self):
  m=self.m;selected={'selection_passed':True}
  with patch.object(m,'adopt_qwen_audit',return_value='existing'),patch.object(m,'audit'),patch.object(m,'remaining'),patch.object(m.p,'independent_statistics_review',return_value=selected),patch.object(m,'validation',return_value={'model_validation_passed':False}) as v:
   m.main();v.assert_called_once_with(selected)
  c=json.loads((m.OUT/'completion.json').read_text());self.assertFalse(c['project_complete']);self.assertFalse(c['independent_validation_review']['model_validation_passed'])
 def test_remaining_sequence_preserves_audit_before_statistics(self):
  m=self.m;events=[]
  def cli(label,*args):events.append(label);return {'job_id':'fixture'}
  with patch.object(m,'rcli',side_effect=cli),patch.object(m.n,'await_success'),patch.object(m,'audit',side_effect=lambda *args:events.append('audit')):
   m.remaining('selection','root',[])
  self.assertEqual(events,['selection_mistral_launch','audit','selection_statistics_launch'])
 def test_validation_call_cap_blocks_models(self):
  m=self.m
  def wait(label,*args):return {'result_binding':{'sha256':'fixture'},'result':{'planned_requests':441}}
  with patch.object(m.p,'run_cli',return_value={'job_id':'fixture'}) as cli,patch.object(m.n,'await_success',side_effect=wait),patch.object(m,'audit') as audit:
   with self.assertRaisesRegex(RuntimeError,'validation_call_cap'):m.validation({'bindings':{'statistics':{'sha256':'fixture'}}})
   self.assertEqual(cli.call_count,2);audit.assert_not_called()
 def test_exclusive_controller_root_refuses_restart(self):
  m=self.m;m.OUT.mkdir()
  with patch.object(m,'adopt_qwen_audit') as adopt:
   with self.assertRaises(FileExistsError):m.main()
   adopt.assert_not_called()
if __name__=='__main__':unittest.main()
