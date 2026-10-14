"""Software fixtures only: no natural labels, network, jobs, activation or inference."""
import importlib.util,json,tempfile,unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_remaining_pipeline as flow
import review_postprocess_terminal as audit

class ControlTests(unittest.TestCase):
    def simulate(self,qwen_success=True,audit_fails=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        c=flow.Coordinator(Path(temp.name));events=[];final=[]
        c.event=lambda *args,**kwargs:None
        def family(alias):events.append('family:'+alias);return {'execution_passed':qwen_success if alias=='qwen' else False}
        def post(mode):
            events.append('audit:'+mode)
            if audit_fails:raise ValueError('deliberate_failed_audit')
            return {'selection_passed':False,'selected_tau':None,'conservative_positives_of90':0}
        c.family=family;c.postprocess=post
        def remote(*args):events.append('submit:mistral');return {'returncode':0,'job_id':'12345'}
        c.remote=remote
        patches=(patch.object(flow,'read',return_value={'job_id':'13574','returncode':0}),patch.object(flow,'bind',return_value={'bytes':1,'sha256':'0'*64}),patch.object(flow,'new_json',side_effect=lambda p,x:final.append(x)))
        with patches[0],patches[1],patches[2]:
            if not qwen_success or audit_fails:
                with self.assertRaises(ValueError):c.run()
            else:c.run()
        return events,final
    def test_success_and_failed_threshold_finish_without_validation(self):
        events,final=self.simulate()
        self.assertEqual(events,['family:qwen','audit:qwen','submit:mistral','family:mistral','audit:mistral','audit:statistics'])
        self.assertTrue(final[0]['passed']);self.assertFalse(final[0]['selection_passed']);self.assertFalse(final[0]['independent_validation_performed'])
    def test_qwen_runtime_failure_is_audited_then_holds_mistral(self):
        events,final=self.simulate(qwen_success=False)
        self.assertEqual(events,['family:qwen','audit:qwen']);self.assertFalse(final)
    def test_audit_failure_prevents_further_model_calls(self):
        events,final=self.simulate(audit_fails=True)
        self.assertEqual(events,['family:qwen','audit:qwen']);self.assertFalse(final)
    def test_uncertain_remote_receipt_is_never_overwritten_or_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'script.py';script.write_text('print(0)');c=flow.Coordinator(root);output=root/'receipt.json'
            with patch.object(flow.subprocess,'run',return_value=SimpleNamespace(returncode=124)) as run:
                with self.assertRaisesRegex(ValueError,'stop_no_retry'):c.remote('test',script,output)
                with self.assertRaises(FileExistsError):c.remote('test',script,output)
                self.assertEqual(run.call_count,1)
    def test_changed_bound_helper_stops_before_remote_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'script.py';script.write_text('original');c=flow.Coordinator(root);c.pins={str(script):flow.bind(script)};script.write_text('changed')
            with patch.object(flow.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'source_changed'):c.remote('test',script,root/'receipt.json')
                run.assert_not_called()

class StatisticsTests(unittest.TestCase):
    def fixture(self,positive=False,uncertain=False):
        spec=importlib.util.spec_from_file_location('frozen_statistics',audit.REPO/'local_run/llm_annotation_statistics_v1_20261014/statistics_core.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        populations=[0]*20;rows=[]
        for b,pop,value in [(17,30,.75),(18,60,.85),(19,120,.95)]:
            populations[b]=pop
            for i in range(30):
                weight=Fraction(pop,30);prob=1/weight
                rows.append({'opaque_pair_id':f'software-{b}-{i}','bin':b,'bin_population':pop,'bin_sample_size':30,'score_hex':value.hex(),'inclusion_weight':audit.fj(weight),'inclusion_probability':audit.fj(prob)})
        reports={};observed=[]
        for alias,model in zip(('qwen','mistral'),core.MODELS):
            panel=[{'opaque_pair_id':r['opaque_pair_id'],'model_id':model,'status':'valid','category':'substantially_same_meaning' if positive else ('uncertain' if uncertain else 'unrelated')} for r in rows]
            observed.extend(panel);reports[alias]={'canonical_observations':panel,'diagnostic_observations':panel[:9]}
        report={'selection_statistics':core.selection(rows,populations,observed),'swap_diagnostics':{a:{'pairs':9,'status_or_category_changes':0,'binary_positive_changes':0,'used_in_threshold_estimator':False} for a in reports}}
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        sample=root/'llm-selection-sampler-v1-20261014/selection-a01/run/output/selection.private.jsonl';sample.parent.mkdir(parents=True);sample.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for alias,r in reports.items():
            out=root/audit.root_name(alias)/'run/report.json';out.parent.mkdir(parents=True);out.write_text(json.dumps(r))
        return root,report
    def test_all_negative_full_denominator(self):
        root,r=self.fixture()
        with patch.object(audit,'REPORTS',root):out=audit.recompute_statistics(r)
        self.assertFalse(out['selection_passed']);self.assertEqual(out['conservative_positives_of90'],0)
    def test_all_positive_earliest_cut(self):
        root,r=self.fixture(positive=True)
        with patch.object(audit,'REPORTS',root):out=audit.recompute_statistics(r)
        self.assertTrue(out['selection_passed']);self.assertEqual(out['selected_tau'],{'numerator':1,'denominator':2})
    def test_uncertain_remains_nonpositive(self):
        root,r=self.fixture(uncertain=True)
        with patch.object(audit,'REPORTS',root):out=audit.recompute_statistics(r)
        self.assertFalse(out['selection_passed'])
    def test_altered_denominator_rejected(self):
        root,r=self.fixture();r['selection_statistics']['grid_ledger'][0]['weighted_denominator']['numerator']-=1
        with patch.object(audit,'REPORTS',root),self.assertRaisesRegex(ValueError,'independent_exact'):audit.recompute_statistics(r)
    def test_promoted_threshold_rejected(self):
        root,r=self.fixture();r['selection_statistics']['selection_passed']=True
        with patch.object(audit,'REPORTS',root),self.assertRaisesRegex(ValueError,'independent_exact'):audit.recompute_statistics(r)

if __name__=='__main__':unittest.main(verbosity=2)
