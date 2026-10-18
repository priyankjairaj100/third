"""Focused Stage 2 wrapper delta checks; no scheduler or scientific execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',HERE/'job_runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class BundleChecks(unittest.TestCase):
    def valid(self):
        return {'status':'completed','all51cuts_completed':True,'threshold_indices':list(range(51)),
                'model_or_human_annotation_calls':0,'task_learner_trained':False,'v4_failure_retained':True,
                'kernel_no_reaccess_isolation_claim':False,'supervised_prediction_effect_claim':False}
    def test_stage2_success_receipt(self):runner.validate_completion(self.valid())
    def test_incomplete_or_old_receipt_rejected(self):
        variants=[{'status':'completed','all_thresholds_completed':True}]
        for key,value in [('status','failed'),('all51cuts_completed',False),('threshold_indices',list(range(50)))]:
            v=self.valid();v[key]=value;variants.append(v)
        for v in variants:
            with self.assertRaises(RuntimeError):runner.validate_completion(v)
    def test_scope_violations_rejected(self):
        for key,value in [('model_or_human_annotation_calls',1),('task_learner_trained',True),('v4_failure_retained',False),('kernel_no_reaccess_isolation_claim',True),('supervised_prediction_effect_claim',True)]:
            v=self.valid();v[key]=value
            with self.assertRaises(RuntimeError):runner.validate_completion(v)
    def test_fixed_package_run_and_slurm_budget(self):
        source=(HERE/'launch.py').read_text();tree=ast.parse(source)
        self.assertNotIn('postnegative-structure-v1',source)
        self.assertIn('postnegative-curation-repair-v1-20261018-a01',source)
        self.assertIn("HERE/'curation_repair.sbatch'",source)
        self.assertIn("'--output=/dev/null','--error=/dev/null'",source)
        self.assertEqual(runner.RUN_NAME,'postnegative-curation-repair-v1-20261018-a01')
        sbatch=(HERE/'curation_repair.sbatch').read_text()
        self.assertIn('#SBATCH --time=01:05:00',sbatch)
        self.assertIn('#SBATCH --mem=8G',sbatch)
        self.assertIn('#SBATCH --exclude=csis.mn1,csis.cn2',sbatch)
    def test_protocol_preserves_prospective_design(self):
        # The original prospective file is part of the local development bundle;
        # compute execution may ship only the frozen operational plan.
        v=json.loads((HERE/'PLAN.json').read_text())
        self.assertEqual(v['status'],'frozen_for_stage2')
        self.assertFalse(v['stage1_outcomes_used_for_design'])
        self.assertEqual(v['request_protocol']['trajectories_per_arm'],1)
        self.assertEqual(v['request_protocol']['initial_horizon'],128)
        self.assertEqual(v['request_protocol']['checkpoints'],[1,8,32,128])
        self.assertEqual(len(v['thresholds']),51)
        self.assertEqual(v['execution_policy']['nfs_small_artifact_budget_bytes'],2*1024**2)
        self.assertEqual(v['resource_limits']['storage_reserve_bytes'],8*1024**3)
        self.assertLess(v['resource_qualification']['linear_graph_build_projection_seconds'],v['resource_limits']['wall_seconds'])
if __name__=='__main__':unittest.main()
