"""Local synthetic tests only; never call the remote launcher main function."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FILE=Path(__file__).with_name('assembly_launch_remote.py')
spec=importlib.util.spec_from_file_location('assembly_launcher_tested',FILE)
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)

class LauncherTests(unittest.TestCase):
    def raw(self):
        return '\n'.join(f'13498_{i}|COMPLETED|0:0|csis.cn1' for i in range(32))

    def test_exact_terminal_tasks(self):
        data=self.raw()+'\n13498_0.batch|COMPLETED|0:0|csis.cn1'
        self.assertEqual(len(x.fresh_terminal(data)),32)

    def test_missing_duplicate_failed_or_wrong_node_refused(self):
        good=self.raw()
        variants=[good.rsplit('\n',1)[0],good+'\n'+good.splitlines()[0],
                  good.replace('COMPLETED','RUNNING',1),good.replace('0:0','1:0',1),
                  good.replace('csis.cn1','csis.cn2',1)]
        for data in variants:
            with self.subTest(data=data[:80]),self.assertRaises(ValueError):x.fresh_terminal(data)

    def test_every_prior_assembly_artifact_refused(self):
        for name in ('assembly_launch_intent.json','assembly_launch_receipt.json','assembly-run'):
            with tempfile.TemporaryDirectory() as tmp,patch.object(x,'ROOT',Path(tmp)):
                x.refuse_previous();(Path(tmp)/name).write_text('{}')
                with self.assertRaises(ValueError):x.refuse_previous()

    def test_broken_prior_symlink_also_refused(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(x,'ROOT',Path(tmp)):
            (Path(tmp)/'assembly_launch_intent.json').symlink_to(Path(tmp)/'absent')
            with self.assertRaises(ValueError):x.refuse_previous()

    def test_materialized_stats_budget_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(x,'DIRECTORY_SLACK',16),patch.object(x,'REMAINING_ALLOWANCE',1024),patch.object(x,'ORIGINAL_ALLOWANCE',8192):
            root=Path(tmp);(root/'tiny').write_bytes(b'x'*512)
            d=x.materialized_budget(root)
            self.assertEqual(d['logical_bytes'],512)
            self.assertGreaterEqual(d['sum_max_logical_or_allocated_bytes'],512)
            (root/'alias').symlink_to(root/'tiny')
            with self.assertRaises(ValueError):x.materialized_budget(root)

    def test_materialized_output_cannot_exceed_old_envelope(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(x,'DIRECTORY_SLACK',16),patch.object(x,'REMAINING_ALLOWANCE',1024),patch.object(x,'ORIGINAL_ALLOWANCE',8192):
            root=Path(tmp);(root/'large').write_bytes(b'x'*10000)
            with self.assertRaises(ValueError):x.materialized_budget(root)

    def test_exclusive_receipt_preserves_previous_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'intent.json';x.write_exclusive(p,{'job':'first'});prior=p.read_bytes()
            with self.assertRaises(FileExistsError):x.write_exclusive(p,{'job':'second'})
            self.assertEqual(prior,p.read_bytes())

    def test_fixed_capacity_and_no_reserve_reduction(self):
        self.assertEqual(x.RESERVE,8589934592)
        self.assertEqual(x.REMAINING_ALLOWANCE,872415232)
        self.assertEqual(x.RESERVE+x.REMAINING_ALLOWANCE,9462349824)
        self.assertEqual(x.ORIGINAL_ALLOWANCE,2415919104)

class StageBudgetTests(unittest.TestCase):
    def test_stage_policy_and_reserve_cannot_silently_change(self):
        import json,copy
        policy=json.loads(Path(__file__).with_name('stage_budget_report.json').read_text())
        x.validate_budget(policy)
        for key,value in [('reserve_bytes_unchanged',0),('rounded_stage_allowance_bytes',1),
                          ('exact_payload_plus_margins_bytes',1),('full_population_cache_acceptance',True)]:
            bad=copy.deepcopy(policy);bad[key]=value
            with self.assertRaises(ValueError):x.validate_budget(bad)
        bad=copy.deepcopy(policy);bad['stage_margins_bytes']['additional_contingency_unchanged']=0
        with self.assertRaises(ValueError):x.validate_budget(bad)

    def test_exact_fresh_capacity_boundary(self):
        wanted=9462349824
        x.require_capacity(wanted)
        for value in (wanted-1,True,float(wanted)):
            with self.assertRaises(ValueError):x.require_capacity(value)

    def test_old_sealed_gate_is_still_byte_identical(self):
        import json,hashlib
        old=Path(__file__).resolve().parent.parent/'assembly_gate_v1'
        manifest=(old/'package_manifest.json').read_bytes()
        self.assertEqual(hashlib.sha256(manifest).hexdigest(),x.ORIGINAL_GATE_SHA)
        for name,e in json.loads(manifest)['files'].items():
            raw=(old/name).read_bytes()
            self.assertEqual({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},e)

if __name__=='__main__':unittest.main()
