"""Synthetic storage/control fixtures. No source labels, model weights or remote calls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('test_bound_runner',HERE/'run_model.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
REPO=Path(os.environ.get('THIRD_INFERENCE_REPO',HERE.parents[1]))
COMP=Path(os.environ.get('THIRD_INFERENCE_COMPONENTS',REPO/'local_run/llm_annotation_inference_v1_20261014'))
sys.path.insert(0,str(COMP))
import contract as C
import ledger as L
import kernel as K

def fake_call(i):
    return {'stage':'selection' if i<2 else 'selection_swap','opaque_pair_id':str(i%2)*64,'model_id':'software-fixture-model',
        'model_revision':'fixture','orientation':'canonical' if i<2 else 'swapped_diagnostic','attempt':0,
        'call_id':R.sha(str(i).encode()),'model_seed':i,'request_index':i,'selection_index':i%2,
        'request_manifest':{'bytes':1,'sha256':'a'*64},'request_row_sha256':'b'*64,'input_token_ids_sha256':'c'*64,'activation_lock_sha256':'d'*64}

def validator(j):raise ValueError('No judgments allowed in control fixtures')

class ScheduleTests(unittest.TestCase):
    def fixture(self,folder,status='context_overflow',failure_index=None):
        calls=[fake_call(i) for i in range(3)];ledger=L.Ledger(folder,100000,calls,validator);visited=[]
        def call(kernel,planned,token,row,writer):
            i=planned['request_index'];visited.append(i)
            if i==failure_index:raise ValueError('synthetic input-integrity fault')
            if status=='runtime_error':return C.envelope(planned,status,error={'type':'SyntheticFailure','message':'mock failure'})
            return C.envelope(planned,status)
        components={'kernel':NS(execute_call=call)};bundle={'tokens':[{}, {}, {}],'rows':[{}, {}, {}]}
        return calls,ledger,visited,components,bundle
    def test_all_original_indices_once_longest_first(self):
        with tempfile.TemporaryDirectory() as td:
            calls,l,v,c,b=self.fixture(Path(td)/'out');fatal,integrity=R.run_schedule(l,None,b,calls,[2,0,1],c)
            self.assertIsNone(fatal);self.assertIsNone(integrity);self.assertEqual(v,[2,0,1])
            rows,summary=l.finish();self.assertEqual([x['request_index'] for x in rows],[0,1,2]);self.assertEqual(summary['recorded_calls'],3)
    def test_runtime_error_stops_rest_not_attempted(self):
        with tempfile.TemporaryDirectory() as td:
            calls,l,v,c,b=self.fixture(Path(td)/'out','runtime_error');fatal,integrity=R.run_schedule(l,None,b,calls,[2,0,1],c)
            self.assertIsNone(integrity);self.assertEqual(v,[2]);rows,_=l.finish(fatal)
            self.assertEqual([r['status'] for r in rows],['not_attempted','not_attempted','runtime_error'])
    def test_integrity_failure_distinct_and_denominator_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            calls,l,v,c,b=self.fixture(Path(td)/'out',failure_index=0);fatal,integrity=R.run_schedule(l,None,b,calls,[2,0,1],c)
            self.assertIn('input-integrity',integrity);rows,_=l.finish(fatal)
            self.assertEqual([r['status'] for r in rows],['missing_output','not_attempted','context_overflow'])
    def test_schedule_cannot_drop_duplicate_or_add(self):
        with tempfile.TemporaryDirectory() as td:
            calls,l,v,c,b=self.fixture(Path(td)/'out')
            for order in ([0,1],[0,1,1],[0,1,2,3]):
                with self.assertRaisesRegex(ValueError,'schedule_permutation'):R.run_schedule(l,None,b,calls,order,c)
            self.assertEqual(v,[])
    def test_no_resume_same_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'out';self.fixture(p)
            with self.assertRaises(FileExistsError):self.fixture(p)

class BindingResourceTests(unittest.TestCase):
    def test_fixed_output_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td).resolve();p=w/'qualification'/'run';p.mkdir(parents=True);out=p/'output'
            self.assertEqual(R.safe_output(w,out,str(out)),out)
            with self.assertRaisesRegex(ValueError,'fixed_activated'):R.safe_output(w,p/'another',str(out))
            out.symlink_to(w/'missing')
            with self.assertRaisesRegex(ValueError,'already_exists'):R.safe_output(w,out,str(out))
    def test_exact_package_tamper_and_extra_file(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'source.py').write_text('x=1\n');m={'files':{'source.py':R.binding(p/'source.py')}}
            (p/'package_manifest.json').write_bytes(R.canonical(m));pin=R.binding(p/'package_manifest.json');R.package(p,pin)
            (p/'foreign.py').write_text('pass')
            with self.assertRaisesRegex(ValueError,'package_inventory'):R.package(p,pin)
            (p/'foreign.py').unlink();(p/'source.py').write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'member_binding'):R.package(p,pin)
    def test_runtime_registration_counts_actual_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            s=L.PrivateOutput(Path(td)/'out',10000);rt=s.root/'runtime';rt.mkdir()
            for n in ('generation_config.json','token_eligibility.json','effective_constraint_config.json','runtime_manifest.json'):(rt/n).write_bytes(b'{}\n')
            self.assertEqual(len(R.track_runtime(s,rt,12)),4);self.assertEqual(s.used,12);s.recheck()
            with self.assertRaisesRegex(ValueError,'duplicate'):R.track_runtime(s,rt,12)
    def test_runtime_extra_file_and_reserve_refused(self):
        with tempfile.TemporaryDirectory() as td:
            s=L.PrivateOutput(Path(td)/'out',10000);rt=s.root/'runtime';rt.mkdir();(rt/'foreign').write_text('bad')
            with self.assertRaisesRegex(ValueError,'inventory'):R.track_runtime(s,rt,12)
    def test_allocation_is_exact_not_gpu_login(self):
        family={'cpus':4,'host_memory_bytes':48*1024**3,'gpu_count':1,'gpu_model':'A10080GB'}
        env={'SLURM_JOB_ID':'123','SLURM_CPUS_PER_TASK':'4','SLURM_MEM_PER_NODE':'49152',
             **{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}}
        with patch.object(R.platform,'system',return_value='Linux'),patch.dict(os.environ,env,clear=True):
            self.assertEqual(R.runtime_preflight(family)['slurm_job_id'],'123')
            with patch.dict(os.environ,{'SLURM_MEM_PER_NODE':'8192'}):
                with self.assertRaisesRegex(ValueError,'memory_allocation'):R.runtime_preflight(family)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'bad'}):
                with self.assertRaisesRegex(ValueError,'numeric_slurm'):R.runtime_preflight(family)
    def test_nonfinite_and_duplicate_metadata_refused(self):
        for raw in ('NaN','1e309','{"a":1,"a":2}'):
            with self.assertRaises(ValueError):R.strict(raw)

if __name__=='__main__':unittest.main()
