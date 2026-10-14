"""Version migration tests and numerical-kernel preservation; no model calls."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import contract as c
import audit_helpers as audit

ROOT=Path(__file__).resolve().parents[2]
V4=Path(os.environ.get('THIRD_V4_PROTOCOL',str(ROOT/'local_run/model_annotation_v4_20261014')))
spec=importlib.util.spec_from_file_location('v4_protocol_test',V4/'protocol.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def call():
    pins=c.load_pins();model=pins['models']['qwen'];activation='a'*64;opaque='b'*64
    return {'stage':'selection','opaque_pair_id':opaque,'model_id':model['model_id'],
        'model_revision':model['revision'],'orientation':'canonical','attempt':0,
        'call_id':p.domain_digest(p.PREFIX+'call-id',[activation,'selection',opaque,model['model_id'],model['revision'],'canonical',0]),
        'model_seed':p.model_seed(pins['protocol_plan_sha256'],pins['frame_result_sha256'],opaque,model['model_id'],model['revision']),
        'request_index':0,'selection_index':0,'request_manifest':{'bytes':1,'sha256':'c'*64},
        'request_row_sha256':'d'*64,'input_token_ids_sha256':'e'*64,'activation_lock_sha256':activation}

class ComponentsTests(unittest.TestCase):
    def test_call_domains_bind_plan_family_and_single_attempt(self):
        value=call();self.assertEqual(c.validate_planned_call(value,p),value)
        for key,v in [('attempt',1),('model_seed',value['model_seed']+1),('activation_lock_sha256','f'*64),
                      ('call_id','f'*64),('orientation','swapped_diagnostic')]:
            other=copy.deepcopy(value);other[key]=v
            with self.assertRaises(ValueError):c.validate_planned_call(other,p)
    def test_old_call_domain_is_rejected(self):
        value=call();value['call_id']=c.sha(c.canonical(['third-llm-only-v3/call-id',
            value['activation_lock_sha256'],value['stage'],value['opaque_pair_id'],value['model_id'],
            value['model_revision'],value['orientation'],0]))
        with self.assertRaises(ValueError):c.validate_planned_call(value,p)
    def test_missing_and_unattempted_remain_explicit(self):
        for status in ('missing_output','not_attempted','context_overflow'):
            response=c.envelope(call(),status,error=None if status=='context_overflow' else {'type':'Failure','message':'fixture'})
            c.validate_envelope(response,call(),lambda x:None)
            self.assertIsNone(response['judgment']);self.assertFalse(response['human_completed'])
    def test_generation_function_is_byte_for_byte_preserved(self):
        old=Path(os.environ.get('THIRD_V3_KERNEL',str(ROOT/'local_run/llm_annotation_inference_v1_20261014/kernel.py'))).read_text()
        new=(Path(__file__).parent/'kernel.py').read_text()
        self.assertEqual(old[old.index('def execute_call('):],new[new.index('def execute_call('):])
        self.assertEqual(old[old.index('    def load('):old.index('\ndef execute_call(')],
                         new[new.index('    def load('):new.index('\ndef execute_call(')])
    def test_independent_audit_reconstructs_call_identity(self):
        expected=call();token={k:expected[k] for k in ('stage','opaque_pair_id','model_id','model_revision',
            'orientation','model_seed','request_index','selection_index','input_token_ids_sha256')}
        expected['request_row_sha256']=c.sha(c.canonical(token))
        rebuilt=audit.planned_calls([token],expected['request_manifest'],expected['activation_lock_sha256'],
            c.load_pins()['models']['qwen'],c.load_pins())
        self.assertEqual(rebuilt,[expected])

if __name__=='__main__':unittest.main()
