import base64
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

P=Path(__file__).with_name('audit_requests.py')
spec=importlib.util.spec_from_file_location('audit_requests_tested',P)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Tokenizer:
    def __init__(self,n=3): self.n=n
    def __len__(self):return 100
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs=={'tokenize':False,'add_generation_prompt':True}
        return 'PROMPT:'+json.dumps(messages,ensure_ascii=False)
    def __call__(self,rendered,**kwargs):
        assert kwargs=={'add_special_tokens':False,'truncation':False}
        return {'input_ids':[5]*self.n,'attention_mask':[1]*self.n}


def setup_row(n=3):
    texts={0:{'text':'café\u2028line','record_id':'α','original_full_row_index':2,'text_sha256':a.digest('café\u2028line'.encode())},
      1:{'text':'B','record_id':'b','original_full_row_index':4,'text_sha256':a.digest(b'B')}}
    r={'request_index':0,'selection_index':0,'stage':'selection','orientation':'canonical','diagnostic_only':False,
       'opaque_pair_id':'opaque','natural_pair_id':'natural','shown_calibration_row_indices':[0,1],
       'shown_record_ids':['α','b'],'shown_text_sha256':[texts[i]['text_sha256'] for i in (0,1)],'model_seeds':{'M':42}}
    pin={'model_id':'M','revision':'revision','tokenizer_identity':{'expected':'identity'}}
    tok=Tokenizer(n)
    inp={'pair_id':'opaque','orientation':'canonical','text_a':texts[0]['text'],'text_b':'B'}
    messages=[{'role':'system','content':'system'},{'role':'user','content':json.dumps({'task_definition':'task',
      'documents':[{'label':'A','text':texts[0]['text']},{'label':'B','text':'B'}]},ensure_ascii=False)}]
    rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    row={**r,'model_id':'M','model_revision':'revision','model_seed':42,'input_row_sha256':a.digest(a.canonical(inp,True)),
      'input_row_hash_serialization':'qualified pilot canonical JSON: ensure_ascii=True, sorted keys, compact separators',
      'rendered_prompt_sha256':a.digest(rendered.encode()),'input_token_ids':[5]*n,'input_tokens':n,
      'input_token_ids_sha256':a.digest(a.canonical([5]*n)),'attention_mask_policy':'one for every unpadded token',
      'max_new_tokens':512,'total_token_budget':32768,'context_fit_with_output_reserve':n<=32256,
      'preparation_status':'ready_for_future_activation' if n<=32256 else 'context_overflow',
      'full_text_preserved':True,'truncation_applied':False,'activation_lock_sha256':None,'call_id':None}
    return texts,r,pin,tok,row


def export_fixture():
    names={'launch_receipt.json','launch_intent.json','run/process_completion.json','run/software_validation.json',
      'run/tests.txt','run/stdout.txt','run/stderr.txt','slurm-123.out','slurm-123.err',
      *{'run/output/'+n for n in a.NAMES}}
    files={n:{**a.binding(b'{}'),'base64':base64.b64encode(b'{}').decode()} for n in names}
    return {'root':str(a.ROOT),'job_id':'123','queue':'',
      'accounting':'123|COMPLETED|0:0|2||csis.cn1\n123.batch|COMPLETED|0:0|2|123K|csis.cn1\n',
      'files':files,'total_bytes':len(files)*2}

class AuditTests(unittest.TestCase):
    def test_strict_and_physical_lf(self):
        rows=[{'text':'a\u2028b\u0085c'}, {'id':2}]
        self.assertEqual(a.lines(b''.join(a.canonical(r)+b'\n' for r in rows)),rows)
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}'):
            with self.assertRaises(ValueError):a.strict(raw)
        with self.assertRaises(ValueError):a.lines(b'{}')
        self.assertFalse(a.exact(True,1))

    def check_export(self,e):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'export.json';p.write_bytes(a.canonical(e));return a.read_export(p)

    def test_complete_transport(self):
        e=export_fixture();result,files,b=self.check_export(e)
        self.assertEqual(len(files),17);self.assertEqual(b,a.binding(a.canonical(e)))

    def test_transport_mutation_and_extra_file(self):
        e=export_fixture();e['files']['run/stdout.txt']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'transport_binding'):self.check_export(e)
        e=export_fixture();e['files']['extra']=e['files']['run/stdout.txt'];e['total_bytes']+=2
        with self.assertRaisesRegex(ValueError,'exact17file'):self.check_export(e)

    def test_failed_or_unrelated_accounting(self):
        for value in ('123|RUNNING|0:0|2||csis.cn1\n','124|COMPLETED|0:0|2||csis.cn1\n',
                      '123|COMPLETED|0:0|2||csis.cn2\n'):
            e=export_fixture();e['accounting']=value
            with self.assertRaises(ValueError):self.check_export(e)
        e=export_fixture();e['queue']='123 RUNNING'
        with self.assertRaises(ValueError):self.check_export(e)

    def test_unicode_and_complete_token_replay(self):
        texts,r,p,t,row=setup_row();result=a.verify_model([row],[r],texts,p,t,'system','task')
        self.assertEqual(result['total_input_tokens'],3);self.assertEqual(result['scheduled_request_indices'],[0])
        self.assertNotEqual(row['input_row_sha256'],a.digest(a.canonical({'pair_id':'opaque','orientation':'canonical',
          'text_a':texts[0]['text'],'text_b':'B'},False)))

    def test_changed_token_or_prompt_hash_refused(self):
        for key,value in [('input_token_ids',[5,5,6]),('rendered_prompt_sha256','0'*64),('call_id','invented'),('extra',1)]:
            texts,r,p,t,row=setup_row();row[key]=value
            with self.assertRaises(ValueError):a.verify_model([row],[r],texts,p,t,'system','task')

    def test_overflow_retained_boundary(self):
        for n,over in ((32256,0),(32257,1)):
            texts,r,p,t,row=setup_row(n);result=a.verify_model([row],[r],texts,p,t,'system','task')
            self.assertEqual(result['requests'],1);self.assertEqual(result['overflow_requests'],over)
            self.assertEqual(result['canonical_overflows'],over)
            self.assertTrue(result['overflow_retains_full_sampling_denominator'])

    def test_missing_duplicate_and_wrong_text_refused(self):
        texts,r,p,t,row=setup_row()
        sample=[{'frame_row_indices':[0,1],'original_full_row_indices':[2,4],'record_ids':['α','b'],
          'canonical_frame_row_indices':[0,1],'opaque_pair_id':'opaque','natural_pair_id':'natural','model_seeds':{'M':42}}]
        tr=[{'calibration_row_index':i,**texts[i]} for i in texts]
        self.assertEqual(a.texts_and_requests(tr,[r],sample,[]),{i:x for i,x in zip(texts,tr)})
        for bad in (tr[:-1],tr+tr[:1]):
            with self.assertRaises(ValueError):a.texts_and_requests(bad,[r],sample,[])
        bad=copy.deepcopy(tr);bad[0]['text']='edited'
        with self.assertRaises(ValueError):a.texts_and_requests(bad,[r],sample,[])
        bad=copy.deepcopy(r);bad['shown_calibration_row_indices']=[1,0]
        with self.assertRaises(ValueError):a.texts_and_requests(tr,[bad],sample,[])

if __name__=='__main__':unittest.main(verbosity=2)
