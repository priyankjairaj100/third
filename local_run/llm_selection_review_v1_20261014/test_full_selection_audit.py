"""Small synthetic rejection fixtures; no production seeds, scores or models."""
import base64,copy,hashlib,importlib.util,json,os,random,sys,tempfile,unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import audit_full_selection as a

REPO=Path(os.environ.get('THIRD_SAMPLER_REPO','/Users/priyankjairaj/Downloads/ACL/Three'))
sys.path.insert(0,str(REPO/'empirical_execution'))
from phase3.calibration import _reservoir_insert
p=REPO/'local_run/model_annotation_v3_20261014/protocol.py'
spec=importlib.util.spec_from_file_location('fixture_protocol',p);protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
PLAN={'models':[{'model_id':'fixture-one','revision':'r1'},{'model_id':'fixture-two','revision':'r2'}]}

def fjson(x):return {'numerator':x.numerator,'denominator':x.denominator}

def fixture():
    populations=[0]*20;populations[18]=220405510;sizes=[min(30,n) for n in populations]
    result={'bin_populations':populations,'bin_sample_counts':sizes,'selected_pairs':30,'diagnostic_swap_pairs':3}
    lock={'seed_hex':{'selection_swap':'2'*64}};sample=[]
    for j in range(1,31):
        ids=['fixture0','fixture'+str(j)];identity=protocol.identities(a.PLAN_SHA,a.FRAME_SHA,ids)
        row={'frame_row_indices':[0,j],'original_full_row_indices':[0,j*2],'record_ids':ids,
             'source_unit_ids':['fixture-source','fixture-source'],'same_source':True,**identity,
             'canonical_frame_row_indices':[0,j] if identity['canonical_record_ids']==ids else [j,0],
             'score_hex':float(.8).hex(),'bin':18,'bin_population':220405510,'bin_sample_size':30,
             'inclusion_probability':fjson(Fraction(30,220405510)),'inclusion_weight':fjson(Fraction(220405510,30)),
             'model_seeds':{m['model_id']:protocol.model_seed(a.PLAN_SHA,a.FRAME_SHA,identity['opaque_pair_id'],m['model_id'],m['revision']) for m in PLAN['models']}}
        sample.append(row)
    rng=random.Random(int(protocol.domain_digest(protocol.PREFIX+'sampling',[a.PLAN_SHA,a.FRAME_SHA,'selection_swap','2'*64,None]),16))
    chosen=[]
    for i in range(30):_reservoir_insert(chosen,i,i+1,3,rng)
    swaps=[{'selection_index':i,'opaque_pair_id':sample[i]['opaque_pair_id'],
            'canonical_frame_row_indices':sample[i]['canonical_frame_row_indices'],
            'swapped_frame_row_indices':sample[i]['canonical_frame_row_indices'][::-1],
            'model_seeds':sample[i]['model_seeds'],'diagnostic_only':True} for i in sorted(chosen)]
    return result,sample,swaps,lock


def transport():
    job='999';names={'launch_receipt.json','launch_intent.json','run/process_completion.json','run/software_validation.json',
      'run/tests.txt','run/stdout.txt','run/stderr.txt','slurm-999.out','slurm-999.err',*{'run/output/'+x for x in a.OUTPUT_NAMES}}
    files={name:{**a.raw_binding(b'{}'),'base64':base64.b64encode(b'{}').decode()} for name in names}
    return {'root':str(a.ROOT),'job_id':job,'accounting':'999|COMPLETED|0:0|100||csis.cn1\n999.batch|COMPLETED|0:0|100|100K|csis.cn1',
            'queue':'','files':files,'total_bytes':len(files)*2}


class FullAuditTests(unittest.TestCase):
    def test_1_valid_synthetic_sample_exact_denominator(self):
        result,sample,swaps,lock=fixture();r=a.verify_sample(result,sample,swaps,lock,PLAN)
        self.assertEqual(r['exact_weighted_denominator'],220405510)

    def test_2_weight_and_bin_count_tampering(self):
        for target in ('weight','population','size'):
            result,sample,swaps,lock=fixture()
            if target=='weight':sample[0]['inclusion_weight']['numerator']+=1
            elif target=='population':result['bin_populations'][18]-=1
            else:result['bin_sample_counts'][18]=True
            with self.assertRaises(ValueError):a.verify_sample(result,sample,swaps,lock,PLAN)

    def test_3_orientation_and_model_seed_tampering(self):
        for target in ('orientation','model'):
            result,sample,swaps,lock=fixture()
            if target=='orientation':sample[0]['canonical_frame_row_indices'].reverse()
            else:sample[0]['model_seeds']['fixture-one']+=1
            with self.assertRaises(ValueError):a.verify_sample(result,sample,swaps,lock,PLAN)

    def test_4_duplicate_order_or_unsolicited_label(self):
        for target in ('duplicate','label','mapping'):
            result,sample,swaps,lock=fixture()
            if target=='duplicate':sample[1]=copy.deepcopy(sample[0])
            elif target=='label':sample[0]['judge_label']='positive'
            else:sample[1]['source_unit_ids'][0]='inconsistent';sample[1]['same_source']=False
            with self.assertRaises(ValueError):a.verify_sample(result,sample,swaps,lock,PLAN)

    def test_5_swaps_are_not_extra_units_and_must_replay(self):
        result,sample,swaps,lock=fixture();swaps[0]['diagnostic_only']=False
        with self.assertRaises(ValueError):a.verify_sample(result,sample,swaps,lock,PLAN)
        result,sample,swaps,lock=fixture();swaps[0]['swapped_frame_row_indices'].reverse()
        with self.assertRaises(ValueError):a.verify_sample(result,sample,swaps,lock,PLAN)

    def test_6_transport_hash_inventory_node(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'export.json';original=transport();path.write_text(json.dumps(original));a.read_export(path)
            for target in ('hash','node','child','extra'):
                value=copy.deepcopy(original)
                if target=='hash':value['files']['run/stdout.txt']['sha256']='0'*64
                elif target=='node':value['accounting']=value['accounting'].replace('csis.cn1','csis.cn2')
                elif target=='child':value['accounting']=value['accounting'].replace('999.batch|COMPLETED|0:0','999.batch|FAILED|1:0')
                else:value['files']['run/output/FAILED.json']=value['files']['run/stdout.txt'];value['total_bytes']+=2
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):a.read_export(path)

    def test_7_no_payload_does_not_claim_score_replay(self):
        value=a.retained_scores(SimpleNamespace(vectors=None,rowmap=None),[],{},REPO)
        self.assertFalse(value['executed']);self.assertFalse(value['retained_scores_independently_verified'])
        self.assertFalse(value['record_to_frame_mapping_independently_verified'])

    def test_8_strict_json_and_unicode_LF(self):
        self.assertEqual(a.parse_lines(b'{"id":"a\\u2028b"}\n'),[{'id':'a\u2028b'}])
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e309}'):
            with self.assertRaises(ValueError):a.strict(raw)
        with self.assertRaises(ValueError):a.parse_lines(b'{}')

if __name__=='__main__':unittest.main()
