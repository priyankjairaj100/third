"""Selection admission rejects legacy, partial and promoted results."""
import copy
import unittest
import prepare as p

def fixture():
    return {'schema':'third-llm-tail-selection-v4','passed':True,'stage':'complete','package_sha256':p.V4_SHA,
        'plan_sha256':'a'*64,'frame_result':{'sha256':'b'*64},'pair_count':220405510,
        'bounded_independent_parity':True,'all_selected_scores_pairwise_replayed':True,
        'model_inference_calls':0,'human_annotation_calls':0,'inference_activated':False,
        'confirmatory_study_ready':False,'populations':[220405510]+[0]*12,
        'sample_counts':[30]+[0]*12,'selected_pairs':30,'swap_pairs':3}

class AdmissionTests(unittest.TestCase):
    def test_complete_sampling_fixture_admitted(self):
        p.admit_selection(fixture(),'a'*64,{'sha256':'b'*64})
    def test_old_partial_or_promoted_evidence_rejected(self):
        for key,value in [('schema','third-llm-selection-sample-v1'),('passed',False),
            ('selected_pairs',29),('swap_pairs',2),('inference_activated',True),
            ('confirmatory_study_ready',True),('human_annotation_calls',1),
            ('bounded_independent_parity',False),('package_sha256','c'*64)]:
            result=fixture();result[key]=value
            with self.assertRaises(ValueError):p.admit_selection(result,'a'*64,{'sha256':'b'*64})
    def test_wrong_plan_or_frame_rejected(self):
        with self.assertRaises(ValueError):p.admit_selection(fixture(),'c'*64,{'sha256':'b'*64})
        with self.assertRaises(ValueError):p.admit_selection(fixture(),'a'*64,{'sha256':'c'*64})

if __name__=='__main__':unittest.main()
