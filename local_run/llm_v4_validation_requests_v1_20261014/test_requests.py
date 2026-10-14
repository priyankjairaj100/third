"""Selection admission rejects legacy, partial and promoted results."""
import copy
import unittest
import prepare as p

def fixture():
    return {'schema':'third-llm-v4-validation-sample-v1','passed':True,'stage':'complete','package_sha256':p.SAMPLE_SHA,
        'plan_sha256':'a'*64,'frame_result':{'sha256':'b'*64},'pair_count':220405510,
        'bounded_independent_parity':True,'all_selected_scores_pairwise_replayed':True,
        'model_inference_calls':0,'human_annotation_calls':0,'inference_activated':False,
        'confirmatory_study_ready':False,'population':283,'selected_pairs':200,'swap_pairs':20,
        'new_entropy_draws':0,'label_filter':False,'full_above_threshold_frame':True,'overlap_requires_fresh_calls':True}

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
