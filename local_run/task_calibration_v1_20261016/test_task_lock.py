"""Synthetic software fixtures; no cluster connection or empirical claims."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.environ.get('THIRD_LOCAL_REPO', str(Path(__file__).resolve().parents[2])))
import task_lock as t
from empirical_execution.phase3 import panels
from empirical_execution.phase4 import model_selection as ms


def row(i):
    while True:
        value = {'record_id':'fixture:'+str(i),'text':'software fixture only '+str(i),
                 'original_fields':{'site':'fixture.example','OwnerUserId':str(i),'Tags':['t'+str(k) for k in range(24)]}}
        unit, kind = panels.source_unit(value,'askubuntu')
        if panels.digest_partition(panels.hash_integer('ccu-v1-source-split',unit)) == 'calibration':
            return {**value,'source_unit_id':unit,'source_kind':kind,'partition':'calibration'}
        i += 1


def mapping(value, index=1):
    return {'calibration_row_index':0,'original_full_row_index':index,
        **{k:value[k] for k in ('record_id','partition','source_unit_id','source_kind')},
        'text_sha256':hashlib.sha256(value['text'].encode()).hexdigest()}


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'records.jsonl';self.r=row(100);self.m=mapping(self.r)
    def write(self, value=None):
        # Noncalibration lines deliberately are not JSON: the reader must not decode them.
        self.path.write_bytes(b'noncalibration bytes\n'+json.dumps(value or self.r).encode()+b'\nnoncalibration bytes\n')
    def test_decodes_only_bound_calibration_rows(self):
        self.write();rows=t.collect_calibration(self.path,[self.m],3,panels)
        self.assertEqual(rows[0]['original_fields']['Tags'],self.r['original_fields']['Tags'])
        self.assertNotIn('text',rows[0]);self.assertEqual(len(rows),1)
    def test_source_test_partition_rejected(self):
        r={**self.r,'partition':'test'};self.write(r)
        with self.assertRaisesRegex(ValueError,'selected_source_partition'):t.collect_calibration(self.path,[self.m],3,panels)
    def test_wrong_source_text_rejected(self):
        self.write({**self.r,'text':'changed'})
        with self.assertRaisesRegex(ValueError,'selected_source_text'):t.collect_calibration(self.path,[self.m],3,panels)
    def test_owner_tamper_rejected(self):
        r=copy.deepcopy(self.r);r['original_fields']['OwnerUserId']='different';self.write(r)
        with self.assertRaisesRegex(ValueError,'selected_native_source'):t.collect_calibration(self.path,[self.m],3,panels)
    def test_duplicate_or_out_of_order_mapping_rejected(self):
        second={**self.m,'calibration_row_index':1,'record_id':'different'}
        with self.assertRaisesRegex(ValueError,'ordered_unique_mapping_indices'):t.validate_mapping([self.m,second],2)
    def test_wrong_population_count_rejected(self):
        self.write()
        with self.assertRaisesRegex(ValueError,'complete_selected_source_rows'):t.collect_calibration(self.path,[self.m],4,panels)
    def test_test_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError,'calibration_only_mapping'):t.validate_mapping([{**self.m,'partition':'test'}],1)
    def test_frozen_task_selector_integration(self):
        import numpy as np
        rows=[row(1000*i+1) for i in range(10)]
        for i,r in enumerate(rows):r['original_fields']['Tags']=['t'+str(k) for k in range(24) if (i+k)%3]
        provenance={'dataset_id':'askubuntu','evidence_role':'software_fixture_only','population_scope':'Synthetic unit test only','encoder_id':'fixture','encoder_revision':'fixture'}
        features=np.eye(3,dtype=np.float32)[np.arange(len(rows))%3]
        vocab=ms.select_tag_vocabulary(rows,provenance);lock=ms.select_ridge_regularization(features,rows,provenance,vocabulary_lock=vocab);decisions=ms.select_tag_decision_thresholds(lock)
        self.assertEqual(len(vocab['vocabulary']),20);self.assertEqual(len(lock['selection_ledger']),6)
        self.assertEqual(len(decisions['thresholds']),20);self.assertEqual(decisions['lambda_lock_sha256'],lock['sha256'])
        self.assertFalse(lock['confirmatory_study_ready'])


if __name__=='__main__':unittest.main()
