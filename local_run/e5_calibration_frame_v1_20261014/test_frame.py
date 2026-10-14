"""Small explicitly synthetic software fixtures; no model or remote execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import frame as f

REPO = Path(os.environ.get('THIRD_FRAME_REPO',Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0,str(REPO/'empirical_execution'))
from phase3 import panels


def fixture():
    salt = 'ccu-v1-source-split'
    found = {}
    for owner in range(1,200):
        raw = {'record_id':f'SOFTWARE_FIXTURE_{owner}','text':'Synthetic software text.',
               'original_fields':{'site':'askubuntu','OwnerUserId':owner}}
        source,kind = panels.source_unit(raw,'askubuntu')
        part = panels.digest_partition(panels.hash_integer(salt,source))
        found.setdefault(part,{**raw,'source_unit_id':source,'source_kind':kind,'partition':part})
    rows = [found['train'],found['calibration'],found['test'],
            {**found['calibration'],'record_id':'SOFTWARE_FIXTURE_same_source'}]
    for i in range(200):
        raw = {'record_id':f'SOFTWARE_FIXTURE_unknown_{i}','text':'Software unknown owner.',
               'original_fields':{'site':'askubuntu','OwnerUserId':-1}}
        source,kind = panels.source_unit(raw,'askubuntu')
        if panels.digest_partition(panels.hash_integer(salt,source)) == 'calibration':
            rows.append({**raw,'source_unit_id':source,'source_kind':kind,'partition':'calibration'});break
    pins = dict(full_rows=5,calibration_rows=3,unordered_pair_count=3,
        partition_counts={'train':1,'calibration':3,'test':1},dataset_id='askubuntu',source_split_salt=salt,
        source_ordered_ids_sha256=hashlib.sha256(json.dumps([r['record_id'] for r in rows],separators=(',',':')).encode()).hexdigest())
    return rows,pins


def manifest_fixture():
    pins = f.read_json(Path(__file__).with_name('expectations.json'))
    plan = {'sha256':'s','kind':'full','shards':[{}]*32,
            'frame_row_indices':list(range(pins['full_rows'])),'documents':pins['full_rows']}
    manifest = dict(schema='third-e5-sharded-cpu-cache-v1',passed=True,plan_sha256='s',
        fingerprint=pins['fingerprint'],runtime={'software':'fixture'},rows=pins['full_rows'],
        shape=[pins['full_rows'],768],dtype='float32',
        frame_ordered_ids_sha256=pins['source_ordered_ids_sha256'],
        full_source_frame_sha256=pins['full_source_frame_sha256'],files=pins['assembly_files'],
        all_rows_fresh_process_replayed_under_this_version=True)
    manifest.update(dict.fromkeys(('source_authenticity_verified','primary_input_acceptance',
        'original_monolithic_full_writer_executed','original_monolithic_full_replay_executed',
        'full_population_cache_acceptance','confirmatory_study_ready'),False))
    return manifest,plan,pins


class FrameTests(unittest.TestCase):
    def test_every_calibration_row_source_order_and_same_source_pairs(self):
        rows,pins = fixture(); records,grouping = f.calibration_mapping(rows,panels,pins)
        self.assertEqual([r['original_full_row_index'] for r in records],[1,3,4])
        self.assertEqual([r['calibration_row_index'] for r in records],[0,1,2])
        self.assertEqual(records[0]['source_unit_id'],records[1]['source_unit_id'])
        self.assertFalse(grouping['within_calibration_same_source_pairs_excluded'])
        self.assertEqual(len(records)*(len(records)-1)//2,3)

    def test_unknown_owner_kept_singleton_and_text_hash_only(self):
        rows,pins = fixture(); records,grouping = f.calibration_mapping(rows,panels,pins)
        self.assertEqual(grouping['calibration_unknown_singletons'],1)
        self.assertEqual(records[-1]['source_kind'],'unknown_singleton')
        self.assertEqual(records[-1]['text_sha256'],hashlib.sha256(rows[-1]['text'].encode()).hexdigest())
        self.assertNotIn('text',records[-1])

    def test_missing_or_reordered_full_frame_refused(self):
        rows,pins = fixture()
        for altered in (rows[:-1],list(reversed(rows))):
            with self.assertRaises(ValueError):f.calibration_mapping(altered,panels,pins)

    def test_duplicate_id_refused_even_with_rebound_order(self):
        rows,pins = fixture(); rows[3]['record_id'] = rows[1]['record_id']
        pins['source_ordered_ids_sha256'] = hashlib.sha256(json.dumps([r['record_id'] for r in rows],separators=(',',':')).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError,'duplicate'):f.calibration_mapping(rows,panels,pins)

    def test_changed_native_owner_or_source_kind_refused(self):
        rows,pins = fixture()
        for key,value in [('source_unit_id','fake'),('source_kind','unknown_singleton')]:
            changed = copy.deepcopy(rows); changed[1][key] = value
            with self.assertRaisesRegex(ValueError,'native_source'):f.calibration_mapping(changed,panels,pins)
        changed = copy.deepcopy(rows); changed[1]['original_fields']['OwnerUserId'] = -1
        with self.assertRaisesRegex(ValueError,'native_source'):f.calibration_mapping(changed,panels,pins)

    def test_changed_partition_and_unnormalized_text_refused(self):
        rows,pins = fixture()
        for key,value in [('partition','train'),('text',' extra  whitespace ' )]:
            changed = copy.deepcopy(rows); changed[1][key] = value
            with self.assertRaises(ValueError):f.calibration_mapping(changed,panels,pins)

    def test_changed_calibration_count_or_pair_universe_refused(self):
        rows,pins = fixture()
        for key in ('calibration_rows','unordered_pair_count'):
            changed = {**pins,key:pins[key]-1}
            with self.assertRaisesRegex(ValueError,'denominator'):f.calibration_mapping(rows,panels,changed)

    def test_complete_versioned_manifest_required(self):
        manifest,plan,pins = manifest_fixture()
        f.check_manifest(manifest,plan,{'software':'fixture'},pins)
        for key,value in [('schema','ccu-semantic-cache-1'),('rows',19),('shape',[19,768]),
                          ('all_rows_fresh_process_replayed_under_this_version',False),('runtime',{})]:
            with self.assertRaises(ValueError):f.check_manifest({**manifest,key:value},plan,{'software':'fixture'},pins)

    def test_missing_shard_or_subset_plan_refused(self):
        manifest,plan,pins = manifest_fixture()
        for key,value in [('shards',plan['shards'][:-1]),('kind','qualification'),
                          ('frame_row_indices',plan['frame_row_indices'][:-1])]:
            with self.assertRaisesRegex(ValueError,'full_plan'):f.check_manifest(manifest,{**plan,key:value},{'software':'fixture'},pins)

    def test_manifest_wrong_payload_source_or_scientific_claim_refused(self):
        manifest,plan,pins = manifest_fixture()
        for key,value in [('files',{}),('frame_ordered_ids_sha256','wrong'),
                          ('full_population_cache_acceptance',True),('source_authenticity_verified',True)]:
            with self.assertRaises(ValueError):f.check_manifest({**manifest,key:value},plan,{'software':'fixture'},pins)

    def test_exact_FP32_gather_and_reorder_or_change_refusal(self):
        full = np.arange(8*768,dtype=np.float32).reshape(8,768)+1
        indices = [1,3,7]; actual = full[indices].copy()
        f.verify_gather(full,actual,indices,np)
        changed = actual.copy(); changed[0,0] = np.nextafter(changed[0,0],np.float32(np.inf))
        for value,index in [(changed,indices),(actual[::-1],indices),(actual,list(reversed(indices)))]:
            with self.assertRaises(ValueError):f.verify_gather(full,value,index,np)

    def test_invalid_vector_dtype_nonfinite_zero_refused(self):
        full = np.ones((5,768),dtype=np.float32); indices = [0,4]
        for changed in (full[indices].astype(np.float64),np.zeros((2,768),dtype=np.float32),np.full((2,768),np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):f.verify_gather(full,changed,indices,np)

    def test_exclusive_private_output_and_no_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder).resolve(); out = workspace/'qualification/new'
            f.new_output(out,workspace)
            with self.assertRaises(ValueError):f.new_output(out,workspace)
            with self.assertRaises(ValueError):f.new_output(workspace/'outside',workspace)
            (workspace/'alias').symlink_to(workspace/'qualification',target_is_directory=True)
            with self.assertRaises(ValueError):f.new_output(workspace/'alias/next',workspace)

    def test_atomic_publication_no_overwrite_and_caps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'result.json'; f.publish(path,{'a':1},100)
            with self.assertRaises(FileExistsError):f.publish(path,{'a':2},100)
            self.assertEqual(f.read_json(path),{'a':1})
            with self.assertRaisesRegex(ValueError,'cap'):f.publish(Path(folder)/'large',{'a':'x'*101},100)
            self.assertFalse((Path(folder)/'large').exists())

    def test_changed_package_or_symlink_member_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder).resolve(); member = folder/'file';member.write_bytes(b'original')
            manifest = folder/'package_manifest.json';manifest.write_bytes(f.canonical({'files':{'file':f.binding(member)}}))
            wanted = f.binding(manifest)['sha256'];f.verify_package(folder,wanted)
            member.write_bytes(b'changed')
            with self.assertRaises(ValueError):f.verify_package(folder,wanted)
            member.unlink();target = folder/'target';target.write_bytes(b'original');member.symlink_to(target)
            with self.assertRaises(ValueError):f.verify_package(folder,wanted)

    def test_strict_JSON_and_exact_output_budget(self):
        for raw in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError):f.strict(raw)
        pins = f.read_json(Path(__file__).with_name('expectations.json')); caps = pins['output_caps']
        exact_npy = pins['calibration_rows']*768*4+128
        self.assertEqual(exact_npy,64499840)
        self.assertLessEqual(exact_npy+sum(caps[k] for k in ('row_metadata_bytes','input_binding_bytes','result_bytes')),caps['frame_directory_bytes'])
        self.assertEqual(caps['frame_directory_bytes'],96*1024**2)


if __name__ == '__main__':
    unittest.main()
