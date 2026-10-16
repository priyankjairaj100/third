"""Synthetic software checks only; never evidence of natural-data effects."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import intake as t
import review
sys.path.insert(0, os.environ.get('THIRD_LOCAL_REPO', str(Path(__file__).resolve().parents[2])))
from empirical_execution.phase3 import panels


def fixture(i, owner=None):
    row = {'record_id':'fixture-'+str(i),'text':'Synthetic software fixture '+str(i),
           'original_fields':{'site':'askubuntu','OwnerUserId':owner,'Tags':['never-export-me']}}
    unit, kind = panels.source_unit(row,'askubuntu')
    row.update(source_unit_id=unit,source_kind=kind,
               partition=panels.digest_partition(panels.hash_integer('ccu-v1-source-split',unit)))
    return row


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'rows.jsonl'
        self.rows = [fixture(i, i//2+1) for i in range(40)] + [fixture(40)]
        self.expected = {'full_rows':len(self.rows),
                         'partition_counts':dict(t.Counter(r['partition'] for r in self.rows)),
                         'source_ordered_ids_sha256':hashlib.sha256(t.canonical([r['record_id'] for r in self.rows])).hexdigest()}
        self.plan = json.loads((t.HERE/'PLAN.json').read_text())
        self.write()
    def write(self):
        self.path.write_bytes(b''.join(t.canonical(r)+b'\n' for r in self.rows))
    def metadata(self):
        return t.collect_metadata(self.path,panels,self.expected)
    def test_metadata_excludes_bodies_labels(self):
        result = self.metadata(); self.assertEqual(len(result),41)
        self.assertNotIn(b'never-export-me',t.canonical(result))
        self.assertTrue(all('text' not in r and 'original_fields' not in r for r in result))
    def test_source_tamper_rejected(self):
        self.rows[0]['source_unit_id']='false'; self.write()
        with self.assertRaisesRegex(ValueError,'native_source_partition'): self.metadata()
    def test_partition_tamper_rejected(self):
        self.rows[0]['partition']='invalid'; self.write()
        with self.assertRaisesRegex(ValueError,'native_source_partition'): self.metadata()
    def test_duplicate_record_rejected(self):
        self.rows[1]['record_id']=self.rows[0]['record_id']; self.write()
        with self.assertRaisesRegex(ValueError,'unique_record'): self.metadata()
    def test_missing_or_reordered_rows_rejected(self):
        self.rows.reverse(); self.write()
        with self.assertRaisesRegex(ValueError,'source_order'): self.metadata()
        self.rows.pop(); self.write()
        with self.assertRaisesRegex(ValueError,'complete_population'): self.metadata()
    def test_exact_fraction_boundary(self):
        cutoff=(2**256+9)//10
        self.assertEqual(t.reserve_role('train',cutoff-1),'development_train')
        self.assertEqual(t.reserve_role('train',cutoff),'evaluation_train')
        self.assertEqual(t.reserve_role('test',0),'development_test')
        self.assertEqual(t.reserve_role('calibration',0),'calibration')
        for bad in (True,-1,2**256):
            with self.assertRaises(ValueError): t.reserve_role('train',bad)
    def test_whole_sources_and_unknown_kept(self):
        rows, counts, _ = t.allocate(self.metadata(),self.plan,panels)
        groups = {}
        for row in rows:
            groups.setdefault(row['source_unit_id'],set()).add(row['role'])
        self.assertTrue(all(len(v)==1 for v in groups.values()))
        self.assertEqual(sum(c['rows'] for c in counts.values()),41)
        unknown=[r for r in rows if r['source_kind']=='unknown_singleton']
        self.assertEqual(len(unknown),1); self.assertFalse(unknown[0]['genuine_source_request_eligible'])
    def test_prefix_order_invariant_whole_boundary_and_shortfall(self):
        rows=self.metadata(); a=t.whole_prefix(rows,1,'fixture-order',panels)
        self.assertEqual(a,t.whole_prefix(list(reversed(rows)),1,'fixture-order',panels))
        self.assertEqual(set(a['record_ids']),{r['record_id'] for r in rows if r['source_unit_id'] in a['source_ids']})
        b=t.whole_prefix(rows,42,'fixture-order',panels)
        self.assertEqual(b['record_ids'][:a['actual']],a['record_ids']); self.assertEqual(b['shortfall'],1)
    def test_overlapping_source_or_duplicate_allocation_rejected(self):
        rows=self.metadata(); rows[1]['partition']='test' if rows[0]['partition']!='test' else 'train'
        with self.assertRaisesRegex(ValueError,'source_role_overlap'): t.allocate(rows,self.plan,panels)
        rows=self.metadata(); rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'duplicate_allocation'): t.allocate(rows,self.plan,panels)
    def test_json_and_changed_fraction_refused(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError): t.strict(raw)
        self.plan['development_reserve_fraction']=[1,5]
        with self.assertRaisesRegex(ValueError,'fixed_reserve_fraction'): t.allocate(self.metadata(),self.plan,panels)
    def audit_fixture(self, tamper=False):
        rows,counts,prefixes=t.allocate(self.metadata(),self.plan,panels)
        if tamper:
            row=next(r for r in rows if r['partition']=='train')
            row['role']='evaluation_train' if row['role']=='development_train' else 'development_train'
        folder=Path(self.tmp.name).resolve()/'run';folder.mkdir()
        for name,value in [('allocation.private.json',rows),('development_prefixes.private.json',prefixes)]:
            (folder/name).write_bytes(t.canonical(value))
        result={'rows':len(rows),'allocation':counts,'outputs':{name:t.binding(folder/name) for name in
            ('allocation.private.json','development_prefixes.private.json')}}
        result.update({k:False for k in ('development_threshold_accepted','semantic_guard_executed',
            'post_guard_panels_frozen','effect_experiments_started','human_validity_claimed',
            'original_primary_input_acceptance','confirmatory_study_ready')})
        (folder/'result.json').write_bytes(t.canonical(result));return folder
    def test_independent_auditor_matches_whole_source_fixture(self):
        self.assertTrue(review.check(self.audit_fixture(),self.plan)['passed'])
    def test_independent_auditor_detects_role_leak_despite_new_payload_hash(self):
        with self.assertRaises(AssertionError):review.check(self.audit_fixture(tamper=True),self.plan)


if __name__ == '__main__':
    unittest.main()
