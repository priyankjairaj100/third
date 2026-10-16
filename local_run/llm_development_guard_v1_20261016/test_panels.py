"""Tiny invented membership fixtures only, never empirical populations."""
import copy
import unittest
import panel_core as p
from review import review_panels


class Panels(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for unit, size, role, part in [('A', 2, 'development_train', 'train'), ('B', 3, 'development_train', 'train'),
                                      ('E', 8, 'evaluation_train', 'train'), ('T', 1, 'development_test', 'test'),
                                      ('V', 1, 'evaluation_test', 'test')]:
            self.rows += [{'record_id': f'{unit}-{i}', 'source_unit_id': unit, 'role': role,
                           'partition': part, 'source_kind': 'native_account'} for i in range(size)]
        self.plan = {'development_targets': [1, 3, 99], 'development_order_salt': 'synthetic-order'}

    def test_whole_surviving_sources_and_no_borrowing(self):
        removed = [{'record_id': 'A-0', 'heldout_witness_id': 'V-0'}]
        got = p.build_panels(self.rows, removed, self.plan)
        reviewed = review_panels(self.rows, removed, got, self.plan)
        self.assertEqual(reviewed['retained_train_rows'], 4)
        self.assertEqual(got['prefixes']['99']['shortfall'], 95)
        self.assertEqual(got['development_test_ids'], ['T-0'])
        self.assertFalse(any(x.startswith('E-') for x in got['retained_development_train_ids']))

    def test_all_removed_is_visible_shortfall(self):
        removed = [{'record_id': r['record_id'], 'heldout_witness_id': 'T-0'} for r in self.rows if r['role'] == 'development_train']
        got = p.build_panels(self.rows, removed, self.plan)
        review_panels(self.rows, removed, got, self.plan)
        self.assertEqual(got['prefixes']['99']['shortfall'], 99)

    def test_partial_source_and_evaluation_leak_rejected(self):
        got = p.build_panels(self.rows, [], self.plan)
        for mutation in ('partial', 'evaluation', 'flags'):
            changed = copy.deepcopy(got)
            if mutation == 'partial': changed['prefixes']['99']['record_ids'].pop()
            if mutation == 'evaluation': changed['development_test_ids'].append('V-0')
            if mutation == 'flags': changed['effects_launched'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                review_panels(self.rows, [], changed, self.plan)

    def test_nontraining_and_duplicate_removal_rejected(self):
        for removed in ([{'record_id':'E-0'}], [{'record_id':'A-0'}, {'record_id':'A-0'}]):
            with self.assertRaises(ValueError): p.build_panels(self.rows, removed, self.plan)

    def test_metadata_only_inventory_and_original_reserve(self):
        import hashlib
        rows = []
        for i in range(100):
            unit = f'synthetic-{i}'
            h = lambda salt: int(hashlib.sha256((salt+'\0'+unit).encode()).hexdigest(),16)
            x = h('ccu-v1-source-split')/2**256
            part = 'train' if x < .7 else ('calibration' if x < .85 else 'test')
            role = 'calibration' if part == 'calibration' else ('development_' if h('third-llm-development-reserve-v1-20261016')*10<2**256 else 'evaluation_')+part
            rows.append({'original_full_row_index':i,'record_id':str(i),'source_unit_id':unit,
                         'source_kind':'native_account','partition':part,'text_sha256':'0'*64,
                         'role':role,'genuine_source_request_eligible':True})
        self.assertEqual(sum(p.inventory(rows).values()), 100)
        for key, value in (('text','private'), ('role','wrong'), ('original_full_row_index',False)):
            bad = copy.deepcopy(rows); bad[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): p.inventory(bad)


if __name__ == '__main__': unittest.main()
