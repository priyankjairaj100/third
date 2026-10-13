"""Independent tiny software fixtures; no model forwards or empirical records."""
import copy
from collections import Counter
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

import common
import driver
import worker


FINGERPRINT = 'a' * 64


def digest_list(rows):
    stream = common.c.ListDigest()
    for row in rows:
        stream.add(row)
    return stream.finish()


def write_lines(path, rows):
    path.write_bytes(b''.join(common.c.canonical(row) + b'\n' for row in rows))


def plan_fixture(kind='full', count=40):
    indices = list(range(count)) if kind == 'full' else list(range(19))
    shard_count = 32 if kind == 'full' else 4
    starts = [i * len(indices) // shard_count for i in range(shard_count + 1)]
    return common.seal({'schema': 'third-e5-cpu-shard-plan-v1', 'fingerprint': FINGERPRINT,
        'kind': kind, 'frame_row_indices': indices,
        'shards': [{'index': i, 'start': starts[i], 'stop': starts[i + 1],
                    'modeled_forward_seconds': 1.0} for i in range(shard_count)]})


def reseal(plan):
    return common.seal({key: value for key, value in plan.items() if key != 'sha256'})


class PlanIntegrityTests(unittest.TestCase):
    def test_full_and_qualification_exact_coverage(self):
        for kind in ('full', 'qualification'):
            common.validate_plan(plan_fixture(kind), FINGERPRINT, 40)

    def test_recomputed_seal_does_not_allow_gaps_overlap_empty_or_missing_shards(self):
        base = plan_fixture()
        mutations = []
        for field, value in [('start', 2), ('start', 0), ('stop', 1), ('index', 3)]:
            bad = copy.deepcopy(base)
            bad['shards'][1][field] = value
            mutations.append(bad)
        bad = copy.deepcopy(base); bad['shards'].pop(); mutations.append(bad)
        bad = copy.deepcopy(base); bad['shards'][-1]['stop'] -= 1; mutations.append(bad)
        for bad in mutations:
            with self.subTest(shards=bad['shards'][:2]), self.assertRaises(ValueError):
                common.validate_plan(reseal(bad), FINGERPRINT, 40)

    def test_wrong_fingerprint_reordered_missing_or_duplicate_frame_rejected(self):
        base = plan_fixture()
        with self.assertRaises(ValueError):
            common.validate_plan(base, 'b' * 64, 40)
        for indices in ([1, 0] + list(range(2, 40)), list(range(39)), [0] + list(range(39)), list(range(41))):
            with self.subTest(indices=indices[:3]), self.assertRaises(ValueError):
                common.validate_plan(reseal({**base, 'frame_row_indices': indices}), FINGERPRINT, 40)
        wrong = copy.deepcopy(base); wrong['shards'][0]['modeled_forward_seconds'] = 2.0
        with self.assertRaises(ValueError):
            common.validate_plan(wrong, FINGERPRINT, 40)


class FullFrameTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.view, self.census = self.root / 'view', self.root / 'census'
        self.view.mkdir(); self.census.mkdir()
        self.rows = [{'record_id': 'software-' + str(i), 'text': 'Synthetic fixture ' + str(i) + ' \u2028 full text',
                      'partition': 'calibration', 'source_unit_id': 'synthetic-source-' + str(i)} for i in range(6)]
        self.ledger = [{'row_index': i, 'record_id': row['record_id'], 'partition': row['partition'],
                        'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest()} for i, row in enumerate(self.rows)]
        write_lines(self.view / 'records.jsonl', self.rows)
        write_lines(self.census / 'row_token_chunk_ledger.private.jsonl', self.ledger)
        self.ctx = {'view': self.view, 'census_dir': self.census,
            'aggregate': {'totals': {'rows': len(self.rows)}, 'partition_counts': dict(Counter(row['partition'] for row in self.rows)),
                'full_prepared_rows_sha256': digest_list(self.rows),
                'text_binding': digest_list([{k: row[k] for k in ('record_id', 'text')} for row in self.rows]),
                'ordered_record_ids_sha256': digest_list([row['record_id'] for row in self.rows])}}

    def test_selected_rows_preserve_objects_unicode_order_and_complete_frame_binding(self):
        selected, entries = common.rows_for_indices(self.ctx, [1, 4])
        self.assertEqual(selected, [self.rows[1], self.rows[4]])
        self.assertEqual(entries, [self.ledger[1], self.ledger[4]])
        self.assertIn('\u2028', selected[0]['text'])

    def test_changed_unselected_source_row_and_reordering_are_not_hidden(self):
        for mutation in ('unselected_text', 'reorder', 'duplicate', 'extra'):
            changed = copy.deepcopy(self.rows)
            if mutation == 'unselected_text': changed[5]['text'] += ' changed'
            if mutation == 'reorder': changed[0], changed[5] = changed[5], changed[0]
            if mutation == 'duplicate': changed[5] = changed[0]
            if mutation == 'extra': changed.append({**changed[5], 'record_id': 'additional-software-row'})
            write_lines(self.view / 'records.jsonl', changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                common.rows_for_indices(self.ctx, [1, 4])

    def test_selected_census_wrong_order_text_hash_or_missing_row_rejected(self):
        for mutation in ('row_index', 'record_id', 'text_sha256', 'missing'):
            changed = copy.deepcopy(self.ledger)
            if mutation == 'missing': changed = changed[:4]
            else: changed[4][mutation] = -1 if mutation == 'row_index' else 'wrong'
            write_lines(self.census / 'row_token_chunk_ledger.private.jsonl', changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                common.rows_for_indices(self.ctx, [1, 4])


class QualificationIntegrityTests(unittest.TestCase):
    def test_boolean_only_or_failed_receipt_is_not_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ctx = {'workspace': root, 'fingerprint': FINGERPRINT, 'watch': common.c.Watch()}
            path = root / 'qualification.json'
            report = {'schema': 'third-e5-sharded-cpu-qualification-v1', 'passed': True,
                'fingerprint': FINGERPRINT, 'golden_all_vector_bytes_equal': True,
                'golden_all_chunk_log_rows_equal': True, 'all_shards_fresh_process_replayed': True,
                'evidence_files': {}}
            path.write_text(json.dumps(common.seal(report)))
            ctx['qualification_sha256'] = common.binding(path)['sha256']
            with self.assertRaisesRegex(ValueError, 'qualification_evidence_inventory_incomplete'):
                common.require_qualification(ctx, path)
            path.write_text(json.dumps(common.seal({**report, 'passed': False})))
            ctx['qualification_sha256'] = common.binding(path)['sha256']
            ctx['watch'] = common.c.Watch()
            with self.assertRaises(ValueError):
                common.require_qualification(ctx, path)

    def test_reviewed_pin_and_seal_are_both_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'qualification.json'
            path.write_text(json.dumps({'schema': 'third-e5-sharded-cpu-qualification-v1', 'passed': True}))
            ctx = {'workspace': root, 'fingerprint': FINGERPRINT, 'watch': common.c.Watch()}
            with self.assertRaisesRegex(ValueError, 'independently_reviewed_qualification_SHA_required'):
                common.require_qualification(ctx, path)
            ctx['qualification_sha256'] = common.binding(path)['sha256']
            with self.assertRaisesRegex(ValueError, 'invalid_seal'):
                common.require_qualification(ctx, path)


class RangeAndPublicationTests(unittest.TestCase):
    def test_long_document_remains_whole_and_all_rows_covered_deterministically(self):
        costs = [0.25] * 17 + [10000.0] + [0.5] * 22
        ranges = driver.weighted_ranges(costs, 32)
        self.assertEqual(ranges, driver.weighted_ranges(costs, 32))
        self.assertEqual([i for s in ranges for i in range(s['start'], s['stop'])], list(range(len(costs))))
        self.assertEqual(sum(s['start'] <= 17 < s['stop'] for s in ranges), 1)
        self.assertTrue(all(s['start'] < s['stop'] for s in ranges))
        self.assertEqual(sum(s['modeled_forward_seconds'] for s in ranges), sum(costs))

    def test_invalid_costs_or_more_shards_than_documents_refused(self):
        for costs, count in [([], 1), ([1], 2), ([1, 0], 1), ([1, -1], 1),
                             ([float('nan')], 1), ([float('inf')], 1)]:
            with self.subTest(costs=costs, count=count), self.assertRaises(ValueError):
                driver.weighted_ranges(costs, count)

    def test_completion_cannot_replace_existing_or_racing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'completion.json'
            common.atomic_completion(path, {'passed': True})
            initial = path.read_bytes()
            with self.assertRaises(ValueError):
                common.atomic_completion(path, {'passed': False})
            self.assertEqual(path.read_bytes(), initial)
            raced = Path(temporary) / 'raced.json'
            original_write = common.write_json
            def publish_other_after_pending(destination, value):
                original_write(destination, value)
                raced.write_bytes(b'other completed result\n')
            with mock.patch.object(common, 'write_json', side_effect=publish_other_after_pending):
                with self.assertRaises(FileExistsError):
                    common.atomic_completion(raced, {'passed': True})
            self.assertEqual(raced.read_bytes(), b'other completed result\n')


class ShardArtifactTests(unittest.TestCase):
    """Every vector is fabricated software data, not an encoder result."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.ctx = {'workspace': self.root, 'np': np, 'fingerprint': FINGERPRINT,
                    'runtime': {'fixture': True}, 'watch': common.c.Watch()}
        self.rows = [{'record_id': 'fixture-' + str(i), 'text': 'software sample ' + str(i),
                      'partition': 'calibration'} for i in range(19)]
        self.entries = [{'record_id': row['record_id'], 'row_index': i, 'partition': 'calibration',
                         'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                         'detail': {'chunks': 1, 'content_tokens': 3}, 'batch_count': 1}
                        for i, row in enumerate(self.rows)]
        self.plan = reseal({**plan_fixture('qualification'),
                           'frame_ordered_ids_sha256': hashlib.sha256(common.c.canonical([r['record_id'] for r in self.rows])).hexdigest(),
                           'full_source_frame_sha256': digest_list(self.rows)})
        self.vectors = np.arange(19 * 768, dtype=np.float32).reshape(19, 768) / np.float32(1000.0)
        self.encoded, self.replayed = [], []
        for i, shard in enumerate(self.plan['shards']):
            self.encoded.append(self.write_shard(i, 'encode'))
            self.replayed.append(self.write_shard(i, 'replay', self.encoded[i]))

    def write_shard(self, index, kind, encoded=None):
        shard = self.plan['shards'][index]
        start, stop = shard['start'], shard['stop']
        path = self.root / (kind + '-' + str(index)); path.mkdir()
        np.save(path / 'vectors.npy', self.vectors[start:stop])
        write_lines(path / 'chunk_log.jsonl', [worker.expected_log(self.rows[i], self.entries[i], i) for i in range(start, stop)])
        receipt = {'schema': worker.SCHEMA, 'kind': kind, 'passed': True,
                   'plan_sha256': self.plan['sha256'], 'fingerprint': FINGERPRINT, 'shard_index': index,
                   'start': start, 'stop': stop, 'rows': stop - start,
                   'content_tokens': sum(entry['detail']['content_tokens'] for entry in self.entries[start:stop]),
                   'chunks': sum(entry['detail']['chunks'] for entry in self.entries[start:stop]),
                   'model_forward_batches': sum(entry['batch_count'] for entry in self.entries[start:stop]),
                   'files': {name: common.binding(path / name) for name in sorted(worker.FILES)},
                   **{name: False for name in worker.FALSE_FLAGS}}
        if encoded is not None:
            receipt.update(encoded_completion=common.binding(encoded / 'completion.json'),
                           exact_vector_bytes_equal=True, complete_chunk_log_equal=True)
        (path / 'completion.json').write_text(json.dumps(receipt))
        return path

    def refresh_bindings(self, path):
        receipt = json.loads((path / 'completion.json').read_text())
        receipt['files'] = {name: common.binding(path / name) for name in sorted(worker.FILES)}
        (path / 'completion.json').write_text(json.dumps(receipt))

    def validate(self, index=0):
        shard = self.plan['shards'][index]
        start, stop = shard['start'], shard['stop']
        return worker.validate_encoded(self.ctx, self.plan, index, self.encoded[index],
                                       self.rows[start:stop], self.entries[start:stop])

    def test_complete_encoding_and_replay_are_accepted(self):
        for i in range(4):
            receipt = self.validate(i)
            driver.validate_replay_receipt(self.ctx, self.plan, i, self.replayed[i], self.encoded[i], receipt)

    def test_partial_and_unexpected_shard_artifacts_cannot_resume(self):
        path = self.encoded[0]
        completion = (path / 'completion.json').read_bytes()
        (path / 'completion.json').unlink()
        with self.assertRaisesRegex(ValueError, 'unexpected_or_partial_shard_inventory'):
            self.validate()
        (path / 'completion.json').write_bytes(completion)
        (path / 'partial.tmp').write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError, 'unexpected_or_partial_shard_inventory'):
            self.validate()

    def test_wrong_hash_is_not_a_resumable_shard(self):
        with (self.encoded[0] / 'vectors.npy').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'encoded_file_hash_mismatch'):
            self.validate()

    def test_resumed_receipt_must_bind_plan_range_and_claim_boundary(self):
        path = self.encoded[0] / 'completion.json'
        initial = json.loads(path.read_text())
        for field, value in [('passed', False), ('plan_sha256', 'f' * 64), ('fingerprint', 'b' * 64),
                             ('start', 1), ('stop', 5), ('shard_index', True),
                             ('model_forward_batches', 999), ('content_tokens', 0), ('chunks', 0),
                             ('full_population_cache_acceptance', True)]:
            path.write_text(json.dumps({**initial, field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate()

    def test_resealed_logs_cannot_change_order_identity_or_content_binding(self):
        path = self.encoded[0]
        logs = worker._read_logs(path / 'chunk_log.jsonl')
        for field in ('record_id', 'text_sha256', 'row_index', 'content_tokens'):
            changed = copy.deepcopy(logs)
            changed[0][field] = -1 if isinstance(changed[0][field], int) else 'changed'
            write_lines(path / 'chunk_log.jsonl', changed); self.refresh_bindings(path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'encoded_complete_chunk_log_mismatch'):
                self.validate()
        write_lines(path / 'chunk_log.jsonl', logs[::-1]); self.refresh_bindings(path)
        with self.assertRaises(ValueError):
            self.validate()

    def test_rehashed_wrong_dtype_shape_nonfinite_and_zero_vectors_fail(self):
        path = self.encoded[0]
        count = self.plan['shards'][0]['stop']
        bad_arrays = [np.ones((count, 768), dtype=np.float64), np.ones((count, 767), dtype=np.float32),
                      np.full((count, 768), np.nan, dtype=np.float32), np.zeros((count, 768), dtype=np.float32)]
        for bad in bad_arrays:
            np.save(path / 'vectors.npy', bad); self.refresh_bindings(path)
            with self.subTest(dtype=str(bad.dtype), shape=bad.shape), self.assertRaises(ValueError):
                self.validate()

    def test_replay_flags_do_not_substitute_for_vector_equality(self):
        path = self.replayed[0]
        values = np.load(path / 'vectors.npy'); values[0, 767] = np.nextafter(values[0, 767], np.float32(np.inf))
        np.save(path / 'vectors.npy', values); self.refresh_bindings(path)
        with self.assertRaisesRegex(ValueError, 'cache_vector_bytes_differ'):
            driver.validate_replay_receipt(self.ctx, self.plan, 0, path, self.encoded[0], self.validate())

    def test_full_shard_cannot_resume_under_different_qualification_binding(self):
        # Range validation is separately tested; this targets per-shard gate identity.
        plan = {**self.plan, 'kind': 'full', 'qualification_result': {'sha256': '1' * 64, 'bytes': 5}}
        path = self.encoded[0] / 'completion.json'
        receipt = json.loads(path.read_text())
        receipt['qualification_result'] = plan['qualification_result']
        path.write_text(json.dumps(receipt))
        shard = plan['shards'][0]
        kwargs = {'rows': self.rows[:shard['stop']], 'expected': self.entries[:shard['stop']]}
        worker.validate_encoded(self.ctx, plan, 0, self.encoded[0], **kwargs)
        for bad in ({'sha256': '2' * 64, 'bytes': 5}, {'sha256': '1' * 64, 'bytes': 6}, None):
            self.ctx['watch'] = common.c.Watch()
            path.write_text(json.dumps({**receipt, 'qualification_result': bad}))
            with self.subTest(binding=bad), self.assertRaisesRegex(ValueError, 'encoded_shard_qualification_binding_mismatch'):
                worker.validate_encoded(self.ctx, plan, 0, self.encoded[0], **kwargs)

    def test_replay_requires_exact_work_counts_and_full_qualification_binding(self):
        encoded = self.validate()
        path = self.replayed[0] / 'completion.json'
        receipt = json.loads(path.read_text())
        for key in ('content_tokens', 'chunks', 'model_forward_batches'):
            self.ctx['watch'] = common.c.Watch()
            path.write_text(json.dumps({**receipt, key: receipt[key] + 1}))
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'replay_work_counters_mismatch'):
                driver.validate_replay_receipt(self.ctx, self.plan, 0, self.replayed[0], self.encoded[0], encoded)
        plan = {**self.plan, 'kind': 'full', 'qualification_result': {'sha256': '1' * 64, 'bytes': 5}}
        for value in (None, {'sha256': '2' * 64, 'bytes': 5}, {'sha256': '1' * 64, 'bytes': 6}):
            self.ctx['watch'] = common.c.Watch()
            path.write_text(json.dumps({**receipt, 'qualification_result': value}))
            with self.subTest(binding=value), self.assertRaisesRegex(ValueError, 'full_replay_qualification_binding_mismatch'):
                driver.validate_replay_receipt(self.ctx, plan, 0, self.replayed[0], self.encoded[0], encoded)

    def test_compare_rejects_reordered_or_truncated_complete_logs(self):
        path = self.replayed[0]
        logs = worker._read_logs(path / 'chunk_log.jsonl')
        for changed in (logs[::-1], logs[:-1]):
            write_lines(path / 'chunk_log.jsonl', changed)
            with self.assertRaises(ValueError):
                driver.compare_cache_files(self.encoded[0], path)

    def assemble(self, encoded=None, replayed=None):
        with mock.patch.object(common, 'rows_for_indices', return_value=(self.rows, self.entries)), \
                mock.patch.object(common, 'verify_final', side_effect=lambda ctx: ctx['watch'].verify()):
            return driver.assemble(self.ctx, self.plan, encoded or self.encoded,
                                   replayed or self.replayed, self.root / 'qualification' / 'assembled')

    def test_assembly_preserves_every_float32_byte_and_original_log_order(self):
        report = self.assemble()
        path = self.root / 'qualification' / 'assembled'
        self.assertEqual(np.load(path / 'vectors.npy').tobytes(), self.vectors.tobytes())
        self.assertEqual(worker._read_logs(path / 'chunk_log.jsonl'),
                         [worker.expected_log(row, entry, i) for i, (row, entry) in enumerate(zip(self.rows, self.entries))])
        self.assertTrue(report['passed']); self.assertFalse(report['full_population_cache_acceptance'])
        common.verify_seal(report)

    def test_assembly_rejects_missing_duplicate_and_reordered_shards_before_output(self):
        cases = [self.encoded[:-1], [self.encoded[0]] * 4, list(reversed(self.encoded))]
        for encoded in cases:
            with self.subTest(paths=[p.name for p in encoded]), self.assertRaises(ValueError):
                self.assemble(encoded=encoded)
            self.assertFalse((self.root / 'qualification' / 'assembled').exists())

    def test_assembly_refuses_frame_order_mismatch_even_with_matching_shards(self):
        self.plan = reseal({**self.plan, 'frame_ordered_ids_sha256': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'assembly_frame_order_mismatch'):
            self.assemble()

    def test_assembled_slice_cannot_differ_from_separately_valid_shard(self):
        self.assemble()
        assembled = self.root / 'qualification' / 'assembled'
        start, stop = self.plan['shards'][1]['start'], self.plan['shards'][1]['stop']
        driver.compare_shard_slice(assembled, self.encoded[1], start, stop)
        values = np.load(assembled / 'vectors.npy')
        values[start, 767] = np.nextafter(values[start, 767], np.float32(np.inf))
        np.save(assembled / 'vectors.npy', values)
        with self.assertRaisesRegex(ValueError, 'assembled_shard_vector_bytes_differ'):
            driver.compare_shard_slice(assembled, self.encoded[1], start, stop)
        np.save(assembled / 'vectors.npy', self.vectors)
        logs = worker._read_logs(assembled / 'chunk_log.jsonl')
        logs[start], logs[start + 1] = logs[start + 1], logs[start]
        write_lines(assembled / 'chunk_log.jsonl', logs)
        with self.assertRaisesRegex(ValueError, 'assembled_shard_log_slice_differs'):
            driver.compare_shard_slice(assembled, self.encoded[1], start, stop)


class FrozenArithmeticWrapperTests(unittest.TestCase):
    """Use frozen NumPy arithmetic with a synthetic tokenizer/backend, no model."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(common.__file__).resolve().parents[2] / 'empirical_execution'))
        from phase4 import embeddings
        cls.e = embeddings

    class Tokenizer:
        def encode(self, text, *, add_special_tokens, truncation=False):
            assert not add_special_tokens and not truncation
            if text == 'query: ': return [11, 12, 13]
            return list(range(20, 20 + int(text.split(':')[1])))
        def num_special_tokens_to_add(self, pair=False): return 2
        def prepare_for_model(self, ids, **kwargs):
            assert kwargs['truncation'] is False and kwargs['padding'] is False
            return {'input_ids': [1] + ids + [2], 'attention_mask': [1] * (len(ids) + 2)}

    class Backend:
        def __init__(self, tokenizer, fail=False):
            self.tokenizer, self.max_length = tokenizer, 512
            self.model_bytes, self.model_buffer_bytes = 0, 0
            self.calls, self.fail = [], fail
        def __call__(self, parts):
            self.calls.append([part['content_start'] for part in parts])
            if self.fail and len(self.calls) == 2: raise RuntimeError('synthetic_backend_failure')
            return np.asarray([np.arange(768, dtype=np.float32) * np.float32(0.001)
                               + np.float32(1 + part['content_start'] / 257)
                               for part in parts], dtype=np.float32)

    def run_fixture(self, root, fail=False):
        tokenizer = self.Tokenizer()
        # Prefix must exactly match the frozen E5 registry, without test inference.
        self.assertEqual(self.e.ENCODERS[self.e.E5], 'query: ')
        rows = [{'record_id': 'software-two-chunks', 'text': 'tokens:300', 'partition': 'calibration'},
                {'record_id': 'software-nine-chunks', 'text': 'tokens:2200', 'partition': 'calibration'}]
        entries, reference = [], []
        for i, row in enumerate(rows):
            vector, detail = self.e.encode_document(tokenizer, self.Backend(tokenizer), row['text'], self.e.E5, 512, batch_size=8)
            reference.append(vector)
            count = detail.pop('batch_count')
            entries.append({'row_index': i, 'record_id': row['record_id'],
                            'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                            'detail': detail, 'batch_count': count})
        plan = {'kind': 'qualification', 'sha256': 'c' * 64, 'frame_row_indices': [0, 1],
                'shards': [{'index': 0, 'start': 0, 'stop': 2}]}
        ctx = {'workspace': root, 'np': np, 'e': self.e, 'fingerprint': FINGERPRINT,
               'runtime': {'synthetic_software_only': True}, 'model_dir': root / 'no-model',
               'tokenizer_dir': root / 'no-tokenizer', 'watch': common.c.Watch()}
        adapter = self.Backend(tokenizer, fail)
        out = root / 'qualification' / 'synthetic-worker'
        args = SimpleNamespace(mode='encode', plan=root / 'unused-plan', shard_index=0,
                               out=out, encoded_dir=None, qualification_result=None)
        with mock.patch.object(common, 'context', return_value=ctx), \
                mock.patch.object(common, 'load_plan', return_value=plan), \
                mock.patch.object(common, 'rows_for_shard', return_value=(rows, entries)), \
                mock.patch.object(common, 'verify_final', side_effect=lambda context: context['watch'].verify()), \
                mock.patch.object(self.e, 'LocalTransformer', return_value=adapter), \
                mock.patch.object(worker, 'runtime_checks', return_value={'synthetic_backend': True}):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'synthetic_backend_failure'):
                    worker.run(args)
                self.assertFalse((out / 'completion.json').exists())
                self.assertFalse(json.loads((out / 'failure.json').read_text())['passed'])
                with self.assertRaisesRegex(ValueError, 'unexpected_or_partial_shard_inventory'):
                    worker.validate_encoded(ctx, plan, 0, out, rows, entries)
            else:
                receipt = worker.run(args)
                self.assertEqual(np.load(out / 'vectors.npy').tobytes(), np.asarray(reference).tobytes())
                self.assertEqual(worker._read_logs(out / 'chunk_log.jsonl'),
                                 [worker.expected_log(row, entry, i) for i, (row, entry) in enumerate(zip(rows, entries))])
                self.assertEqual(adapter.calls, [[0, 256], list(range(0, 2048, 256)), [2048]])
                self.assertEqual(receipt['model_forward_batches'], 3)
                worker.validate_encoded(ctx, plan, 0, out, rows, entries)

    def test_worker_preserves_frozen_coordinate_arithmetic_and_document_local_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.run_fixture(Path(temporary).resolve())

    def test_failed_forward_never_publishes_resumable_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.run_fixture(Path(temporary).resolve(), fail=True)

    def test_prospective_calibration_selection_and_batch_tail_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rows = [{'record_id': f'fixture-bucket-{k}-candidate-{i}', 'text': f'tokens:{256 * k - 3}',
                     'partition': 'calibration'} for k in range(1, 9) for i in range(3)]
            rows += [{'record_id': f'fixture-long-{k}-candidate-{i}', 'text': f'tokens:{256 * k - 3}',
                      'partition': 'calibration'} for k in (9, 17, 51) for i in range(2)]
            rows += [{'record_id': 'fixture-' + p, 'text': 'tokens:10', 'partition': p} for p in ('train', 'test')]
            ledger = root / 'row_token_chunk_ledger.private.jsonl'
            with ledger.open('w') as stream:
                aggregate = common.c.census_rows(rows, self.Tokenizer(), self.e.chunk_inputs,
                                                self.e.E5, 512, 261, stream)
            aggregate = json.loads(json.dumps(aggregate))
            ctx = {'census_dir': root, 'aggregate': aggregate, 'fingerprint': FINGERPRINT,
                   'pins': {'rows': len(rows), 'census_files': {ledger.name: common.binding(ledger)},
                            'budget_files': {'budget_projection.json': {'sha256': 'e' * 64, 'bytes': 1}}},
                   'projection': {'model_forward_seconds_by_batch_size': {str(k): {'max': k * 0.125} for k in range(1, 9)}},
                   'qualified_result_binding': {'sha256': 'f' * 64, 'bytes': 1}}
            plan = driver.make_plan(ctx, 'qualification')
            entries = list(common.c.physical_rows(ledger))
            selected = []
            for k in list(range(1, 9)) + [9, 17, 51]:
                candidates = [r for r in entries if r['partition'] == 'calibration' and r['detail']['chunks'] == k]
                ranked = sorted(candidates, key=lambda r: (hashlib.sha256((common.b.SALT + '\0' + r['record_id']).encode()).hexdigest(), r['record_id']))
                selected += ranked[:2 if k <= 8 else 1]
            self.assertEqual(plan['frame_row_indices'], sorted(r['row_index'] for r in selected))
            self.assertEqual(plan['documents'], 19)
            self.assertEqual(plan['chunks_per_pass'], 149)
            self.assertEqual(sum(s['modeled_forward_seconds'] for s in plan['shards']), 149 * 0.125)
            self.assertEqual(plan, driver.make_plan(ctx, 'qualification'))
            full = driver.make_plan(ctx, 'full')
            self.assertEqual(full['frame_row_indices'], list(range(len(rows))))
            self.assertEqual(len(full['shards']), 32)
            self.assertEqual(full['qualification_result'], ctx['qualified_result_binding'])
            reordered = entries[:]; reordered[0], reordered[1] = reordered[1], reordered[0]
            write_lines(ledger, reordered)
            with self.assertRaisesRegex(ValueError, 'census_row_order_mismatch'):
                driver.make_plan(ctx, 'qualification')


class GoldenModuleLaunchTests(unittest.TestCase):
    def test_real_module_entry_avoids_direct_script_dependency_shadowing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repo = root / 'repo'; phase = repo / 'empirical_execution/phase4'; phase.mkdir(parents=True)
            site = repo / 'fixture_site/requests'; site.mkdir(parents=True)
            (site / '__init__.py').write_text('from . import exceptions\n')
            (site / 'exceptions.py').write_text('class HTTPError(Exception): pass\n')
            (phase / 'requests.py').write_text('WORKLOAD_MODULE = True\n')
            # Mimic only the frozen file's entry-path insertion. Fixture package
            # substitutes for installed requests; no real model/library load.
            (phase / 'embeddings.py').write_text(
                'from pathlib import Path\nimport sys,os,json\n'
                'repo=Path(__file__).resolve().parents[2]\n'
                'sys.path.insert(0,str(repo/"empirical_execution"))\n'
                'sys.path.insert(2,str(repo/"fixture_site"))\n'
                'from requests.exceptions import HTTPError\nimport requests\n'
                'print(json.dumps({"requests_origin":requests.__file__,"cwd":os.getcwd(),"pythonpath":os.environ.get("PYTHONPATH")}))\n')
            with mock.patch.dict(os.environ, {'PYTHONPATH': str(root / 'uncontrolled')}):
                with self.assertRaisesRegex(ValueError, 'fresh_child_failed'):
                    driver.run_child([sys.executable, str(phase / 'embeddings.py'), 'encode'], root / 'direct', cwd=repo)
                driver.run_child([sys.executable, '-m', 'phase4.embeddings', 'encode'], root / 'module',
                                 cwd=repo, pythonpath=repo / 'empirical_execution')
            self.assertIn("'requests' is not a package", (root / 'direct/stderr.txt').read_text())
            output = json.loads((root / 'module/stdout.txt').read_text())
            self.assertEqual(Path(output['requests_origin']), site / '__init__.py')
            self.assertEqual(output['cwd'], str(repo))
            self.assertEqual(output['pythonpath'], str(repo / 'empirical_execution'))
            receipt = json.loads((root / 'module/process.json').read_text())
            self.assertEqual(receipt['cwd'], output['cwd'])
            self.assertEqual(receipt['pythonpath'], output['pythonpath'])
            self.assertEqual(receipt['returncode'], 0)

    def test_golden_receipt_rejects_wrong_entry_directory_environment_or_controls(self):
        repo, root = Path('/software-fixture/repo'), Path('/software-fixture/qualification')
        ctx = {'repo': repo, 'assets_path': Path('/software-fixture/models/assets.json'),
               'model_dir': Path('/software-fixture/models/model'), 'tokenizer_dir': Path('/software-fixture/models/tokenizer')}
        command = [sys.executable, '-m', 'phase4.embeddings', 'encode',
                   '--records', str(root / 'engineering_subset/records.jsonl'),
                   '--preparation-audit', str(root / 'engineering_subset/audit.json'),
                   '--asset-manifest', str(ctx['assets_path']), '--model-dir', str(ctx['model_dir']),
                   '--tokenizer-dir', str(ctx['tokenizer_dir']), '--out', str(root / 'golden'),
                   '--batch-size', '8', '--threads', '1', '--evidence-role', 'engineering_nonconfirmatory']
        receipt = {'command': command, 'cwd': str(repo), 'pythonpath': str(repo / 'empirical_execution')}
        common.validate_golden_launch(ctx, root, receipt)
        cases = [{**receipt, 'cwd': str(repo / 'empirical_execution/phase4')},
                 {**receipt, 'pythonpath': str(repo / 'empirical_execution') + os.pathsep + '/unexpected'},
                 {**receipt, 'pythonpath': None},
                 {**receipt, 'command': [sys.executable, str(repo / 'empirical_execution/phase4/embeddings.py'), 'encode', *command[4:]]}]
        for flag, value in [('--records', '/wrong'), ('--out', '/wrong'), ('--batch-size', '16'), ('--threads', '2'),
                            ('--preparation-audit', '/wrong'), ('--asset-manifest', '/wrong'), ('--model-dir', '/wrong'),
                            ('--tokenizer-dir', '/wrong'), ('--evidence-role', 'confirmatory_calibration')]:
            changed = list(command); changed[changed.index(flag) + 1] = value
            cases.append({**receipt, 'command': changed})
        cases.append({**receipt, 'command': command + ['--threads', '1']})
        for case in cases:
            with self.subTest(receipt=case), self.assertRaises(ValueError):
                common.validate_golden_launch(ctx, root, case)


if __name__ == '__main__':
    unittest.main()
