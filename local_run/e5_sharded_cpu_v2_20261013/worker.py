#!/usr/bin/env python3
"""Fresh-process frozen CPU E5 shard encoding/replay; never submits work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

try:
    import common
except Exception as error:
    print(json.dumps({'schema': 'third-e5-cpu-worker-failure-v1', 'passed': False,
        'stage': 'binding_helper_import', 'error_type': type(error).__name__,
        'error': str(error), 'original_route_acceptance': False,
        'confirmatory_study_ready': False}), file=sys.stderr)
    raise SystemExit(2)

SCHEMA = 'third-e5-cpu-shard-completion-v1'
FILES = {'vectors.npy', 'chunk_log.jsonl'}
FALSE_FLAGS = ('source_authenticity_verified', 'primary_input_acceptance',
               'original_route_acceptance', 'full_population_cache_acceptance',
               'confirmatory_study_ready')
require = common.require


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate_json_key')
            value[key] = item
        return value
    def reject(_):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def expected_log(row, census, frame_index):
    """The exact frozen encode_local chunk-log object, indexed in this plan."""
    require(row['record_id'] == census['record_id'], 'source_census_record_mismatch')
    text_sha = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
    require(text_sha == census['text_sha256'], 'source_census_text_mismatch')
    detail = dict(census['detail'])
    require('batch_count' not in detail, 'unexpected_census_detail_batch_count')
    detail['batch_count'] = census['batch_count']
    require(type(detail['chunks']) is int and detail['chunks'] > 0
            and detail['batch_count'] == (detail['chunks'] + 7) // 8, 'census_batch_count_mismatch')
    return {'row_index': frame_index, 'record_id': row['record_id'],
            'text_sha256': text_sha, **detail}


def runtime_checks(adapter):
    torch = adapter.torch
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in adapter.model.parameters()),
            'CPU_FP32_parameters_required')
    require(not adapter.model.training and not torch.backends.mkldnn.enabled
            and torch.get_num_threads() == torch.get_num_interop_threads() == 1
            and torch.are_deterministic_algorithms_enabled() and torch.initial_seed() == 0,
            'frozen_CPU_runtime_contract_mismatch')
    require(getattr(adapter.model.config, '_attn_implementation', None) == 'eager'
            and adapter.max_length == 512 and adapter.input_pad_length == 261,
            'attention_or_padding_contract_mismatch')
    return {'device': 'cpu', 'dtype': 'float32', 'threads': 1, 'interop_threads': 1,
            'deterministic_algorithms': True, 'seed': 0, 'mkldnn': False,
            'attention_implementation': 'eager', 'max_length': 512,
            'input_pad_length': 261, 'within_document_chunk_batch_size': 8}


def shard_rows(ctx, plan, index, rows=None, expected=None):
    require(type(index) is int and 0 <= index < len(plan['shards']), 'invalid_shard_index')
    shard = plan['shards'][index]
    start, stop = shard['start'], shard['stop']
    require(type(start) is int and type(stop) is int and 0 <= start < stop <= len(plan['frame_row_indices']),
            'invalid_shard_range')
    require((rows is None) == (expected is None), 'both_bound_rows_and_census_required')
    if rows is None:
        rows, expected = common.rows_for_shard(ctx, plan, index)
    require(len(rows) == len(expected) == stop - start, 'wrong_shard_row_count')
    logs = []
    for offset, (row, census) in enumerate(zip(rows, expected)):
        frame_index = start + offset
        require(census['row_index'] == plan['frame_row_indices'][frame_index], 'wrong_original_source_row_index')
        logs.append(expected_log(row, census, frame_index))
    return start, stop, rows, expected, logs


def _exact_inventory(path, expected_names):
    names = {p.name for p in path.iterdir()}
    require(names == set(expected_names), 'unexpected_or_partial_shard_inventory')
    require(all((path / name).is_file() and not (path / name).is_symlink() for name in expected_names),
            'shard_files_must_be_regular_local_files')


def _read_logs(path):
    values = []
    with path.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n') and bool(line.strip()), 'incomplete_or_blank_chunk_log_line')
            values.append(strict_json(line.decode('utf-8')))
    return values


def validate_encoded(ctx, plan, index, path, rows=None, expected=None):
    """Accept only an entire immutable encode shard; never accept a partial dir."""
    path = Path(path).resolve()
    require(path.is_relative_to(Path(ctx['workspace']).resolve()) and path.is_dir(), 'encoded_shard_outside_workspace')
    _exact_inventory(path, FILES | {'completion.json'})
    receipt_binding = common.binding(path / 'completion.json')
    receipt = strict_json((path / 'completion.json').read_bytes())
    start, stop, rows, expected, logs = shard_rows(ctx, plan, index, rows, expected)
    require(all(type(receipt.get(key)) is int for key in ('shard_index', 'start', 'stop', 'rows')),
            'encoded_integer_fields_required')
    require(receipt.get('schema') == SCHEMA and receipt.get('kind') == 'encode'
            and receipt.get('passed') is True and receipt.get('plan_sha256') == plan['sha256']
            and receipt.get('fingerprint') == ctx['fingerprint'] and receipt.get('shard_index') == index
            and receipt.get('start') == start and receipt.get('stop') == stop and receipt.get('rows') == stop - start,
            'encoded_completion_identity_mismatch')
    require(all(receipt.get(name) is False for name in FALSE_FLAGS), 'encoded_shard_claim_boundary_changed')
    if plan.get('kind') == 'full':
        require(receipt.get('qualification_result') == plan['qualification_result'],
                'encoded_shard_qualification_binding_mismatch')
    counters = {'content_tokens': sum(row['content_tokens'] for row in logs),
                'chunks': sum(row['chunks'] for row in logs),
                'model_forward_batches': sum(row['batch_count'] for row in logs)}
    require(all(type(receipt.get(name)) is int and receipt[name] == value for name, value in counters.items()),
            'encoded_completion_work_counts_mismatch')
    require(set(receipt['files']) == FILES, 'wrong_encoded_output_bindings')
    for name, expected_binding in receipt['files'].items():
        require(common.binding(path / name) == expected_binding, 'encoded_file_hash_mismatch')
    require(_read_logs(path / 'chunk_log.jsonl') == logs, 'encoded_complete_chunk_log_mismatch')
    np = ctx['np']
    vectors = np.load(path / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    try:
        require(vectors.dtype == np.dtype('float32') and vectors.shape == (stop - start, 768)
                and vectors.flags.c_contiguous, 'encoded_vector_shape_dtype_mismatch')
        require(bool(np.isfinite(vectors).all()) and bool(np.any(vectors != 0, axis=1).all()), 'invalid_encoded_vector')
    finally:
        if hasattr(vectors, '_mmap'):
            vectors._mmap.close()
    for name, expected_binding in receipt['files'].items():
        require(common.binding(path / name) == expected_binding, 'encoded_file_changed_during_validation')
    require(common.binding(path / 'completion.json') == receipt_binding, 'encoded_receipt_changed_during_validation')
    if 'watch' in ctx:
        ctx['watch'].add(path / 'completion.json', receipt_binding)
        for name, expected_binding in receipt['files'].items():
            ctx['watch'].add(path / name, expected_binding)
    return receipt


def run(args):
    begun = time.perf_counter()
    ctx = common.context(args)
    phases = {'context_validation': time.perf_counter() - begun}
    tick = time.perf_counter()
    plan = common.load_plan(ctx, args.plan)
    require(plan['kind'] in ('qualification', 'full'), 'unsupported_plan_kind')
    qualification_binding = None
    if plan['kind'] == 'full':
        require(args.qualification_result is not None, 'full_plan_requires_passed_qualification')
        common.require_qualification(ctx, args.qualification_result)
        qualification_binding = common.binding(Path(args.qualification_result))
        require(qualification_binding == plan['qualification_result'], 'full_plan_qualification_binding_mismatch')
    require(args.mode in ('encode', 'replay'), 'invalid_worker_mode')
    require((args.mode == 'replay') == (args.encoded_dir is not None), 'replay_requires_only_encoded_dir')
    start, stop, rows, expected, expected_logs = shard_rows(ctx, plan, args.shard_index)
    phases['plan_qualification_and_complete_source_row_validation'] = time.perf_counter() - tick
    encoded = encoded_binding = prior_vectors = None
    if args.mode == 'replay':
        tick = time.perf_counter()
        encoded = validate_encoded(ctx, plan, args.shard_index, args.encoded_dir, rows, expected)
        encoded_binding = common.binding(Path(args.encoded_dir) / 'completion.json')
        phases['prior_encoded_shard_validation'] = time.perf_counter() - tick
    out = Path(args.out).resolve()
    common.new_output(ctx, out)
    receipt = {'schema': SCHEMA, 'kind': args.mode, 'passed': False, 'plan_sha256': plan['sha256'],
        'fingerprint': ctx['fingerprint'], 'shard_index': args.shard_index, 'start': start, 'stop': stop,
        'rows': stop - start, 'evidence_role': 'versioned_CPU_execution_engineering_only',
        'runtime': ctx['runtime'], 'phase_seconds': phases, **{name: False for name in FALSE_FLAGS}}
    try:
        e, np = ctx['e'], ctx['np']
        tick = time.perf_counter()
        adapter = e.LocalTransformer(ctx['model_dir'], ctx['tokenizer_dir'], e.E5, threads=1)
        receipt['numerical_contract'] = runtime_checks(adapter)
        phases['cold_model_and_tokenizer_load_including_checks'] = time.perf_counter() - tick
        if encoded is not None:
            prior_vectors = np.load(Path(args.encoded_dir) / 'vectors.npy', mmap_mode='r', allow_pickle=False)
        tick = time.perf_counter()
        vectors = np.lib.format.open_memmap(out / 'vectors.npy', mode='w+', dtype=np.float32, shape=(stop - start, 768))
        backend_seconds, backend_batches, content_tokens, chunks = 0.0, 0, 0, 0
        encode_seconds = 0.0
        try:
            with (out / 'chunk_log.jsonl').open('x', encoding='utf-8') as log:
                for offset, row in enumerate(rows):
                    calls = 0
                    def backend(parts):
                        nonlocal backend_seconds, backend_batches, calls
                        require(0 < len(parts) <= 8, 'invalid_document_chunk_batch')
                        before = time.perf_counter()
                        result = adapter(parts)
                        backend_seconds += time.perf_counter() - before
                        backend_batches += 1
                        calls += 1
                        return result
                    before = time.perf_counter()
                    vector, detail = e.encode_document(adapter.tokenizer, backend, row['text'], e.E5,
                                                        adapter.max_length, batch_size=8)
                    encode_seconds += time.perf_counter() - before
                    actual = {'row_index': start + offset, 'record_id': row['record_id'],
                        'text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(), **detail}
                    require(actual == expected_logs[offset] and calls == expected[offset]['batch_count'],
                            'fresh_encoding_token_chunk_or_batch_mismatch')
                    require(vector.dtype == np.dtype('float32') and vector.shape == (768,)
                            and bool(np.isfinite(vector).all()) and bool(np.any(vector)), 'invalid_fresh_vector')
                    if prior_vectors is not None:
                        require(vector.tobytes() == prior_vectors[offset].tobytes(), 'fresh_replay_vector_bytes_differ')
                    vectors[offset] = vector
                    log.write(json.dumps(actual, sort_keys=True, allow_nan=False) + '\n')
                    content_tokens += detail['content_tokens']; chunks += detail['chunks']
                log.flush(); os.fsync(log.fileno())
            vectors.flush()
        finally:
            vectors.flush()
            vectors._mmap.close()
        with (out / 'vectors.npy').open('rb') as stream:
            os.fsync(stream.fileno())
        phases['encoding_validation_and_durable_vector_log_output'] = time.perf_counter() - tick
        receipt['encode_document_seconds'] = encode_seconds
        receipt['backend_padding_forward_masked_mean_seconds'] = backend_seconds
        receipt['model_forward_batches'] = backend_batches
        receipt['content_tokens'] = content_tokens
        receipt['chunks'] = chunks
        require(backend_batches == sum(x['batch_count'] for x in expected), 'total_forward_batch_count_mismatch')
        require(_read_logs(out / 'chunk_log.jsonl') == expected_logs, 'durable_chunk_log_differs')
        tick = time.perf_counter()
        if encoded is not None:
            require(common.binding(Path(args.encoded_dir) / 'completion.json') == encoded_binding,
                    'prior_encoded_receipt_changed_during_replay')
            for name, bound in encoded['files'].items():
                require(common.binding(Path(args.encoded_dir) / name) == bound, 'prior_encoded_files_changed_during_replay')
            receipt['encoded_completion'] = encoded_binding
            receipt['exact_vector_bytes_equal'] = True
            receipt['complete_chunk_log_equal'] = True
        common.verify_final(ctx)
        if qualification_binding is not None:
            require(common.binding(Path(args.qualification_result)) == qualification_binding,
                    'qualification_changed_during_full_encoding')
            receipt['qualification_result'] = qualification_binding
        receipt['files'] = {name: common.binding(out / name) for name in sorted(FILES)}
        phases['final_input_verification_and_output_hashes'] = time.perf_counter() - tick
        receipt['model_parameter_bytes'] = adapter.model_bytes
        receipt['model_buffer_bytes'] = adapter.model_buffer_bytes
        receipt['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt['worker_seconds_before_completion_publication'] = time.perf_counter() - begun
        receipt['timing_boundary'] = 'Includes context, full source scans, cold load, encoding, durable vectors/logs and final hashes; excludes atomic completion publication and process exit. Backend time is nested inside document and worker time.'
        receipt['created_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['passed'] = True
        common.atomic_completion(out / 'completion.json', receipt)
        return receipt
    except Exception as error:
        failure = {**receipt, 'passed': False, 'error_type': type(error).__name__, 'error': str(error),
                   'worker_seconds_at_failure': time.perf_counter() - begun,
                   'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        common.write_json(out / 'failure.json', failure)
        raise
    finally:
        if prior_vectors is not None:
            prior_vectors._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_arguments(parser)
    parser.add_argument('--mode', choices=('encode', 'replay'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--encoded-dir', type=Path)
    parser.add_argument('--qualification-result', type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(json.dumps({'schema': 'third-e5-cpu-worker-failure-v1', 'passed': False,
            'kind': args.mode, 'shard_index': args.shard_index,
            'error_type': type(error).__name__, 'error': str(error),
            'source_authenticity_verified': False, 'primary_input_acceptance': False,
            'original_route_acceptance': False, 'confirmatory_study_ready': False}), file=sys.stderr)
        return 2
    print(json.dumps({'passed': result['passed'], 'kind': result['kind'],
                      'shard_index': result['shard_index'], 'rows': result['rows']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
