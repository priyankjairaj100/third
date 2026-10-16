"""Bind original AskUbuntu calibration tags to unchanged frozen task selectors."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/nfs_home/users/poonam/ccu-acl2027/repo')
W = REPO.parent / 'local_workspace'
FRAME = W / 'qualification/e5-calibration-frame-v1-20261014-a01/run/frame'
SOURCE = W / 'qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu/records.jsonl'
ROOT = W / 'qualification/askubuntu-task-calibration-v2-20261016-a01'
CAP = 64 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, 'regular_absolute_input')
    h = hashlib.sha256(); size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block); size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def publish(path, value):
    raw = canonical(value) + b'\n'
    require(len(raw) <= CAP, 'file_budget')
    require(sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file()) + len(raw) <= CAP, 'aggregate_budget')
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def verify_package(pin):
    require(binding(HERE / 'package_manifest.json')['sha256'] == pin, 'package_pin')
    manifest = json.loads((HERE / 'package_manifest.json').read_bytes())
    for name, wanted in manifest['files'].items():
        require(Path(name).name == name and name != 'package_manifest.json', 'package_member_name')
        require(binding(HERE / name) == wanted, 'package_member:' + name)
    return manifest


def validate_mapping(mapping, rows):
    require(len(mapping) == rows and rows > 0, 'complete_calibration_mapping')
    ids, indices = set(), []
    for i, item in enumerate(mapping):
        index = item.get('original_full_row_index')
        require(item.get('calibration_row_index') == i and type(index) is int and index >= 0, 'mapping_index')
        require(item.get('partition') == 'calibration', 'calibration_only_mapping')
        rid = item.get('record_id')
        require(isinstance(rid, str) and rid and rid not in ids, 'unique_mapping_ids')
        ids.add(rid); indices.append(index)
    require(indices == sorted(set(indices)), 'ordered_unique_mapping_indices')


def collect_calibration(source, mapping, total_rows, panels):
    """Decode selected calibration lines only; other rows contribute bytes/count only."""
    validate_mapping(mapping, len(mapping))
    lookup = {m['original_full_row_index']: m for m in mapping}
    kept = []; count = 0
    with source.open('rb') as stream:
        for index, line in enumerate(stream):
            count += 1
            item = lookup.get(index)
            if item is None:
                continue
            row = json.loads(line)
            require(row.get('partition') == 'calibration', 'selected_source_partition')
            require(row.get('record_id') == item['record_id'], 'selected_source_identity')
            text = row.get('text')
            require(isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == item['text_sha256'], 'selected_source_text')
            unit, kind = panels.source_unit(row, 'askubuntu', allow_preview_sources=False)
            require(unit == item['source_unit_id'] == row.get('source_unit_id') and kind == item['source_kind'] == row.get('source_kind'), 'selected_native_source')
            require(panels.digest_partition(panels.hash_integer('ccu-v1-source-split', unit)) == 'calibration', 'source_split')
            fields = row['original_fields']
            # Preserve only the fields consumed by the frozen selectors.
            kept.append({k: row[k] for k in ('record_id', 'source_unit_id', 'source_kind', 'partition')})
            kept[-1]['original_fields'] = {k: fields.get(k) for k in ('site', 'OwnerUserId', 'Tags')}
    require(count == total_rows and len(kept) == len(mapping), 'complete_selected_source_rows')
    return kept


def main():
    p = argparse.ArgumentParser(); p.add_argument('--package-sha256', required=True); a = p.parse_args()
    require(os.environ.get('SLURM_JOB_ID', '').isdigit() and socket.gethostname() in ('csis.cn1', 'cn1'), 'compute_node_only')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'single_thread_runtime')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_only')
    manifest = verify_package(a.package_sha256)
    expected = json.loads((HERE / 'expectations.json').read_bytes())
    for name, wanted in expected['repository_files'].items():
        require(binding(REPO / name) == wanted, 'frozen_source:' + name)
    sys.path.insert(0, str(REPO))
    import numpy as np
    from empirical_execution.phase3 import panels
    from empirical_execution.phase4 import model_selection as ms
    require(platform.python_version() == expected['python'] and np.__version__ == expected['numpy'], 'fixed_task_runtime')
    require(ROOT.is_dir() and ROOT.resolve() == ROOT, 'launched_root')
    out = ROOT / 'run'; out.mkdir(mode=0o700, exist_ok=False)
    start = time.perf_counter()
    watched = {FRAME/'result.json': expected['frame_result'], SOURCE: expected['source']}
    require(binding(FRAME/'result.json') == expected['frame_result'], 'frame_receipt')
    frame = json.loads((FRAME/'result.json').read_bytes())
    require(frame['passed'] is True and frame['rows'] == 20996 and frame['shape'] == [20996,768], 'admitted_frame_shape')
    for name, wanted in frame['files'].items():
        require(Path(name).name == name, 'frame_member_name'); watched[FRAME/name] = wanted
    for path, wanted in watched.items():
        require(binding(path) == wanted, 'input_binding:' + str(path))
    mapping = [json.loads(line) for line in (FRAME/'records.private.jsonl').read_bytes().splitlines()]
    validate_mapping(mapping, 20996)
    rows = collect_calibration(SOURCE, mapping, 140250, panels)
    z = np.load(FRAME/'vectors.npy', mmap_mode='r', allow_pickle=False)
    require(z.dtype == np.dtype('float32') and z.shape == (len(rows),768), 'aligned_FP32_features')
    provenance = {'dataset_id':'askubuntu','evidence_role':'engineering_nonconfirmatory',
        'population_scope':'All 20996 fixed calibration records; prospective native task configuration for the separate LLM-only development branch',
        'encoder_id':'intfloat/multilingual-e5-base','encoder_revision':expected['encoder_revision'],
        'feature_contract':'native_normalized_fp32','frame_result_sha256':expected['frame_result']['sha256'],
        'source_file_sha256':expected['source']['sha256'],'annotation_mode':'fully_llm','human_annotation_required':False,
        'source_authenticity_claimed':False,'task_calibration_package_sha256':a.package_sha256}
    vocab = ms.select_tag_vocabulary(rows, provenance)
    print(json.dumps({'stage':'vocabulary_complete','calibration_rows':len(rows),'outputs':len(vocab['vocabulary'])}), flush=True)
    lock = ms.select_ridge_regularization(z, rows, provenance, vocabulary_lock=vocab)
    thresholds = ms.select_tag_decision_thresholds(lock)
    # Cross-check the frozen decisions from the full retained ledgers.
    chosen = min(lock['selection_ledger'], key=lambda item:(item['mean_squared_loss'],-item['lambda']))
    require(chosen['lambda'] == lock['lambda'] and lock['vocabulary_lock_sha256'] == vocab['sha256'], 'task_lock_linkage')
    for value in (vocab, lock, thresholds): ms.verify(value)
    require(thresholds['lambda_lock_sha256'] == lock['sha256'], 'threshold_lock_linkage')
    for path, wanted in watched.items(): require(binding(path) == wanted, 'input_changed_during_task_calibration')
    for name, wanted in expected['repository_files'].items(): require(binding(REPO/name) == wanted, 'frozen_source_changed')
    verify_package(a.package_sha256)
    for name, value in [('vocabulary.json',vocab),('ridge.json',lock),('thresholds.json',thresholds)]: publish(out/name,value)
    result = {'schema':'third-native-task-calibration-v1','passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'package_sha256':a.package_sha256,
        'calibration_rows':len(rows),'source_units':vocab['source_units'],'genuine_source_groups':vocab['genuine_source_groups'],
        'unknown_singleton_groups':vocab['unknown_singleton_groups'],'all_zero_target_questions':vocab['all_zero_target_questions'],
        'vocabulary':vocab['vocabulary'],'lambda':lock['lambda'],'lambda_hex':lock['lambda_hex'],
        'six_lambda_five_fold_cv':True,'no_intercept':True,'regularization_scales_with_current_count':True,
        'source_result':expected['frame_result'],'source_file':expected['source'],
        'outputs':{n:binding(out/n) for n in ('vocabulary.json','ridge.json','thresholds.json')},
        'vocabulary_seal':vocab['sha256'],'ridge_seal':lock['sha256'],'thresholds_seal':thresholds['sha256'],
        'source_field_scope':'Calibration-only native tags and contributor metadata; noncalibration source lines are not JSON-decoded',
        'test_labels_decoded_or_used':False,'deletion_outcomes_used':False,'LLM_responses_used_for_task_settings':False,
        'human_annotation_calls':0,'model_inference_calls':0,'source_authenticity_verified':False,'primary_input_acceptance':False,
        'full_population_cache_acceptance':False,'confirmatory_study_ready':False,
        'scope':'Native task configuration only; no held-out utility result, semantic threshold approval or effect study',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},
        'elapsed_seconds_before_final_publication':time.perf_counter()-start,'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cost_limit':'This timer excludes scheduler queue, process startup, final result write and exit; parent receipt/accounting required'}
    publish(out/'result.json', result)
    print(json.dumps({'passed':True,'lambda':lock['lambda'],'calibration_rows':len(rows),'human_annotation_calls':0}),flush=True)


if __name__ == '__main__': main()
