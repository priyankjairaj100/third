#!/usr/bin/env python3
"""Admit exact sharded cache bytes into a distinct calibration-only frame."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time
from types import SimpleNamespace

FOLDER = Path(__file__).resolve().parent
SCHEMA = 'third-e5-calibration-frame-v1'
FALSE_FLAGS = ('source_authenticity_verified', 'primary_input_acceptance',
    'original_route_acceptance', 'original_monolithic_full_writer_executed',
    'original_monolithic_full_replay_executed', 'full_population_cache_acceptance',
    'semantic_validity_established', 'confirmatory_study_ready',
    'sampling_performed', 'threshold_selected', 'human_validation_claimed')


def require(value, why):
    if not value:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode()


def strict(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path):
    return strict(Path(path).read_bytes())


def binding(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular_input_required')
    before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before,k) == getattr(after,k) for k in keys), 'input_changed_during_hash')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def no_symlinks(path):
    path = Path(path).absolute()
    require(all(not p.is_symlink() for p in (path, *path.parents)), 'symlink_path_prohibited')
    return path


def verify_package(folder, wanted_sha):
    folder = no_symlinks(folder)
    require(binding(folder/'package_manifest.json')['sha256'] == wanted_sha,
            'package_manifest_mismatch')
    files = read_json(folder/'package_manifest.json')['files']
    require(isinstance(files, dict) and files, 'empty_package')
    for name, expected in files.items():
        require(Path(name).name == name and name not in ('.','..','package_manifest.json'),
                'unsafe_package_member')
        require(binding(folder/name) == expected, 'package_member_mismatch')
    return binding(folder/'package_manifest.json')


def new_output(path, workspace):
    path, workspace = no_symlinks(path), no_symlinks(workspace)
    require(path.is_relative_to(workspace/'qualification') and not path.exists(),
            'new_private_output_required')
    os.umask(0o077); path.mkdir(parents=True, exist_ok=False)
    return path


def publish(path, value, cap):
    raw = canonical(value) + b'\n'
    require(len(raw) <= cap, 'metadata_output_cap_exceeded')
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_manifest(manifest, plan, runtime, pins):
    require(manifest.get('schema') == 'third-e5-sharded-cpu-cache-v1'
            and manifest.get('passed') is True and manifest.get('plan_sha256') == plan['sha256']
            and manifest.get('fingerprint') == pins['fingerprint']
            and manifest.get('runtime') == runtime and manifest.get('rows') == pins['full_rows']
            and manifest.get('shape') == [pins['full_rows'],pins['dimension']]
            and manifest.get('dtype') == 'float32'
            and manifest.get('frame_ordered_ids_sha256') == pins['source_ordered_ids_sha256']
            and manifest.get('full_source_frame_sha256') == pins['full_source_frame_sha256']
            and manifest.get('files') == pins['assembly_files']
            and manifest.get('all_rows_fresh_process_replayed_under_this_version') is True,
            'full_assembly_manifest_mismatch')
    require(all(manifest.get(k) is False for k in ('source_authenticity_verified',
        'primary_input_acceptance','original_monolithic_full_writer_executed',
        'original_monolithic_full_replay_executed','full_population_cache_acceptance',
        'confirmatory_study_ready')), 'assembly_claim_boundary_changed')
    require(plan['kind'] == 'full' and len(plan['shards']) == 32
            and plan['frame_row_indices'] == list(range(pins['full_rows']))
            and plan['documents'] == pins['full_rows'], 'complete_full_plan_required')


def calibration_mapping(rows, panels, pins):
    """No filtering beyond the original partition; same-source pairs stay eligible."""
    require(len(rows) == pins['full_rows'], 'incomplete_source_frame')
    ordered = json.dumps([r.get('record_id') for r in rows],sort_keys=True,
                         ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    require(hashlib.sha256(ordered).hexdigest() == pins['source_ordered_ids_sha256'],
            'full_source_order_changed')
    seen, source_parts, parts, kinds, records = set(), {}, Counter(), Counter(), []
    for full_index, row in enumerate(rows):
        rid, text = row.get('record_id'), row.get('text')
        require(isinstance(rid,str) and rid and rid not in seen
                and isinstance(text,str) and text and text == panels.normalized_text(text),
                'invalid_duplicate_or_unnormalized_source_row')
        seen.add(rid)
        source, kind = panels.source_unit(row, pins['dataset_id'], allow_preview_sources=False)
        partition = panels.digest_partition(panels.hash_integer(pins['source_split_salt'], source))
        require(row.get('source_unit_id') == source and row.get('source_kind') == kind
                and row.get('partition') == partition, 'native_source_or_partition_mismatch')
        require(source not in source_parts or source_parts[source] == partition,
                'source_crosses_partitions')
        source_parts[source] = partition; parts[partition] += 1; kinds[kind] += 1
        if partition == 'calibration':
            records.append({'calibration_row_index':len(records),'original_full_row_index':full_index,
                'record_id':rid,'source_unit_id':source,'source_kind':kind,'partition':partition,
                'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
    require(dict(parts) == pins['partition_counts'] and len(records) == pins['calibration_rows']
            and len(records)*(len(records)-1)//2 == pins['unordered_pair_count'],
            'complete_calibration_denominator_mismatch')
    ids = [r['original_full_row_index'] for r in records]
    require(ids == sorted(set(ids)), 'calibration_indices_not_source_order')
    return records, {'partition_counts':dict(parts),'source_kind_counts':dict(kinds),
        'full_source_units':len(source_parts),'source_disjoint_across_partitions':True,
        'calibration_source_units':len({r['source_unit_id'] for r in records}),
        'calibration_unknown_singletons':sum(r['source_kind']=='unknown_singleton' for r in records),
        'within_calibration_same_source_pairs_excluded':False,
        'source_grouping_limit':'Native account metadata is a contributor proxy; unknown owners remain record singletons. No new authenticity or actual-authorship claim.'}


def verify_gather(full, gathered, indices, np):
    require(full.dtype == gathered.dtype == np.dtype('float32')
            and full.ndim == gathered.ndim == 2 and full.shape[1] == gathered.shape[1] == 768
            and gathered.shape[0] == len(indices) and list(indices) == sorted(set(indices))
            and all(type(i) is int and 0 <= i < len(full) for i in indices), 'invalid_gather_shape_indices')
    for start in range(0,len(indices),256):
        expected = full[indices[start:start+256]]
        actual = gathered[start:start+256]
        require(bool(np.isfinite(actual).all()) and bool(np.any(actual != 0,axis=1).all())
                and actual.tobytes() == expected.tobytes(), 'gather_vector_bytes_differ')


def verify_processes(ctx, dispatch, args, stage, mode, index, bindings):
    """Replay exact recorded CLI and distinct encode/replay process evidence."""
    watch = ctx['watch']; launch = read_json(stage/'launch.json'); result = read_json(stage/'dispatch_result.json')
    for name in ('launch.json','dispatch_result.json'):
        watch.add(stage/name)
    require(launch['mode'] == result['mode'] == mode and launch['shard_index'] == result['shard_index'] == index
            and launch['hostname'] == 'csis.cn1' and str(launch['slurm_job_id']).isdigit()
            and launch['bindings'] == result['bindings'] == bindings and result['passed'] is True
            and result['schema'] == 'third-e5-sharded-dispatch-result-v1', 'dispatch_identity_mismatch')
    dargs = SimpleNamespace(**vars(args)); dargs.mode = mode; dargs.full_plan = Path(args.plan)
    dargs.dispatch_root = Path(args.assembly_root); dargs.qualification_sha = args.qualification_sha256
    wanted = dispatch.commands(dargs,stage,index)
    require(len(result['children']) == len(wanted), 'dispatch_child_count_mismatch')
    pids, launcher_pids = [], set()
    for child,(name,command,artifact) in zip(result['children'],wanted):
        path = stage/(name+'.process.json'); watch.add(path)
        receipt = read_json(path)
        require(child['name'] == name and child['process_receipt'] == binding(path)
                and child['artifact'] == binding(artifact) and receipt['command'] == command
                and receipt['schema'] == 'third-e5-sharded-dispatch-child-v1'
                and type(receipt['returncode']) is int and receipt['returncode'] == 0,
                'fresh_child_command_or_artifact_mismatch')
        require(all(type(receipt[k]) is int and receipt[k] > 0 for k in ('child_pid','launcher_pid'))
                and receipt['child_pid'] != receipt['launcher_pid'], 'fresh_child_identity_missing')
        elapsed = receipt['child_wall_seconds_through_exit']
        require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed > 0
                and child['child_wall_seconds_through_exit'] == elapsed
                and receipt['timer_includes_child_artifact_publication_and_exit'] is True,
                'child_timer_mismatch')
        require(all(value.get(key) is False for value in (launch,result,receipt)
                    for key in dispatch.FALSE_FLAGS), 'dispatch_scope_changed')
        pids.append(receipt['child_pid']); launcher_pids.add(receipt['launcher_pid'])
    require(len(pids) == len(set(pids)) and len(launcher_pids) == 1, 'fresh_replay_process_missing')
    return {'stage':str(stage),'dispatch_result':binding(stage/'dispatch_result.json'),
            'children':len(wanted),'fresh_child_pids':pids}


def load_modules(repo, pins):
    core_folder = repo/pins['core_folder']; dispatch_folder = repo/pins['dispatch_folder']
    verify_package(core_folder,pins['core_package_sha256'])
    verify_package(dispatch_folder,pins['dispatch_package_sha256'])
    sys.path.insert(0,str(core_folder))
    modules = [importlib.import_module(name) for name in ('common','driver','worker')]
    for module,name in zip(modules,('common','driver','worker')):
        require(Path(module.__file__).resolve() == core_folder/(name+'.py'), 'wrong_core_module')
    spec = importlib.util.spec_from_file_location('frame_sealed_dispatch',dispatch_folder/'dispatch.py')
    dispatch = importlib.util.module_from_spec(spec); sys.modules[spec.name] = dispatch; spec.loader.exec_module(dispatch)
    return (*modules,dispatch)


def validate_cache(ctx,m,driver,worker,dispatch,args,pins,rows,census,plan):
    root = Path(args.assembly_root); cache = root/'assembly-run/cache'
    no_symlinks(cache)
    worker._exact_inventory(cache,{'manifest.json','vectors.npy','chunk_log.jsonl'})
    ctx['watch'].add(cache/'manifest.json',pins['assembly_manifest'])
    manifest = read_json(cache/'manifest.json'); m.verify_seal(manifest)
    require(manifest['sha256'] == pins['assembly_manifest_seal'], 'assembly_seal_changed')
    check_manifest(manifest,plan,ctx['runtime'],pins)
    for name,expected in pins['assembly_files'].items():
        ctx['watch'].add(cache/name,expected)
    bindings = {'core_package':binding(ctx['folder']/'package_manifest.json'),
        'dispatch_package':binding(Path(args.repo)/pins['dispatch_folder']/'package_manifest.json'),
        'qualification_result':binding(args.qualification_result),'full_plan':binding(args.plan)}
    receipts, process_evidence = [], []
    counters = Counter()
    for index,shard in enumerate(plan['shards']):
        stage = root/'shards'/f'shard-{index:04d}'; encoded,replay = stage/'encode',stage/'replay'
        for path in (stage,encoded,replay):
            no_symlinks(path)
        start,stop = shard['start'],shard['stop']
        receipt = worker.validate_encoded(ctx,plan,index,encoded,rows[start:stop],census[start:stop])
        driver.validate_replay_receipt(ctx,plan,index,replay,encoded,receipt)
        driver.compare_shard_slice(cache,encoded,start,stop)
        receipts.append({'index':index,'encoded_completion':binding(encoded/'completion.json'),
                         'replay_completion':binding(replay/'completion.json')})
        process_evidence.append(verify_processes(ctx,dispatch,args,stage,'shard',index,bindings))
        counters.update({k:receipt[k] for k in ('rows','content_tokens','chunks','model_forward_batches')})
    require(receipts == manifest['shard_receipts'], 'assembly_shard_receipt_order_changed')
    require(dict(counters) == {'rows':pins['full_rows'],'content_tokens':pins['full_content_tokens'],
        'chunks':pins['full_chunks'],'model_forward_batches':pins['full_model_forward_batches']},
        'complete_encoded_work_counts_mismatch')
    process_evidence.append(verify_processes(ctx,dispatch,args,root/'assembly-run','assembly',None,bindings))
    return cache,manifest,process_evidence,dict(counters)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo','workspace','view','replay-result','census-dir','budget-dir','qualification-result',
                 'qualification-sha256','plan','assembly-root','repair-result','frame-package-sha256','out'):
        p.add_argument('--'+name,required=True)
    return p


def execute(args):
    started = time.perf_counter(); out = None
    report = {'schema':SCHEMA,'passed':False,'stage':'preflight','model_forwards':0,
        'new_model_judgments':0,'human_responses':0,'requires_human_annotations':False,
        **dict.fromkeys(FALSE_FLAGS,False)}
    try:
        verify_package(FOLDER,args.frame_package_sha256)
        pins = read_json(FOLDER/'expectations.json')
        for name in ('repo','workspace','view','replay_result','census_dir','budget_dir',
                     'qualification_result','plan','assembly_root','repair_result'):
            setattr(args,name,no_symlinks(getattr(args,name)).resolve(strict=True))
        require(args.assembly_root == args.workspace/pins['assembly_root_relative']
                and args.plan == args.assembly_root/'plan-run/output/plan.json'
                and binding(args.plan) == pins['plan']
                and args.qualification_sha256 == pins['qualification_result']['sha256']
                and binding(args.qualification_result) == pins['qualification_result']
                and binding(args.repair_result) == pins['repair_result'], 'fixed_source_paths_or_pins_changed')
        require(Path(sys.executable).absolute() == args.workspace/'envs/encoder-cpu/bin/python',
                'pinned_CPU_interpreter_required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_single_thread_environment_required')
        out = new_output(args.out,args.workspace)
        m,driver,worker,dispatch = load_modules(args.repo,pins)
        args.package_sha256 = pins['core_package_sha256']; ctx = m.context(args)
        m.c.verify_package(FOLDER,args.frame_package_sha256,ctx['watch'])
        m.c.verify_package(args.repo/pins['dispatch_folder'],pins['dispatch_package_sha256'],ctx['watch'])
        require(ctx['fingerprint'] == pins['fingerprint'], 'qualified_runtime_changed')
        ctx['watch'].add(args.repair_result,pins['repair_result'])
        repair = read_json(args.repair_result)
        require(repair['passed'] is True and repair['stage'] == 'complete'
                and repair['all_preexisting_versions_and_RECORDs_unchanged'] is True
                and importlib.metadata.version('scipy') == repair['scipy_version'] == '1.17.1', 'SciPy_repair_required')
        m.require_qualification(ctx,args.qualification_result)
        ctx['watch'].add(args.plan,pins['plan']); plan = m.load_plan(ctx,args.plan)
        require(plan['kind'] == 'full' and plan == driver.make_plan(ctx,'full'), 'prospective_full_plan_changed')
        for name,expected in pins['repository_files'].items():
            ctx['watch'].add(args.repo/name,expected)
        design = read_json(args.repo/'output/empirical_program/study_design.json')
        require(design['seeds']['source_split_salt'] == pins['source_split_salt']
                and ctx['audit']['study_design_sha256'] == pins['repository_files']['output/empirical_program/study_design.json']['sha256']
                and ctx['audit']['split']['source_split_salt'] == pins['source_split_salt'], 'original_partition_policy_changed')
        report['stage'] = 'full_source_and_cache_validation'
        rows,census = m.rows_for_indices(ctx,plan['frame_row_indices'])
        require(ctx['cal'].digest([r['record_id'] for r in rows]) == pins['source_ordered_ids_sha256'], 'full_row_order_changed')
        from phase3 import panels
        records,grouping = calibration_mapping(rows,panels,pins)
        cache,manifest,processes,counters = validate_cache(ctx,m,driver,worker,dispatch,args,pins,rows,census,plan)
        report['payload_validation_seconds'] = time.perf_counter()-started
        indices = [r['original_full_row_index'] for r in records]
        metadata = b''.join(canonical(r)+b'\n' for r in records)
        caps = pins['output_caps']; require(len(metadata) <= caps['row_metadata_bytes'], 'row_metadata_cap_exceeded')
        input_receipt = {'schema':'third-e5-calibration-frame-inputs-v1','frame_package_sha256':args.frame_package_sha256,
            'full_plan':pins['plan'],'source_cache_manifest':pins['assembly_manifest'],
            'source_cache_files':pins['assembly_files'],'qualification_result':pins['qualification_result'],
            'repair_result':pins['repair_result'],'source_records_file':binding(ctx['view']/'records.jsonl'),
            'source_audit_file':binding(ctx['view']/'audit.json'),'source_audit_seal':ctx['audit']['sha256'],
            'full_source_frame_sha256':ctx['aggregate']['full_prepared_rows_sha256'],
            'full_text_binding':ctx['aggregate']['text_binding'],'full_ordered_record_ids_sha256':pins['source_ordered_ids_sha256'],
            'files':{str(p):v for p,v in ctx['watch'].files.items()}}
        require(len(canonical(input_receipt))+1 <= caps['input_binding_bytes']
                and len(canonical(input_receipt))+1+len(metadata)+len(indices)*768*4+128
                    +caps['result_bytes'] <= caps['frame_directory_bytes'], 'prewrite_aggregate_output_cap_exceeded')
        publish(out/'input_bindings.json',input_receipt,caps['input_binding_bytes'])
        report['stage'] = 'gather_calibration'
        with (out/'records.private.jsonl').open('xb') as stream:
            stream.write(metadata); stream.flush(); os.fsync(stream.fileno())
        np = ctx['np']; full = np.load(cache/'vectors.npy',mmap_mode='r',allow_pickle=False)
        require(full.dtype == np.dtype('float32') and full.shape == (pins['full_rows'],768)
                and full.flags.c_contiguous, 'full_vector_shape_or_dtype')
        gathered = np.lib.format.open_memmap(out/'vectors.npy',mode='w+',dtype=np.float32,shape=(len(indices),768))
        for start in range(0,len(indices),256):
            gathered[start:start+256] = full[indices[start:start+256]]
        gathered.flush(); del gathered
        with (out/'vectors.npy').open('rb') as stream:
            os.fsync(stream.fileno())
        gathered = np.load(out/'vectors.npy',mmap_mode='r',allow_pickle=False)
        verify_gather(full,gathered,indices,np)
        del full,gathered
        require(binding(out/'vectors.npy')['bytes'] == len(indices)*768*4+128, 'unexpected_NPY_storage_bytes')
        outputs = {name:binding(out/name) for name in ('vectors.npy','records.private.jsonl','input_bindings.json')}
        require(sum(v['bytes'] for v in outputs.values())+caps['result_bytes'] <= caps['frame_directory_bytes'], 'aggregate_output_cap_exceeded')
        report['stage'] = 'final_immutability'; final_start = time.perf_counter()
        m.verify_final(ctx)
        for name,expected in outputs.items():
            require(binding(out/name) == expected, 'output_changed_during_final_verification')
        require({p.name for p in out.iterdir()} == set(outputs), 'unexpected_frame_output_inventory')
        require((out/'records.private.jsonl').read_bytes() == metadata, 'row_mapping_changed')
        worker._exact_inventory(cache,{'manifest.json','vectors.npy','chunk_log.jsonl'})
        for index in range(32):
            for kind in ('encode','replay'):
                worker._exact_inventory(args.assembly_root/'shards'/f'shard-{index:04d}'/kind,worker.FILES|{'completion.json'})
        report.update(passed=True,stage='complete',created_utc=datetime.now(timezone.utc).isoformat(),
            frame_package_sha256=args.frame_package_sha256,files=outputs,rows=len(records),shape=[len(records),768],
            dtype='float32',unordered_pair_count=pins['unordered_pair_count'],within_calibration_same_source_exclusion=False,
            full_population_rows=pins['full_rows'],source_grouping=grouping,
            source_cache_manifest=pins['assembly_manifest'],source_cache_files=pins['assembly_files'],
            full_plan=pins['plan'],qualification_result=pins['qualification_result'],repair_result=pins['repair_result'],
            fingerprint=ctx['fingerprint'],inherited_derivation_runtime=ctx['runtime'],
            actual_cache_payload_hashes_verified=True,all_32_encoded_and_fresh_replay_payloads_verified=True,
            all_32_assembled_slices_exact=True,all_calibration_vectors_exact_FP32_gather=True,
            input_code_assets_initial_final_equal=True,source_counts=counters,validated_process_evidence=processes,
            calibration_ordered_ids_sha256=ctx['cal'].digest([r['record_id'] for r in records]),
            calibration_ordered_source_ids_sha256=ctx['cal'].digest([r['source_unit_id'] for r in records]),
            original_full_row_indices_sha256=ctx['cal'].digest(indices),
            source_full_rows_sha256=ctx['aggregate']['full_prepared_rows_sha256'],source_text_binding=ctx['aggregate']['text_binding'],
            prepared_source_records_file=input_receipt['source_records_file'],prepared_source_audit_file=input_receipt['source_audit_file'],
            final_verification_seconds=time.perf_counter()-final_start,
            runtime={'hostname':socket.gethostname(),'job_id':os.environ.get('SLURM_JOB_ID'),
                'python':sys.version,'numpy':np.__version__,'affinity':sorted(os.sched_getaffinity(0))},
            frame_scope='Every calibration row of the complete replayed prepared frame, in original order. This versioned adapter supports later LLM-only sampling; no sampling or judgments performed.',
            downstream_text_lookup='Use original_full_row_index against bound full source, require record_id and text_sha256 equality. No full text is duplicated here.')
    except Exception as error:
        report.update(passed=False,error_type=type(error).__name__,error=str(error))
    report.update(elapsed_seconds_before_publication=time.perf_counter()-started,
                  peak_rss_kib_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if out is not None:
        report['sha256'] = hashlib.sha256(canonical(report)).hexdigest()
        publish(out/'result.json',report,2*1024**2)
    print(json.dumps({k:report[k] for k in ('schema','passed','stage','rows','unordered_pair_count','error_type','error') if k in report},sort_keys=True))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    sys.exit(execute(parser().parse_args()))
