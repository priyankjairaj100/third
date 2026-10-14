"""Post-v3 score-support diagnostic, with no sampling or model calls."""
import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
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

REPO = Path('/nfs_home/users/poonam/ccu-acl2027/repo')
W = REPO.parent/'local_workspace'
ROOT = W/'qualification/llm-tail-census-v1-20261014-a01'
SAMPLER = W/'tools/llm-selection-sampler-v1-20261014'
SAMPLER_SHA = 'c1da01e546c4ee0b5e84503b8513ec66f6ee5ae1ded16f46b2ce6b7801e18ce2'
CONTROL = W/'tools/llm-sampler-control-v1-20261014'
CONTROL_SHA = 'a1779a1ea98b0968ab46680be6a0249983381bcd25a02a2c405877a32b237cf9'
PREVIOUS = W/'qualification/llm-selection-sampler-v1-20261014-selection-a01/run/output/result.json'
PREVIOUS_SHA = '1b17c194be4825da9fc9b7af2d3e63ad5d3ed0c377e3fdfb33374bae8094225d'
FOLDER = Path(__file__).resolve().parent

def require(ok, why):
    if not ok:
        raise ValueError(why)

def verify(folder, wanted):
    require(not folder.is_symlink(), 'package_symlink')
    raw = (folder/'package_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == wanted, 'package_seal')
    for name, expected in json.loads(raw)['files'].items():
        require(Path(name).name == name and name != 'package_manifest.json', 'package_member')
        p = folder/name
        require(p.is_file() and not p.is_symlink(), 'package_regular_file')
        data = p.read_bytes()
        require({'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()} == expected,
                'package_payload')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def finish_histogram(histogram, cuts):
    # Bucket k contains scores exceeding exactly the first k cuts.
    require(len(histogram) == len(cuts)+1, 'histogram_shape')
    running = 0
    tails = [0]*len(cuts)
    for k in range(len(cuts), 0, -1):
        running += histogram[k]
        tails[k-1] = running
    return tails

def collect(vectors, cal, cuts, progress=False):
    require(cuts == sorted(set(cuts)) and all(math.isfinite(x) for x in cuts), 'cuts_order')
    bins = [0]*20
    histogram = [0]*(len(cuts)+1)
    count = 0
    stream = hashlib.sha256()
    audit = {}
    minimum, maximum = math.inf, -math.inf
    started = time.perf_counter()
    for i, j, score in cal.iter_pair_scores(vectors, 256, audit):
        require(0 <= i < j < len(vectors) and math.isfinite(score), 'invalid_pair')
        histogram[bisect_left(cuts, score)] += 1
        bins[cal._bin(score)] += 1
        count += 1
        minimum, maximum = min(minimum, score), max(maximum, score)
        stream.update(f'{i}\t{j}\t{score.hex()}\n'.encode('ascii'))
        if progress and count % 10000000 == 0:
            print(json.dumps({'pairs':count, 'elapsed_seconds':time.perf_counter()-started}), flush=True)
    require(count == len(vectors)*(len(vectors)-1)//2, 'incomplete_stream')
    return {'pair_count':count, 'bin_populations':bins, 'cut_interval_populations':histogram,
            'counts_strictly_above':finish_histogram(histogram, cuts),
            'score_stream_sha256':stream.hexdigest(), 'minimum_score':minimum,
            'maximum_score':maximum, 'numerical_audit':audit,
            'elapsed_seconds':time.perf_counter()-started}

def bounded_reference(vectors, panels, cuts):
    require(len(vectors) <= 512, 'reference_bound')
    normalized = panels.graph_normalize(vectors)
    matrix = panels.reference_cosines(normalized, normalized)
    tails = [0]*len(cuts)
    stream = hashlib.sha256()
    count = 0
    for left in range((len(vectors)+255)//256):
        for right in range(left, (len(vectors)+255)//256):
            for i in range(left*256, min((left+1)*256, len(vectors))):
                for j in range(max(i+1, right*256), min((right+1)*256, len(vectors))):
                    score = float(matrix[i,j])
                    for k, threshold in enumerate(cuts):
                        tails[k] += int(score > threshold)
                    stream.update((str(i)+'\t'+str(j)+'\t'+float.hex(score)+'\n').encode('ascii'))
                    count += 1
    return {'pair_count':count, 'counts_strictly_above':tails, 'score_stream_sha256':stream.hexdigest()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-sha256', required=True)
    args = parser.parse_args()
    verify(FOLDER, args.package_sha256)
    verify(CONTROL, CONTROL_SHA)
    verify(SAMPLER, SAMPLER_SHA)
    require(socket.gethostname() == 'csis.cn1' and os.environ.get('SLURM_JOB_ID'), 'worker_only')
    c = module('tail_prior_control', CONTROL/'control.py')
    s = module('tail_prior_sampler', SAMPLER/'sampler.py')
    context = SimpleNamespace(mode='diagnostic', repo=REPO, workspace=W, frame_dir=c.FRAME,
        frame_result_sha256=c.FRAME_SHA, protocol_package=c.PROTOCOL, frame_package=c.FRAME_PACKAGE,
        protocol_evidence=c.EVIDENCE, protocol_evidence_sha256=c.EVIDENCE_SHA,
        package_sha256=SAMPLER_SHA)
    out = s.safe(ROOT/'run')
    os.umask(0o077)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    completion = {'schema':'third-tail-census-process-v1', 'passed':False,
                  'slurm_job_id':os.environ['SLURM_JOB_ID']}
    try:
        watch = s.Watch()
        s.verify_package(FOLDER, args.package_sha256, watch)
        s.verify_package(CONTROL, CONTROL_SHA, watch)
        previous_binding = watch.add(PREVIOUS)
        require(previous_binding['sha256'] == PREVIOUS_SHA, 'previous_result_pin')
        previous = s.read_json(PREVIOUS)
        s.verify_seal(previous, True)
        require(previous['passed'] is True and previous['mode'] == 'selection', 'previous_census')
        pins, own, plan, protocol, cal, panels, runtime, frame, rows, vectors = s.load_context(context, watch)
        cuts = plan['calibration']['threshold_grid']
        require(len(cuts) == 51 and cuts[-1] == 0.995, 'registered_grid')
        bounded = collect(vectors[:512], cal, cuts)
        reference = bounded_reference(vectors[:512], panels, cuts)
        require(all(bounded[k] == v for k,v in reference.items()), 'bounded_reference_mismatch')
        s.publish(out/'bounded_parity.json', {'passed':True, 'rows':512, 'observed':bounded,
                                             'independent_reference':reference})
        observed = collect(vectors, cal, cuts, progress=True)
        require(observed['pair_count'] == previous['unordered_pair_count'] == pins['pairs'], 'full_count')
        require(observed['score_stream_sha256'] == previous['score_stream_sha256'], 'full_stream_replay')
        require(observed['bin_populations'] == previous['bin_populations'], 'original_bins_replay')
        require(sum(observed['cut_interval_populations']) == observed['pair_count'], 'histogram_sum')
        s.closed_inventories(context, pins)
        watch.recheck()
        s.publish(out/'input_bindings.json', {'files':watch.files})
        result = s.seal({'schema':'third-llm-tail-census-v1', 'passed':True,
            'created_utc':datetime.now(timezone.utc).isoformat(), 'runtime':runtime,
            'package_sha256':args.package_sha256, 'previous_census':previous_binding,
            'frame_result':pins['frame_result'], 'threshold_grid':cuts, 'strict_comparison':'>',
            'observed':observed, 'all_220405510_pair_scores_replayed_exactly':True,
            'bounded_first512_independent_reference_passed':True,
            'model_inference_calls':0, 'human_annotation_calls':0, 'sampling_performed':False,
            'threshold_selected':False, 'semantic_validity_established':False,
            'confirmatory_study_ready':False, 'primary_input_acceptance':False,
            'purpose':'Post-v3 development diagnostic of threshold-tail support; cannot revise v3 outcomes.',
            'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'elapsed_seconds':time.perf_counter()-started})
        s.publish(out/'result.json', result)
        completion['passed'] = True
        print(json.dumps({'passed':True, 'result':s.binding(out/'result.json')}), flush=True)
    except BaseException as error:
        completion.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        completion['elapsed_seconds'] = time.perf_counter()-started
        s.publish(out/'process_completion.json', completion)

if __name__ == '__main__':
    main()
