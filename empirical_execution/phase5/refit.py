"""Versioned full-refit boundary audit. Official kernels and CPU reference stay distinct.

The vendored Meta kernels are CC BY-NC 4.0; see vendor/SemDeDup/LICENSE.
This local adapter changes orchestration, not the upstream source files.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import random
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(__file__).parent / 'vendor/SemDeDup'
UPSTREAM_COMMIT = '6b4194511202c29b0e1ac8c730996777449ea2a4'
OFFICIAL = 'pinned_faiss_and_upstream_rank_suppression_cpu'
REFERENCE = 'separate_numpy_spherical_lloyd_reference'

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def array_digest(a):
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes(order='C')).hexdigest()

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def measured(fn):
    t = time.perf_counter(); result = fn()
    return result, time.perf_counter() - t

@dataclass(frozen=True)
class Config:
    clusters: int
    iterations: int
    epsilon: float
    seed: int
    backend: str
    policy: str = 'hard'
    development_config_id: str = ''
    spherical: bool = True
    threads: int = 1

    def validate(self):
        for name in ('clusters', 'iterations', 'threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('nonnegative integer seed required')
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)) or not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError('epsilon must lie strictly between zero and one')
        if self.backend not in (OFFICIAL, REFERENCE) or self.policy not in ('hard', 'easy', 'random'):
            raise ValueError('unknown backend or selection policy')
        if self.spherical is not True:
            raise ValueError('this version explicitly pins spherical clustering')
        if not isinstance(self.development_config_id, str) or not self.development_config_id:
            raise ValueError('explicit development-only configuration identity required')

def validate_inputs(features, ids):
    x = np.asarray(features); ids = tuple(ids)
    if x.ndim != 2 or x.dtype != np.float32 or x.shape[0] != len(ids) or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('aligned finite stored FP32 curator features required')
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('unique nonempty record IDs required')
    if len(x) and not np.allclose(np.linalg.norm(x.astype(np.float64), axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError('input unit normalization required; never silently renormalize')
    return x, ids

def vendor_integrity():
    manifest = json.loads((VENDOR / 'UPSTREAM_MANIFEST.json').read_text())
    if manifest['commit'] != UPSTREAM_COMMIT:
        raise ValueError('wrong upstream commit')
    for entry in manifest['files']:
        body = (VENDOR / entry['path']).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        if blob != entry['git_blob_sha1'] or hashlib.sha256(body).hexdigest() != entry['sha256']:
            raise ValueError('vendored file changed: ' + entry['path'])
    return manifest

def backend_status():
    missing = [name for name in ('torch', 'faiss', 'pandas', 'tqdm') if importlib.util.find_spec(name) is None]
    return dict(official_backend_available=not missing, missing_modules=missing,
                official_runtime_executed=False, upstream_commit=UPSTREAM_COMMIT)

def _official_kernels():
    """Compile exact AST bodies from verified upstream files, avoiding SLURM CLI."""
    vendor_integrity()
    status = backend_status()
    if not status['official_backend_available']:
        raise RuntimeError('Official SemDeDup CPU backend unavailable: ' + ', '.join(status['missing_modules']))
    import torch, pandas as pd, faiss
    import os
    from typing import List, Tuple, Union
    from tqdm import tqdm
    scope = dict(np=np, torch=torch, pd=pd, os=os, time=time, List=List, Tuple=Tuple,
                 Union=Union, tqdm=tqdm)
    rank_ast = ast.parse((VENDOR / 'clustering/sort_clusters.py').read_text())
    node = next(x for x in rank_ast.body if isinstance(x, ast.FunctionDef) and x.name == 'rank_within_cluster')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(VENDOR / 'clustering/sort_clusters.py'), 'exec'), scope)
    sem_ast = ast.parse((VENDOR / 'semdedup.py').read_text())
    cls = next(x for x in sem_ast.body if isinstance(x, ast.ClassDef) and x.name == 'SemDeDupJob')
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'semdedup')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(VENDOR / 'semdedup.py'), 'exec'), scope)
    return faiss, torch, pd, scope['rank_within_cluster'], scope['semdedup']

def _numpy_cluster(x, cfg):
    """Separate CPU reference, intentionally NOT a Faiss-equivalent implementation."""
    rng = np.random.default_rng(cfg.seed)
    centers = x[rng.choice(len(x), cfg.clusters, replace=False)].copy()
    for _ in range(cfg.iterations):
        assignments = np.argmax(x @ centers.T, axis=1)
        next_centers = centers.copy()
        for cluster in range(cfg.clusters):
            members = x[assignments == cluster]
            if len(members):
                c = np.sum(members.astype(np.float64), axis=0)
                norm = np.linalg.norm(c)
                if norm > 0:
                    next_centers[cluster] = (c / norm).astype(np.float32)
        centers = next_centers
    similarities = x @ centers.T
    assignments = np.argmax(similarities, axis=1)
    return centers, assignments, similarities[np.arange(len(x)), assignments]

def _suppress(x, ids, orders, cfg, kernels=None):
    selected = []; pair_elements = 0
    if cfg.backend == OFFICIAL:
        kernels = kernels or _official_kernels()
        _, torch, _, _, semdedup = kernels
    for order in orders:
        if not order:
            continue
        idx = np.asarray(order, dtype=np.int64)
        pair_elements += len(idx)**2
        if cfg.backend == OFFICIAL:
            cluster = np.asarray([[ids[i], str(i)] for i in idx])
            owner = SimpleNamespace(_contains_duplicates=lambda a: len(np.unique(a)) != len(a))
            with contextlib.redirect_stdout(io.StringIO()):
                maximum = semdedup(owner, cluster, torch.from_numpy(x[idx].copy()), 'cpu').numpy()
        else:
            # The exact upstream zero-triangular convention; epsilon keeps tau > 0.
            maximum = np.max(np.triu(x[idx] @ x[idx].T, k=1), axis=0)
        selected.extend(int(i) for i, score in zip(idx, maximum) if not score > 1-cfg.epsilon)
    return sorted(selected), pair_elements

def fit_curator(features, record_ids, config, work_dir):
    config.validate(); x, ids = validate_inputs(features, record_ids)
    work = Path(work_dir)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('Every fit requires a fresh directory; no stale centroid/cluster reuse')
    work.mkdir(parents=True, exist_ok=True)
    if 0 < len(x) < config.clusters:
        raise ValueError('Retained count below fixed cluster count; explicit failure, no adaptive cluster reduction')
    t = time.perf_counter()
    kernels = None
    if config.backend == OFFICIAL:
        kernels, backend_setup_seconds = measured(_official_kernels)
        faiss, torch, pd, rank, _ = kernels
        faiss.omp_set_num_threads(config.threads); torch.set_num_threads(config.threads)
        if len(x):
            def clustering():
                km = faiss.Kmeans(x.shape[1], config.clusters, niter=config.iterations,
                                  verbose=False, seed=config.seed, spherical=True, gpu=False)
                km.train(x); distances, assignments = km.index.search(x, 1)
                return km.centroids, assignments[:, 0], distances[:, 0]
            (centers, assignments, similarity), cluster_seconds = measured(clustering)
            frame = pd.DataFrame(dict(paths_list=list(ids), nearest_cent=assignments, dist_to_cent=similarity))
            with contextlib.redirect_stdout(io.StringIO()):
                ranked, ordering_seconds = measured(lambda: rank(x, frame, centers, 'cosine', True, True,
                    list(range(config.clusters)), str(work / 'upstream_sorted_clusters')))
            orders = [[int(item[1]) for item in cluster] for cluster in ranked]
        else:
            centers = np.empty((0, x.shape[1]), np.float32); assignments = np.empty(0, int)
            similarity = np.empty(0, np.float32); orders = []; cluster_seconds = ordering_seconds = 0.
    else:
        backend_setup_seconds = 0.
        if len(x):
            (centers, assignments, similarity), cluster_seconds = measured(lambda: _numpy_cluster(x, config))
            def ordered():
                return [sorted(np.flatnonzero(assignments == c).tolist(), key=lambda i: float(1-similarity[i]), reverse=True)
                        for c in range(config.clusters)]
            orders, ordering_seconds = measured(ordered)
        else:
            centers = np.empty((0, x.shape[1]), np.float32); assignments = np.empty(0, int)
            similarity = np.empty(0, np.float32); orders = []; cluster_seconds = ordering_seconds = 0.
    policy_start = time.perf_counter()
    rng = random.Random(config.seed)
    for order in orders:
        if config.policy == 'easy': order.reverse()
        elif config.policy == 'random': rng.shuffle(order)
    ordering_seconds += time.perf_counter() - policy_start
    (selected, pair_elements), suppression_seconds = measured(lambda: _suppress(x, ids, orders, config, kernels))
    fitted = dict(record_ids=list(ids), features_sha256=array_digest(x), orders=[[ids[i] for i in order] for order in orders],
                  selected_ids=[ids[i] for i in selected], assignments=assignments.tolist(),
                  centroid_similarity=similarity.tolist(), config=asdict(config))
    fit_identity = digest(fitted)
    persistence_start = time.perf_counter()
    np.save(work / 'centroids.npy', centers)
    write(work / 'fitted.json', {**fitted, 'fit_sha256': fit_identity})
    persisted_bytes = sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
    ledger = dict(backend_setup_seconds=backend_setup_seconds, clustering_seconds=cluster_seconds,
        ordering_seconds=ordering_seconds, suppression_seconds=suppression_seconds,
        persistence_seconds=time.perf_counter()-persistence_start,
        end_to_end_seconds=time.perf_counter()-t, retained_records=len(ids),
        curator_input_bytes=x.nbytes, centroids_array_bytes=centers.nbytes,
        similarity_matrix_elements_computed=pair_elements, persisted_bytes=persisted_bytes,
        no_peak_RSS_claim=True, warm_single_process_timing=True)
    return {**fitted, 'fit_sha256': fit_identity, 'ledger': ledger}

def frozen_fitted_selection(features, record_ids, original_fit, deleted, config):
    """Restrict original cluster membership/order, rerun suppression on surviving raw points."""
    x, ids = validate_inputs(features, record_ids); lookup = {r: i for i, r in enumerate(ids)}
    if original_fit['record_ids'] != list(ids) or original_fit['config'] != asdict(config):
        raise ValueError('frozen fitted state/input/config mismatch')
    sealed_fields={key:original_fit[key] for key in ('record_ids','features_sha256','orders','selected_ids','assignments','centroid_similarity','config')}
    if original_fit.get('fit_sha256') != digest(sealed_fields):
        raise ValueError('frozen fitted state integrity mismatch')
    if original_fit['features_sha256'] != array_digest(x):
        raise ValueError('frozen curator feature bytes changed')
    dead = set(deleted)
    if not dead <= set(ids): raise ValueError('unknown deletion')
    orders = [[lookup[r] for r in cluster if r not in dead] for cluster in original_fit['orders']]
    (selected, pair_elements), seconds = measured(lambda: _suppress(x, ids, orders, config))
    return [ids[i] for i in selected], dict(suppression_seconds=seconds,
        similarity_matrix_elements_computed=pair_elements, clusters_and_orders_refit=False)

def initial_graph(features, record_ids, source_ids, fit, config):
    """Initial cluster-local blocker graph for graph-only request generation, not main G."""
    from ccu.core import BlockerGraph
    x, ids = validate_inputs(features, record_ids); lookup = {r: i for i, r in enumerate(ids)}
    if len(source_ids) != len(ids): raise ValueError('source alignment')
    blockers = [[] for _ in ids]; flat = []
    torch = _official_kernels()[1] if config.backend == OFFICIAL else None
    for order_ids in fit['orders']:
        order = [lookup[r] for r in order_ids]; flat.extend(order)
        if not order: continue
        reps = x[order]
        scores = (torch.from_numpy(reps) @ torch.from_numpy(reps).T).numpy() if torch else reps @ reps.T
        for j, later in enumerate(order):
            blockers[later] = sorted(order[k] for k in range(j) if scores[k, j] > 1-config.epsilon)
    ptr = np.asarray([0] + list(np.cumsum([len(b) for b in blockers])), dtype=np.int64)
    index = np.asarray([i for b in blockers for i in b], dtype=np.int64)
    return BlockerGraph(ids, tuple(source_ids), np.asarray(flat, dtype=np.int64), ptr, index, 1-config.epsilon)

def ridge_head(features, targets, ids, selected, lambda_reg):
    from ccu.core import ridge_moments, solve_ridge
    lookup = {r: i for i, r in enumerate(ids)}
    indices = np.asarray(sorted(lookup[r] for r in selected), dtype=np.int64)
    moments, moment_seconds = measured(lambda: ridge_moments(features[indices], targets[indices]))
    head, head_seconds = measured(lambda: solve_ridge(moments, lambda_reg))
    return head.weights, dict(moment_seconds=moment_seconds, head_seconds=head_seconds, selected_count=len(selected))

def run_boundary(features, learner, targets, ids, sources, source_kinds, config, output_dir, *,
                 lambda_reg, design, allocations, seed_variants, prediction_features=None,
                 evidence_role='engineering_nonconfirmatory'):
    """Own initial state; fresh R, conditional F, frozen-selection targets and seed audit.

    Primary execution stays blocked until official backend and study-wide gates are
    authenticated/integrated. This is full software orchestration, not that gate.
    """
    from phase4.requests import generate_manifest
    if evidence_role != 'engineering_nonconfirmatory':
        raise RuntimeError('Confirmatory full-refit execution requires external primary study integration')
    config.validate(); x, ids = validate_inputs(features, ids)
    z = np.asarray(learner); y = np.asarray(targets)
    if y.ndim == 1: y = y[:, None]
    if z.ndim != 2 or z.dtype != np.float32 or len(z) != len(ids) or z.shape[1] < 1 or y.ndim != 2 or y.dtype != np.float64 or len(y) != len(ids) or y.shape[1] < 1 or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError('aligned FP32 learner and FP64 targets required')
    if isinstance(lambda_reg, bool) or not math.isfinite(lambda_reg) or lambda_reg <= 0:
        raise ValueError('positive finite lambda required')
    prediction = z if prediction_features is None else np.asarray(prediction_features)
    if prediction.ndim != 2 or prediction.dtype != np.float32 or prediction.shape[1] != z.shape[1] or not len(prediction) or not np.isfinite(prediction).all():
        raise ValueError('nonempty aligned FP32 prediction features required')
    if len(seed_variants) != 3 or len(set(seed_variants)) != 3 or config.seed in seed_variants:
        raise ValueError('three distinct alternative seeds required')
    for seed in seed_variants: replace(config, seed=seed).validate()
    if len(sources) != len(ids) or any(not isinstance(s, str) or not s for s in sources) or set(source_kinds) != set(sources):
        raise ValueError('source metadata coverage required; unknowns remain explicitly unknown')
    if set(allocations) != {'R', 'S', 'A'}: raise ValueError('explicit R/S/A allocation required')
    if config.backend == OFFICIAL and not backend_status()['official_backend_available']:
        raise RuntimeError('Official backend dependencies missing; no execution evidence created')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError('refusing to overwrite boundary run')
    out.mkdir(parents=True, exist_ok=True)
    lock = dict(schema='ccu-full-refit-boundary-1', config=asdict(config), upstream_commit=UPSTREAM_COMMIT,
        evidence_role=evidence_role, primary_study=False, backend_requested=config.backend,
        official_backend_executed=False, lock_written_before_execution=True,
        official_whole_CLI_executed=False, reference_is_faiss_equivalent=False,
        feature_hash=array_digest(x), learner_hash=array_digest(z), target_hash=array_digest(y),
        ids_hash=digest(ids), source_hash=digest(sources), source_kinds=source_kinds,
        prediction_features_hash=array_digest(prediction), lambda_reg=lambda_reg, design=design,
        allocations=allocations, same_seed_repeats=5, seed_variants=seed_variants,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        timing_scope='warm same-process per-stage costs; no systems or speed claim')
    write(out/'lock.json', lock)
    original = fit_curator(x, ids, config, out/'original')
    original_head, original_head_ledger = ridge_head(z, y, ids, original['selected_ids'], lambda_reg)
    write(out/'original_head.json', dict(selected_ids=original['selected_ids'], weights=original_head.tolist(), ledger=original_head_ledger))
    graph, graph_seconds = measured(lambda: initial_graph(x, ids, sources, original, config))
    requests = generate_manifest(graph, dataset_id='full_refit_boundary', panel_id='branch_own_initial_fit',
        source_kinds=source_kinds, design=design, allocations={**allocations, 'U':0},
        evidence_role=evidence_role, include_excluded_blocker_stress=False)
    write(out/'requests.json', requests)
    write(out/'request_origin.json', dict(fit_sha256=original['fit_sha256'], graph_build_seconds=graph_seconds,
        curator='cluster_local_upstream_style_ordered_raw_suppression',
        generic_request_graph_binding_label='global_earlier_raw_neighbor_suppression identifies the abstract oriented-graph sampler, not the main global semantic curator G'))
    repeats = []
    for i, seed in enumerate([config.seed]*5 + list(seed_variants)):
        fit = fit_curator(x, ids, replace(config, seed=seed), out/f'seed_audit/{i}')
        head, head_ledger = ridge_head(z, y, ids, fit['selected_ids'], lambda_reg)
        repeats.append(dict(repeat=i, seed=seed, same_declared_seed=seed==config.seed,
            selected_symmetric_difference=len(set(fit['selected_ids']) ^ set(original['selected_ids'])),
            prediction_rms=float(np.sqrt(np.mean((prediction.astype(np.float64) @ (head-original_head))**2))),
            weights=head.tolist(), fit_ledger=fit['ledger'], head_ledger=head_ledger))
    write(out/'seed_audit.json', repeats)
    lookup = {r:i for i,r in enumerate(ids)}; counts = {'R':0,'S':0}; rows = []; failures = 0
    for trajectory in requests['trajectories']:
        pid=trajectory['trajectory_id']; arm=trajectory['arm']; unit=trajectory['unit']
        if arm in counts: counts[arm]+=1
        all_steps=arm in counts and counts[arm]<=2
        checkpoints = list(range(1,len(trajectory['deletion_order'])+1)) if all_steps else trajectory['checkpoints']
        for checkpoint in checkpoints:
            checkpoint_start = time.perf_counter()
            row=dict(trajectory_id=pid, arm=arm, checkpoint=checkpoint, every_step_oracle_path=all_steps,
                     main_checkpoint=checkpoint in trajectory['checkpoints'])
            try:
                deleted_units=set(trajectory['deletion_order'][:checkpoint])
                dead = deleted_units if unit=='record' else {r for r,s in zip(ids,sources) if s in deleted_units}
                retained=np.asarray([i for i,r in enumerate(ids) if r not in dead],dtype=np.int64)
                (retained_x,retained_ids),gather_seconds=measured(lambda:(x[retained],[ids[i] for i in retained]))
                fresh = fit_curator(retained_x, retained_ids, config, out/f'paths/{pid}/{checkpoint}')
                selected_f, ledger_f = frozen_fitted_selection(x, ids, original, dead, config)
                selected_b = [r for r in original['selected_ids'] if r not in dead]
                selected = {'R_full_refit':fresh['selected_ids'], 'F_frozen_fitted':selected_f, 'B_frozen_selection':selected_b}
                heads={}; ledgers={}
                for name, chosen in selected.items(): heads[name],ledgers[name]=ridge_head(z,y,ids,chosen,lambda_reg)
                ref=heads['R_full_refit']; pred=prediction.astype(np.float64)
                row.update(status='completed', deleted_records=len(dead), deleted_sources=len(deleted_units) if unit=='source' else None,
                    selected_ids=selected, weights={k:v.tolist() for k,v in heads.items()},
                    selected_symmetric_difference_vs_refit={k:len(set(v)^set(fresh['selected_ids'])) for k,v in selected.items()},
                    prediction_rms_vs_refit={k:float(np.sqrt(np.mean((pred@(v-ref))**2))) for k,v in heads.items()},
                    admissions={k:len(set(v)-set(original['selected_ids'])) for k,v in selected.items()},
                    curator_ledgers={'R_full_refit':fresh['ledger'],'F_frozen_fitted':ledger_f}, head_ledgers=ledgers,
                    retained_gather_seconds=gather_seconds,retained_gather_array_bytes=retained_x.nbytes)
            except Exception as error:
                failures+=1; row.update(status='failed', error_type=type(error).__name__, error=str(error))
            row['checkpoint_end_to_end_before_report_seconds']=time.perf_counter()-checkpoint_start
            rows.append(row)
            with (out/'checkpoints.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
    summary=dict(scope='engineering boundary orchestration; corpus provenance is external', checkpoint_rows=len(rows), failures=failures,
        completed_rows=sum(r['status']=='completed' for r in rows), trajectory_count=len(requests['trajectories']),
        source_trajectories=sum(r['arm']=='S' for r in requests['trajectories']),
        same_seed_repeat_rows=5, seed_sensitivity_rows=3, branch_original_selected=len(original['selected_ids']),
        official_backend_executed=config.backend==OFFICIAL, primary_study=False,
        whole_output_persisted_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        warm_single_process_timings_not_speedup_evidence=True)
    write(out/'summary.json', summary)
    return summary

def main():
    """Consume caller-supplied local arrays and a development-only locked JSON recipe."""
    import argparse, sys
    sys.path.insert(0,str(ROOT))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); bundle=json.loads(args.bundle.read_text()); base=args.bundle.resolve().parent
    def local_array(key):
        entry=bundle[key]; path=(base/entry['path']).resolve()
        if not path.is_relative_to(base):raise ValueError('array path escapes local bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['file_sha256']:
            raise ValueError('array file integrity mismatch: '+key)
        return np.load(path,allow_pickle=False)
    result=run_boundary(local_array('curator'),local_array('learner'),local_array('targets'),
        bundle['record_ids'],bundle['source_ids'],bundle['source_kinds'],Config(**bundle['config']),
        args.output_dir,lambda_reg=bundle['lambda_reg'],design=bundle['design'],
        allocations=bundle['allocations'],seed_variants=bundle['seed_variants'],
        prediction_features=local_array('prediction') if 'prediction' in bundle else None,
        evidence_role=bundle.get('evidence_role','engineering_nonconfirmatory'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
