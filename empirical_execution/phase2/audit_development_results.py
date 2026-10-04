#!/usr/bin/env python3
"""Independent saved-result audit; imports no proposed repair/analysis code.

Reconstructs natural-data partitions, label-blind request streams and the
earlier-raw-neighbor target. Uses the dual ridge system to independently check
the runner's primal-system predictions. This does not upgrade development data
to confirmation or certify semantic-duplicate quality.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'phase2/results'


def read(name):
    return json.loads((OUT / name).read_text())


def lines(name):
    return [json.loads(x) for x in (OUT / name).read_text().splitlines()]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def features(rows, d):
    return HashingVectorizer(n_features=d, analyzer='word', ngram_range=(1, 2),
        alternate_sign=False, norm='l2', lowercase=False,
        token_pattern=r'(?u)\b\w+\b', dtype=np.float32).transform(
            [normalized(r['text']) for r in rows]).toarray()


def main():
    result = read('summary.json')
    assert result['status'] == 'completed_development_not_confirmation'
    design = read('design_lock.json')
    for path, digest in design['source_hashes'].items():
        assert sha((ROOT/path).read_bytes()) == digest, path
    raw = ROOT/'data/civil_comments_engineering_preview.jsonl'
    assert sha(raw.read_bytes()) == design['corpus_sha256']
    records = [json.loads(x) for x in raw.read_text().splitlines()]
    partitions = read('partition_manifest.json')
    assert [r['record_id'] for r in records] == [r['record_id'] for r in partitions]
    train, evaluation = [], []
    for row, part in zip(records, partitions):
        text = normalized(row['text'])
        digest = sha((design['split_salt']+'\0'+text).encode())
        selected = int(digest, 16)*5 < 4*(1 << 256)
        assert part['partition_hash'] == digest
        assert part['text_group_sha256'] == sha(text.encode())
        assert part['partition'] == ('development_train' if selected else 'development_evaluation')
        (train if selected else evaluation).append(row)
    assert not ({normalized(r['text']) for r in train} &
                {normalized(r['text']) for r in evaluation})
    ids = [r['record_id'] for r in train]
    lookup = {name:i for i,name in enumerate(ids)}
    order = sorted(range(len(ids)), key=lambda i:(
        hashlib.sha256(('ccu-priority-v1\0'+ids[i]).encode()).digest(), ids[i]))
    z = features(train, 128).astype(np.float64)
    z /= np.linalg.norm(z, axis=1)[:,None]
    blockers = [set() for _ in ids]
    for position, i in enumerate(order):
        blockers[i] = {j for j in order[:position] if float(z[j] @ z[i]) > .6}
    original = {i for i in order if not blockers[i]}
    geometry = read('geometry.json')
    ez = features(evaluation,128).astype(np.float64)
    ez /= np.linalg.norm(ez,axis=1)[:,None]
    crosssplit = z @ ez.T
    assert geometry['training_records'] == len(train)
    assert geometry['evaluation_records'] == len(evaluation)
    assert geometry['initially_selected'] == len(original)
    assert geometry['initially_excluded'] == len(train)-len(original)
    assert geometry['train_graph_edges'] == sum(map(len,blockers))
    assert geometry['crosssplit_lexical_pairs_above_threshold'] == int((crosssplit>.6).sum())
    assert np.isclose(geometry['crosssplit_max_lexical_score'],crosssplit.max(),atol=1e-14,rtol=1e-12)
    assert not geometry['semantic_or_source_leakage_guard_complete']
    frame_all = tuple(sorted(ids))
    frame_excluded = tuple(sorted(ids[i] for i in range(len(ids)) if blockers[i]))
    requests = read('requests.json')
    assert Counter(r['arm'] for r in requests) == Counter(design['allocations'])
    for request in requests:
        arm = request['arm']
        path_index = str(int(request['trajectory_id'][1:]))
        digest = hashlib.sha256()
        for value in ('20271003','civil_phase2_development',arm,path_index):
            value = value.encode()
            digest.update(len(value).to_bytes(8,'big'));digest.update(value)
        seed = int.from_bytes(digest.digest()[:16], 'big')
        assert seed == request['seed']
        rng = random.Random(seed)
        frame = frame_excluded if arm == 'U' else frame_all
        horizon = min(8, len(frame))
        if arm != 'A':
            expected = tuple(rng.sample(frame, horizon))
        else:
            pool = [name for name in frame_excluded if len(blockers[lookup[name]]) <= horizon]
            candidates = rng.sample(pool, min(len(pool),1024))
            def admissions(dead):
                return sum(bool(blockers[i]) and i not in dead and blockers[i] <= dead
                           for i in range(len(ids)))
            candidate = min(candidates, key=lambda name:(-admissions(blockers[lookup[name]]),name)) if candidates else None
            prefix = tuple(sorted(ids[b] for b in blockers[lookup[candidate]])) if candidate else ()
            expected = prefix + tuple(rng.sample([name for name in frame if name not in prefix], horizon-len(prefix)))
            assert request['selected_candidate'] == candidate
        assert tuple(request['deletion_order']) == expected
        assert request['initial_horizon'] == horizon
        assert request['checkpoints'] == sorted({min(k,horizon) for k in (1,2,4,8)}) if horizon else not request['checkpoints']
    request_map = {r['trajectory_id']:r for r in requests}
    outcomes = lines('checkpoints.jsonl')
    predictions = lines('predictions.jsonl')
    key = lambda r:(r['dimension'],r['trajectory_id'],r['checkpoint'])
    predicted = {key(r):r for r in predictions}
    assert len(predicted) == len(predictions) == len(outcomes)
    expected_keys = {(d,r['trajectory_id'],k) for d in design['dimensions']
                     for r in requests for k in r['checkpoints']}
    assert {key(r) for r in outcomes} == expected_keys
    target = np.asarray([r['label'] for r in evaluation])[:,None]
    labels = np.asarray([r['label'] for r in train])[:,None]
    target_artifact = read('evaluation_targets.json')
    assert target_artifact['record_ids'] == [r['record_id'] for r in evaluation]
    assert target_artifact['original_labels'] == target[:,0].tolist()
    max_prediction_error = 0.0
    max_metric_error = 0.0
    counts = {'source_hashes':len(design['source_hashes']), 'partition_assignments':len(records),
              'request_streams':len(requests), 'checkpoint_rows':len(outcomes),
              'metric_scalar_checks':0, 'bootstrap_metric_intervals':0,
              'utility_context_checks':0, 'geometry_fields':8}
    utility = {r['dimension']:r for r in read('utility_context.json')}
    feature_metadata = {r['dimension']:r for r in read('feature_metadata.json')}
    for d in design['dimensions']:
        x32 = features(train,d);ex32 = features(evaluation,d)
        assert feature_metadata[d]['train']['array_sha256'] == sha(x32.tobytes())
        assert feature_metadata[d]['evaluation']['array_sha256'] == sha(ex32.tobytes())
        x = x32.astype(np.float64);ex = ex32.astype(np.float64)
        def predict(selected):
            ix = sorted(selected)
            if not ix:return np.zeros_like(target)
            block = x[ix]
            dual = np.linalg.solve(block @ block.T + .01*len(ix)*np.eye(len(ix)),labels[ix])
            return (ex @ block.T) @ dual
        pre = predict(original)
        before_loss = float(np.mean((target-pre)**2))
        shifted = pre-pre[:1]
        before_sd = float(np.sqrt(np.mean((shifted-shifted.mean(axis=0,keepdims=True))**2)))
        control_order = sorted(range(len(ids)),key=lambda i:sha(('ccu-phase2-size-control-v1\0'+ids[i]).encode()))
        utility_values = {'predelete_curated_mse':before_loss,'predelete_score_sd':before_sd,
            'uncurated_mse':float(np.mean((target-predict(range(len(ids))))**2)),
            'same_size_hash_mse':float(np.mean((target-predict(control_order[:len(original)]))**2)),
            'training_mean_constant_mse':float(np.mean((target-labels.mean(axis=0))**2)),
            'training_mean':float(labels.mean()),'evaluation_records':len(evaluation)}
        for name,value in utility_values.items():
            assert np.isclose(utility[d][name],value,atol=2e-11,rtol=2e-10)
            counts['utility_context_checks'] += 1
        assert np.allclose(utility[d]['predelete_predictions'],pre[:,0],atol=2e-11,rtol=2e-10)
        cache = {}
        for row in (r for r in outcomes if r['dimension']==d):
            req = request_map[row['trajectory_id']]
            deleted = req['deletion_order'][:row['checkpoint']]
            assert deleted == row['deleted_ids']
            dead = {lookup[name] for name in deleted}
            live_order = [i for i in order if i not in dead]
            selected = {i for p,i in enumerate(live_order)
                        if not any(float(z[j] @ z[i]) > .6 for j in live_order[:p])}
            additions = selected-original
            assert set(row['selected_ids']) == {ids[i] for i in selected}
            assert set(row['added_ids']) == {ids[i] for i in additions}
            assert row['addition_count'] == len(additions)
            assert row['removed_selected_count'] == len(original & dead)
            assert row['selected_count'] == len(selected)
            assert row['frozen_selected_count'] == len(original-dead)
            cache_key = tuple(sorted(dead))
            if cache_key not in cache:cache[cache_key] = (predict(selected), predict(original-dead))
            oracle, frozen = cache[cache_key]
            stored = predicted[key(row)]
            for name,value in [('oracle',oracle),('frozen',frozen)]:
                err = float(np.max(np.abs(value[:,0]-np.asarray(stored[name+'_predictions']))))
                max_prediction_error = max(max_prediction_error,err)
                assert err < 2e-11,(key(row),name,err)
            ol = float(np.mean((target-oracle)**2));fl = float(np.mean((target-frozen)**2))
            rms = float(np.sqrt(np.mean((oracle-frozen)**2)))
            metrics = {'oracle_mse':ol,'frozen_mse':fl,'mse_effect':fl-ol,
                'predelete_mse':before_loss,'predelete_score_sd':before_sd,
                'relative_mse_effect':(fl-ol)/before_loss if before_loss else None,
                'prediction_rms':rms,'normalized_prediction_rms':rms/before_sd if before_sd else None,
                'baseline_mse':float(np.mean((target-labels.mean(axis=0))**2))}
            for name,value in metrics.items():
                counts['metric_scalar_checks'] += 1
                if value is None:assert row[name] is None
                else:
                    error = abs(value-row[name]);max_metric_error = max(max_metric_error,error)
                    assert np.isclose(value,row[name],atol=2e-11,rtol=2e-10),(key(row),name,value,row[name])
        analysis = read(f'analysis_d{d}.json')
        for group in analysis['groups']:
            chosen = sorted([r for r in outcomes if r['dimension']==d and
                r['arm']==group['arm'] and r['checkpoint']==group['checkpoint']],key=lambda r:r['trajectory_id'])
            rng = np.random.default_rng(int(group['bootstrap_group_seed']))
            samples = rng.integers(0,len(chosen),size=(design['bootstrap_samples'],len(chosen)))
            assert group['trajectory_ids'] == [r['trajectory_id'] for r in chosen]
            for name, measured in group['metrics'].items():
                if name == 'activation_frequency':values = np.array([float(r['addition_count']>0) for r in chosen])
                elif name == 'addition_count_conditional_active':values = np.array([r['addition_count'] or np.nan for r in chosen])
                else:values = np.array([r.get(name) if r.get(name) is not None else np.nan for r in chosen],dtype=float)
                valid = np.isfinite(values)
                assert measured['n_defined'] == int(valid.sum())
                if valid.any():assert np.isclose(measured['mean'],values[valid].mean(),atol=1e-14,rtol=1e-12)
                else:assert measured['mean'] is None
                drawn = values[samples]
                denominator = np.isfinite(drawn).sum(axis=1)
                boot = np.nansum(drawn,axis=1)[denominator>0]/denominator[denominator>0]
                assert measured['bootstrap_defined_replicates'] == len(boot)
                if len(boot):
                    interval = np.quantile(boot,[.025,.975],method='linear')
                    assert np.allclose(interval,measured['bootstrap_percentile_interval_95'],atol=1e-14,rtol=1e-12)
                else:assert measured['bootstrap_percentile_interval_95'] is None
                counts['bootstrap_metric_intervals'] += 1
    audits = read('strict_audits.json')
    assert len(audits) == result['strict_audited_checkpoints'] == 12
    assert all(a['fresh_bytes_equal'] for a in audits)
    report = {'status':'passed','checks':counts,'max_dual_vs_saved_prediction_absolute_error':max_prediction_error,
        'max_recomputed_metric_absolute_error':max_metric_error,
        'independent_solver':'dual ridge on selected FP32-promoted features; production uses primal Cholesky',
        'imports_production_curation_repair_or_analysis_code':False,
        'semantic_quality_or_confirmatory_gate_passed':False,
        'audit_sha256':sha(Path(__file__).read_bytes()),
        'result_summary_sha256':sha((OUT/'summary.json').read_bytes())}
    (OUT/'independent_result_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
