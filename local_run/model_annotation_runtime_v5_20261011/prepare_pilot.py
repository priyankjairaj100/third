"""Freeze authentic engineering-pair inputs and cached model assets, without inference."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def frozen_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--repo', required=True, type=Path)
    a.add_argument('--workspace', required=True, type=Path)
    a.add_argument('--model-path', required=True, type=Path)
    a.add_argument('--output-dir', required=True, type=Path)
    a.add_argument('--seed', type=int, default=20261011)
    a.add_argument('--prior-run', required=True, type=Path)
    args = a.parse_args()
    runtime_plan_file = Path(__file__).parent / 'runtime_plan.json'
    runtime_plan_sha = sha(runtime_plan_file)
    runtime_plan = json.loads(runtime_plan_file.read_text())
    for name, expected in runtime_plan['package_files'].items():
        assert sha(runtime_plan_file.parent / name) == expected, name
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    plan_file = args.repo.resolve() / 'local_run/model_annotation_v2_20261011/PLAN.json'
    plan_sha = sha(plan_file)
    plan = json.loads(plan_file.read_text())
    for name, expected in plan['package_files'].items():
        assert sha(plan_file.parent / name) == expected, name
    pack = args.workspace.resolve() / 'calibration/askubuntu/e5/mean_removed_a01/selection/private_sampling_manifest.json'
    records_file = args.workspace.resolve() / 'prepared/askubuntu/a02/population/records.jsonl'
    pack_sha, records_sha = sha(pack), sha(records_file)
    source = json.loads(pack.read_text())
    assert source['sha256'] == frozen_digest({k: v for k, v in source.items() if k != 'sha256'}), 'Source pack seal mismatch'
    pairs = source['pairs']
    assert len(pairs) == 420, 'Unexpected source pack size; inspect before changing pilot'
    identities = [tuple(sorted([x['left_id'], x['right_id']])) for x in pairs]
    assert len(set(identities)) == len(identities), 'Duplicate unordered native pair'
    assert all(x[0] != x[1] for x in identities)
    ranked = sorted(pairs, key=lambda p: (digest(['pilot-pair-v2', args.seed, pack_sha, sorted([p['left_id'], p['right_id']])]), p['pair_id']))
    selected = ranked[:40]
    swap_ids = {p['pair_id'] for p in sorted(selected, key=lambda p: digest(['pilot-swap-v2', args.seed, pack_sha, p['pair_id']]))[:4]}
    wanted = {p[k] for p in pairs for k in ('left_id', 'right_id')}
    texts, frame_records = {}, []
    with records_file.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['partition'] == 'calibration':
                frame_records.append({k: row[k] for k in ('record_id', 'text')})
            if row['record_id'] in wanted:
                assert row['record_id'] not in texts
                assert row['partition'] == 'calibration'
                assert isinstance(row['text'], str) and row['text']
                texts[row['record_id']] = row['text']
    frame_sha = frozen_digest(frame_records)
    assert frame_sha == source['frame']['records_sha256'], 'Text/record order differs from source sampling frame'
    assert len({r['record_id'] for r in frame_records}) == len(frame_records), 'Repeated calibration ID'
    assert len(frame_records) * (len(frame_records) - 1) // 2 == source['pair_population']
    assert set(texts) == wanted, 'Source records missing'
    inputs, mapping = [], []
    for p in selected:
        native = sorted([p['left_id'], p['right_id']])
        opaque = digest(['pilot-opaque-v2', args.seed, pack_sha, native])
        flip = int(digest(['orientation-v2', plan_sha, native]), 16) % 2
        shown = native[::-1] if flip else native
        row = {'pair_id': opaque, 'text_a': texts[shown[0]], 'text_b': texts[shown[1]], 'orientation': 'canonical'}
        inputs.append(row)
        mapping.append({'pair_id': opaque, 'source_pair_id': p['pair_id'], 'native_pair': native, 'shown_ids': shown, 'source_bin': p['bin'], 'swapped_diagnostic': p['pair_id'] in swap_ids})
    for row, item in zip(inputs[:], mapping):
        if item['swapped_diagnostic']:
            inputs.append({**row, 'text_a': row['text_b'], 'text_b': row['text_a'], 'orientation': 'swapped_diagnostic'})
    inputs_file = out / 'inputs.jsonl'
    inputs_file.write_text(''.join(json.dumps(r, ensure_ascii=True, sort_keys=True) + '\n' for r in inputs))
    assert sha(inputs_file) == '9fbf8ec87c2571d4dee65d5c0b851f36b50c430a3cf1973796e54cbe0def1a5a', 'Original pilot inputs changed'
    assert sha(inputs_file) == sha(args.prior_run / 'preparation/inputs.jsonl')
    save(out / 'private_source_mapping.json', mapping)
    model = args.model_path.absolute()
    assert model.name == plan['models'][0]['revision'], 'Unexpected cached revision'
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    needed = set(index['weight_map'].values()) | {'model.safetensors.index.json', 'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}
    for optional in ('special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja'):
        if (model / optional).exists():
            needed.add(optional)
    expectations_file = args.repo.resolve() / 'local_run/model_annotation_runtime_v2_20261011/qwen_asset_expectations.json'
    official = json.loads(expectations_file.read_text())
    assert official['revision'] == model.name
    needed |= {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file()}
    assets = []
    for name in sorted(needed):
        p = model / name
        assert p.is_file(), name
        expected = official['files'][name]
        size, observed_sha = p.stat().st_size, sha(p)
        assert size == expected['size_bytes'], name
        if 'sha256' in expected:
            assert observed_sha == expected['sha256'], name
        else:
            blob = p.read_bytes()
            git_sha = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
            assert git_sha == expected['git_blob_sha1'], name
        assets.append({'relative_path': name, 'size_bytes': size, 'sha256': observed_sha})
        print('hashed', name, flush=True)
    weights = out / 'weights_manifest.json'
    save(weights, {'schema': 'third-judge-assets-v1', 'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'revision': model.name, 'files': assets, 'total_bytes': sum(x['size_bytes'] for x in assets), 'official_expectations_sha256': sha(expectations_file), 'official_identity_verified': True})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    system = (plan_file.parent / 'judge_system_prompt.txt').read_text()
    task = (plan_file.parent / 'judge_task_definition.txt').read_text()
    census = []
    for p in pairs:
        native = sorted([p['left_id'], p['right_id']])
        if int(digest(['orientation-v2', plan_sha, native]), 16) % 2:
            native.reverse()
        for orientation, ids in [('canonical', native), ('swapped_diagnostic', native[::-1])]:
            messages = [{'role':'system','content':system},{'role':'user','content':json.dumps({'task_definition':task,'documents':[{'label':'A','text':texts[ids[0]]},{'label':'B','text':texts[ids[1]]}]},ensure_ascii=False)}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            n = len(tokenizer(rendered, add_special_tokens=False, truncation=False)['input_ids'])
            census.append({'source_pair_id':p['pair_id'],'orientation':orientation,'input_tokens':n,'fits_32768_with_512_output':n+512<=32768})
    census_path = out / 'engineering_frame_token_census.private.jsonl'
    census_path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in census))
    census_summary = {'scope':'all420 existing engineering pairs in both orientations, not the full scientific pair population','rows':len(census),'max_input_tokens':max(x['input_tokens'] for x in census),'over_native_budget':sum(not x['fits_32768_with_512_output'] for x in census),'native_total_budget':32768,'output_reserve':512,'census_sha256':sha(census_path)}
    save(out / 'engineering_frame_token_census_summary.json', census_summary)
    save(out / 'preparation.json', {'schema': 'third-pilot-preparation-v5', 'evidence_role': 'engineering_only', 'source_pack_sha256': pack_sha, 'source_records_sha256': records_sha, 'ordered_calibration_frame_sha256': frame_sha, 'source_pack_seal_verified': True, 'source_pack_pairs': len(pairs), 'selected_unique_pairs': len(selected), 'scheduled_calls': len(inputs), 'swapped_diagnostic_pairs': 4, 'seed': args.seed, 'selection': 'First 40 SHA256 hash-ranked unique native pairs from frozen 420-pair engineering pack; not a population precision sample', 'source_mapping_sha256': sha(out / 'private_source_mapping.json'), 'full_text_source': str(records_file), 'prior_model_exposure': 'Existing engineering pack and earlier subset labels are exposed development history; no prior judgments appear in inputs', 'new_human_responses': 0, 'prior_run': str(args.prior_run), 'engineering_frame_token_census': census_summary, 'prepare_script_sha256': sha(Path(__file__))})
    assert sha(pack) == pack_sha and sha(records_file) == records_sha, 'Source changed during preparation'
    assert sha(plan_file) == plan_sha, 'Plan changed during preparation'
    manifest = {'schema': 'third-judge-pilot-launch-v5', 'runtime_plan_file': str(runtime_plan_file), 'runtime_plan_sha256': runtime_plan_sha, 'engineering_frame_census_file': str(census_path), 'engineering_frame_census_sha256': sha(census_path), 'pilot_id': out.parent.name, 'repository_root': str(args.repo.resolve()), 'runtime_dependency_source': 'Dedicated judge venv referencing the unchanged encoder-cuda site-packages read-only; package versions and module paths sealed by runner', 'plan_file': str(plan_file), 'plan_sha256': plan_sha, 'inputs_file': str(inputs_file), 'inputs_sha256': sha(inputs_file), 'weights_manifest_file': str(weights), 'weights_manifest_sha256': sha(weights), 'model_path': str(model), 'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'model_revision': model.name, 'source_role': 'engineering_only', 'seed': args.seed, 'expected_pairs': 40, 'expected_calls': 44, 'preparation_sha256': sha(out / 'preparation.json'), 'driver_sha256': sha(Path(__file__).parent / 'run_pilot.py')}
    assert sha(runtime_plan_file) == runtime_plan_sha
    save(out / 'manifest.json', manifest)
    print(json.dumps({'prepared': True, 'manifest_sha256': sha(out / 'manifest.json'), 'pairs': 40, 'calls': 44}), flush=True)


if __name__ == '__main__':
    main()
