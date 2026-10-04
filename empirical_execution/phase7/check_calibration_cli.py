"""Independent intake-interface checks; mocked acceptance is not semantic evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccu.data import read_natural_jsonl, lexical_engineering_features
from phase4 import adapters
from phase7 import prepare_calibration as cli


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve prior checker result')
    outcomes = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        outcomes.append({'check': name, 'passed': True})

    def rejects(name, fn):
        try:
            fn()
        except (ValueError, OSError, KeyError, TypeError):
            check(name, True)
        else:
            raise AssertionError('Expected refusal: ' + name)

    natural = ROOT / 'data/civil_comments_engineering_preview.jsonl'
    rows = read_natural_jsonl(natural, require_labels=True)[:6]
    features, _ = lexical_engineering_features(rows, 32)
    provenance = {'dataset_id': 'civil_comments', 'encoder_id': 'lexical32-interface-test',
                  'encoder_revision': 'existing-lexical-software-only',
                  'population_scope': 'six-existing-Civil-preview-records-interface-test',
                  'evidence_role': 'engineering_nonconfirmatory'}
    source_paths = [Path(__file__), Path(cli.__file__), Path(cli.acceptance.__file__),
                    Path(cli.cal.__file__), Path(cli.task.__file__)]
    sources = {str(p.relative_to(ROOT)): cli.file_hash(p) for p in source_paths}
    mock_replay = {'schema': 'explicit-software-interface-mock-not-acceptance',
                   'machine_acceptance_passed': True, 'execution_allowed': False,
                   'source_authenticity_verified': False,
                   'human_authenticity_verified': False,
                   'complete_encoder_replay_performed': False}
    with tempfile.TemporaryDirectory(prefix='phase7-calibration-cli-') as temporary:
        base = Path(temporary)

        def fixture(name, source=None):
            folder = base / name
            folder.mkdir()
            (folder/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            np.save(folder/'features.npy', features, allow_pickle=False)
            write(folder/'provenance.json', provenance)
            write(folder/'source.json', source if source is not None else {'dataset_id': 'civil_comments'})
            def bound(file, kind):
                return {'path': file, 'kind': kind, 'sha256': cli.file_hash(folder/file)}
            dossier = {'source_acceptance': bound('source.json', 'json'),
                       'records': bound('records.jsonl', 'jsonl'),
                       'features': bound('features.npy', 'npy'),
                       'provenance': bound('provenance.json', 'json')}
            write(folder/'dossier.json', dossier)
            return folder, dossier

        folder, descriptor = fixture('bound')
        loaded, binding = cli.load_dossier(folder/'dossier.json')
        check('bound_records_equal_existing_natural_rows', loaded['records'] == rows)
        check('bound_FP32_features_exact', loaded['features'].dtype == np.float32 and np.array_equal(loaded['features'], features))
        check('all_four_descriptor_hashes_recorded', set(binding['bound_files']) == {'records.jsonl','features.npy','provenance.json','source.json'})
        check('loaded_fingerprint_recomputes', binding['loaded_value_sha256'] == cli.cal.digest(cli.task.fingerprint(loaded)))
        with patch.object(cli.acceptance, 'accept_calibration_inputs', return_value=copy.deepcopy(mock_replay)) as replay:
            report = cli.run('civil_comments', 'e5', folder/'dossier.json', folder/'success')
        check('positive_interface_path_only_with_explicit_mock', report['selection_pack_created'] and replay.call_count == 1)
        call = replay.call_args
        check('replay_receives_exact_loaded_values', call.args[:2] == ('civil_comments','e5') and cli.task.fingerprint(call.args[2]) == cli.task.fingerprint(loaded))
        check('dossier_parent_and_frozen_design_forwarded', call.kwargs['base_dir'] == folder.resolve() and call.kwargs['design'] == json.loads(cli.DESIGN.read_text()))
        pack = folder/'success/selection'
        manifest = json.loads((pack/'private_sampling_manifest.json').read_text())
        assignments = json.loads((pack/'blinded_assignments.json').read_text())
        template = json.loads((pack/'responses.template.json').read_text())
        check('fixed_seed_and_block_size', manifest['seed'] == 20271003 and manifest['frame']['scorer']['block_size'] == 256)
        check('pair_population_is_exact_existing_six_rows', manifest['pair_population'] == 15 and len(manifest['pairs']) == 15)
        check('three_blank_assignments_per_pair', len(assignments) == len(template['responses']) == 45)
        empty_fields = ('response_id','annotator_id','category','label','completed_at')
        check('every_response_is_blank', all(r['human_completed'] is False and all(r[k] == '' for k in empty_fields) for r in template['responses']))
        check('collection_declaration_remains_blank_and_false', template['collection_declaration']['responsible_collector'] == '' and all(template['collection_declaration'][k] is False for k in ('human_only','independent','blinded')))
        texts = {r['text'] for r in rows}
        check('assignment_texts_are_only_existing_texts', all(a['text_a'] in texts and a['text_b'] in texts for a in assignments))
        check('no_threshold_or_primary_authorization', not report['primary_execution_allowed'] and not report['source_authenticity_verified'] and report['human_responses_created'] == 0 and not report['human_collection_dispatched'] and not any('threshold' in p.name for p in pack.iterdir()))
        saved_before = {p.name: cli.file_hash(p) for p in pack.iterdir()}
        rejects('output_collision_refused', lambda: cli.run('civil_comments','e5',folder/'dossier.json',folder/'success'))
        check('collision_preserves_original_pack', saved_before == {p.name:cli.file_hash(p) for p in pack.iterdir()})

        for name, change in [('wrong_hash', lambda d: d['features'].update(sha256='0'*64)),
                             ('unsupported_kind', lambda d: d['features'].update(kind='pickle')),
                             ('path_escape', lambda d: d['features'].update(path='../bound/features.npy')),
                             ('nonstring_path', lambda d: d['features'].update(path=7))]:
            f, d = fixture(name); change(d); write(f/'dossier.json', d)
            rejects(name+'_refused', lambda f=f: cli.load_dossier(f/'dossier.json'))
        f, d = fixture('symlink_escape')
        (f/'external.npy').symlink_to(folder/'features.npy')
        d['features']['path'] = 'external.npy'; write(f/'dossier.json', d)
        rejects('symlink_escape_refused', lambda:cli.load_dossier(f/'dossier.json'))

        for name, file, text in [('duplicate_top','dossier.json','{"records":[],"records":[]}'),
                                 ('duplicate_nested_json','source.json','{"dataset_id":"civil_comments","dataset_id":"cc_news"}'),
                                 ('duplicate_jsonl','records.jsonl','{"record_id":"a","record_id":"b"}\n')]:
            f, d = fixture(name); (f/file).write_text(text)
            if file != 'dossier.json':
                key = 'source_acceptance' if file == 'source.json' else 'records'
                d[key]['sha256'] = cli.file_hash(f/file); write(f/'dossier.json',d)
            rejects(name+'_refused', lambda f=f:cli.load_dossier(f/'dossier.json'))
        rejects('nonfinite_JSON_refused',lambda:cli.strict_json('{"x": NaN}'))

        f, d = fixture('nested')
        write(f/'inner.json', {'dataset_id':'civil_comments'})
        write(f/'source.json', {'path':'inner.json','kind':'json','sha256':cli.file_hash(f/'inner.json')})
        d['source_acceptance']['sha256'] = cli.file_hash(f/'source.json'); write(f/'dossier.json',d)
        nested, nested_binding = cli.load_dossier(f/'dossier.json')
        check('nested_JSON_descriptor_loaded_and_bound', nested['source_acceptance'] == {'dataset_id':'civil_comments'} and 'inner.json' in nested_binding['bound_files'])
        # Controlled hash hook reaches the cycle guard; no cryptographic fixed point is claimed.
        f = base/'cycle'; f.mkdir()
        cycle = {'path':'dossier.json','kind':'json','sha256':'cycle-test-only'}
        write(f/'dossier.json',cycle)
        with patch.object(cli,'file_hash',return_value='cycle-test-only'):
            rejects('cycle_guard_refuses_self_reference_under_explicit_hash_mock',lambda:cli.load_dossier(f/'dossier.json'))

        missing_source = {'dataset_id':'civil_comments','adapter':{'directory':'absent-adapter',
                          'inputs':{'records':'absent-original.jsonl'},'kwargs':{'input_schema':'engineering_preview'}}}
        f, _ = fixture('actual_missing_source',missing_source)
        actual = cli.run('civil_comments','e5',f/'dossier.json',f/'out')
        check('actual_unmocked_missing_source_refused', not actual['selection_pack_created'] and actual['failure']['stage'] == 'original_source_and_complete_encoder_replay')
        check('no_pack_after_actual_replay_failure',not (f/'out/selection').exists())
        check('failure_receipt_and_replay_preserved',(f/'out/receipt.json').is_file() and (f/'out/acceptance.json').is_file())
        f, _ = fixture('missing_dossier')
        actual = cli.run('civil_comments','e5',f/'absent.json',f/'out')
        check('actual_missing_dossier_refused_without_pack',not actual['selection_pack_created'] and not (f/'out/selection').exists())

        adapter_dir = base/'actual_preview_adapter'
        adapters.adapt_civil(natural,adapter_dir,input_schema='engineering_preview')
        preview_spec = {'dataset_id':'civil_comments','adapter':{'directory':str(adapter_dir),
                        'inputs':{'records':str(natural)},'kwargs':{'input_schema':'engineering_preview'}}}
        f, _ = fixture('actual_preview',preview_spec)
        actual = cli.run('civil_comments','e5',f/'dossier.json',f/'out')
        replay = json.loads((f/'out/acceptance.json').read_text())
        check('actual_preview_bytes_replayed', replay['source']['rows'] == 100 and replay['source']['engineering_preview'])
        check('engineering_preview_cannot_create_semantic_pack',not actual['selection_pack_created'] and not replay['machine_acceptance_passed'] and not (f/'out/selection').exists())

        f, _ = fixture('wcep')
        with patch.object(cli.acceptance,'accept_calibration_inputs') as replay:
            rejected = cli.run('wcep100','e5',f/'dossier.json',f/'out')
        check('WCEP_refused_before_replay',not rejected['selection_pack_created'] and replay.call_count == 0 and not (f/'out/selection').exists())
        f, _ = fixture('machine_false')
        with patch.object(cli.acceptance,'accept_calibration_inputs',return_value={**mock_replay,'machine_acceptance_passed':False}):
            rejected = cli.run('civil_comments','e5',f/'dossier.json',f/'out')
        check('explicit_machine_refusal_never_creates_pack',not rejected['selection_pack_created'] and not (f/'out/selection').exists())
        f, _ = fixture('truthy_not_boolean')
        with patch.object(cli.acceptance,'accept_calibration_inputs',return_value={**mock_replay,'machine_acceptance_passed':'yes'}):
            rejected = cli.run('civil_comments','e5',f/'dossier.json',f/'out')
        check('truthy_nonboolean_acceptance_refused',not rejected['selection_pack_created'])
        f, _ = fixture('mutation')
        def mutate(*args, **kwargs):
            (f/'records.jsonl').write_text((f/'records.jsonl').read_text()+'\n')
            return copy.deepcopy(mock_replay)
        with patch.object(cli.acceptance,'accept_calibration_inputs',side_effect=mutate):
            rejected = cli.run('civil_comments','e5',f/'dossier.json',f/'out')
        check('input_mutation_during_replay_refused',not rejected['selection_pack_created'] and rejected['failure']['stage']=='unchanged_inputs_and_code')
        check('no_pack_after_detected_input_mutation',not (f/'out/selection').exists())
        check('sampling_and_loader_sources_explicitly_bound', all(cli.file_hash(p) in cli.code_bindings().values() for p in (Path(cli.cal.__file__),Path(cli.task.__file__))))

    check('checker_and_frozen_dependencies_unchanged',all(cli.file_hash(ROOT/name)==sha for name,sha in sources.items()))
    result = {'schema':'ccu-calibration-cli-independent-checks-1','status':'passed',
              'check_count':len(outcomes),'checks':outcomes,'source_sha256':sources,
              'natural_input_sha256':cli.file_hash(natural),'positive_pack_scope':'Transient interface mock using six actual Civil rows and lexical features; not source or encoder acceptance.',
              'mock_pack_pairs':15,'mock_blank_assignments':45,
              'actual_source_rich_archive_accepted':False,'actual_semantic_encoder_replayed':False,
              'actual_human_responses_created':0,'primary_study_started':False,
              'pack_artifacts_retained':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    write(out,result)
    return result


if __name__ == '__main__':
    destination = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'phase7/results/calibration_cli_checks.json'
    print(json.dumps(run(destination),indent=2))
