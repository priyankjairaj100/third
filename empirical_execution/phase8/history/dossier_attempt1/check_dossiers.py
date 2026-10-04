"""Scoped dossier-interface checks on existing Civil text; never semantic evidence."""
from pathlib import Path
import copy
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from ccu.data import lexical_engineering_features
from phase4 import adapters
from phase8 import dossiers as d


def run(out):
    out=Path(out)
    if out.exists(): raise FileExistsError('Preserve prior result')
    checks=[]
    def check(name,value):
        if not value: raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try: fn()
        except (ValueError,KeyError,OSError,RuntimeError,TypeError): check(name,True)
        else: raise AssertionError(name)
    codes=d.codes(); codes['phase8/check_dossiers.py']=d.file_hash(__file__)
    source=ROOT/'data/civil_comments_engineering_preview.jsonl'
    design=d.strict_json(d.DESIGN.read_text())
    with tempfile.TemporaryDirectory(prefix='phase8-dossier-software-') as temporary:
        t=Path(temporary)
        adapters.adapt_civil(source,t/'adapter',input_schema='engineering_preview')
        raw=d.prep.read_rows(t/'adapter/records.jsonl')
        rows,_,_=d.panels.prepare_population(raw,'civil_comments',design,target=10)
        features,_=lexical_engineering_features(rows,768)
        calrows=[r for r in rows if r['partition']=='calibration']; ix=[i for i,r in enumerate(rows) if r['partition']=='calibration']
        provenance={'dataset_id':'civil_comments','encoder_id':d.emb.E5,'encoder_revision':'explicit-lexical-mock-only',
            'population_scope':'existing-Civil-preparation-calibration-subset','evidence_role':'engineering_nonconfirmatory',
            'software_mock_not_semantic_cache':True}
        spec={'dataset_id':'civil_comments','adapter':{'directory':str(t/'adapter'),'inputs':{'records':str(source)},
            'kwargs':{'input_schema':'engineering_preview'}},'prepared':{'records':str(t/'prepared.jsonl'),'audit':str(t/'prepared-audit.json')},
            'caches':{'e5':{'directory':str(t/'cache'),'model_dir':str(t/'absent-model'),'tokenizer_dir':str(t/'absent-model')}}}
        (t/'cache').mkdir(); np.save(t/'cache/vectors.npy',features,allow_pickle=False)
        d.write(t/'source.json',spec)
        dossier={'source_acceptance':spec,'records':calrows,'features':features[ix],'provenance':provenance}
        mock={'schema':'explicit-software-mock-no-original-archive-or-encoder-replay','machine_acceptance_passed':True,
              'execution_allowed':False,'replay_binding_sha256':'f'*64}
        def archive_mock(*args): return {'required':True,'scope':'explicit-mock-no-archive-replay','replay_binding_sha256':'a'*64}
        with patch.object(d,'archive_replay',side_effect=archive_mock),patch.object(d,'calibration_values',return_value=dossier),\
             patch.object(d.acceptance,'accept_calibration_inputs',return_value=mock) as replay:
            result=d.build_calibration(t/'source.json','e5',t/'calibration')
        check('calibration_stage_positive_path_explicitly_mocked',result['status']=='calibration_dossier_ready_external_review_pending')
        saved=d.load_value(t/'calibration/dossier.json')
        check('exact_prethreshold_four_field_schema',set(saved)=={'source_acceptance','records','features','provenance'})
        check('complete_original_calibration_rows_and_lexical_values_preserved',d.task.fingerprint(saved)==d.task.fingerprint(dossier))
        check('actual_acceptance_api_receives_no_threshold_or_responses',replay.call_count==1 and 'threshold_lock' not in replay.call_args.args[2] and 'selection_responses' not in replay.call_args.args[2])
        check('calibration_stage_creates_no_annotations',not (t/'calibration/selection').exists() and result['human_responses_created']==0)
        check('calibration_stage_remains_unapproved',not result['primary_execution_allowed'] and not result['external_testimony_created'])
        check('private_output_guard_exists',(t/'calibration/.gitignore').read_text()=='*\n')
        frozen=d.file_hash(t/'calibration/dossier.json')
        rejects('existing_stage_output_refused',lambda:d.build_calibration(t/'source.json','e5',t/'calibration'))
        check('existing_output_bytes_unchanged',d.file_hash(t/'calibration/dossier.json')==frozen)
        with patch.object(d,'archive_replay',side_effect=archive_mock),patch.object(d,'calibration_values',return_value=dossier),\
             patch.object(d.acceptance,'accept_calibration_inputs',return_value={**mock,'machine_acceptance_passed':False}):
            rejected=d.build_calibration(t/'source.json','e5',t/'rejected_replay')
        check('machine_refusal_blocks_dossier_output',rejected['status']=='blocked' and not (t/'rejected_replay/dossier.json').exists())
        absent=d.build_calibration(t/'absent.json','e5',t/'absent')
        check('actual_missing_source_spec_fails_closed',absent['status']=='blocked' and not (t/'absent/dossier.json').exists())
        noarchive=d.build_calibration(t/'source.json','e5',t/'no_archive')
        check('actual_missing_original_archive_binding_fails_closed',noarchive['status']=='blocked' and not (t/'no_archive/dossier.json').exists())
        d.write(t/'wcep_source.json',{'dataset_id':'wcep100'})
        rejected=d.build_calibration(t/'wcep_source.json','e5',t/'wcep')
        check('WCEP_cannot_create_own_calibration_dossier',rejected['status']=='blocked')

        # Actual blank response validation refuses before any threshold exists.
        manifest=d.cal.prepare_selection(features[ix],calrows,provenance)
        d.cal.write_blinded_pack(manifest,calrows,t/'blank')
        evidence={'selection_manifest':'blank/private_sampling_manifest.json','selection_responses':'blank/responses.template.json',
                  'validation_manifest':'blank/private_sampling_manifest.json','validation_responses':'blank/responses.template.json'}
        with patch.object(d,'calibration_values',return_value=dossier):
            rejects('actual_blank_human_responses_cannot_complete_curator',lambda:d.completed_curator(spec,'e5',evidence,t))

        # Pure assembly fixture: original rows/labels/source ownership; only the
        # unavailable human and task-choice locks are explicit software mocks.
        registry,_=d.recipes.build_registry()
        group=copy.deepcopy(next(g for g in registry['groups'] if g['group_id']=='C_full_methods/civil_comments/primary-10000/native'))
        allocation_group={**group,'allocations':{'R':1,'S':1,'U':0,'A':0}}
        small_registry={'groups':[allocation_group]}
        train=[r['record_id'] for r in rows if r['partition']=='train'][:20]
        evaluation=[r['record_id'] for r in rows if r['partition']=='test'][:8]
        threshold=d.task.engineering_parameter('threshold',.6,representation='e5')
        threshold=d.cal._seal({**{k:v for k,v in threshold.items() if k!='sha256'},'threshold':.6})
        curator={'encoder_id':d.emb.E5,'encoder_revision':provenance['encoder_revision'],
                 'threshold_lock':threshold,'calibration':{k:v for k,v in dossier.items() if k!='source_acceptance'}}
        def parameters(values,records,prov,vocab,encoder,revision,logistic=False):
            return {'encoder_id':encoder,'encoder_revision':revision,'calibration':{'features':values,'records':records,'provenance':prov},
                    'lambda_lock':d.task.engineering_parameter('lambda',.01,representation='e5')}
        with patch.object(d,'_learner_parameters',side_effect=parameters):
            bundle=d.assemble_bundle([group],small_registry,copy.deepcopy(spec),rows,train,evaluation,{'e5':curator},{'e5':features},design)
        lookup={r['record_id']:r for r in rows}
        check('original_train_toxicity_targets_preserved',bundle['train_y'].dtype==np.float64 and bundle['train_y'].tolist()==[[lookup[i]['original_fields']['toxicity']] for i in train])
        check('original_heldout_targets_only_encoded_for_evaluation',bundle['evaluation_y'].tolist()==[[lookup[i]['original_fields']['toxicity']] for i in evaluation])
        check('source_ids_derive_from_actual_prepared_rows',bundle['source_ids']==[lookup[i]['source_unit_id'] for i in train])
        check('unknown_sources_not_promoted',set(bundle['source_kinds'].values())=={'unknown_singleton'} and not bundle['requests']['genuine_source_sampling_frame'])
        check('ordered_encoder_rows_preserved',np.array_equal(bundle['curators']['e5']['train'],features[[rows.index(lookup[i]) for i in train]]))
        check('calibration_heldout_from_train_and_test',not set(r['record_id'] for r in calrows)&set(train+evaluation))
        check('task_choice_receives_only_calibration_rows',bundle['learners']['e5']['calibration']['records']==calrows)
        check('request_source_arm_remains_unavailable',all(not p['checkpoints'] for p in bundle['requests']['trajectories'] if p['arm']=='S'))
        check('selected_group_scope_bound_to_bundle',bundle['_dossier_group_ids']==[group['group_id']])
        projected=copy.deepcopy(group);projected['configuration']={**projected['configuration'],'learner_dimension':32,'projection_seed':20271005}
        with patch.object(d,'_learner_parameters',side_effect=parameters):
            projected_bundle=d.assemble_bundle([projected],small_registry,copy.deepcopy(spec),rows,train,evaluation,{'e5':curator},{'e5':features},design)
        matrix=projected_bundle['projections']['32']['matrix']; parameters32=projected_bundle['learners']['e5']['projection_parameters']['32']
        check('projection_uses_frozen_seeded_matrix',np.array_equal(matrix,d.task.seeded_projection(768,32,20271005)))
        check('projection_calibration_is_exact_stored_FP32_transform',np.array_equal(parameters32['calibration']['features'],(features[ix].astype(np.float64)@matrix).astype(np.float32)))
        check('projection_lineage_binds_original_full_cache',parameters32['calibration']['provenance']['projection_lineage']['source_feature_cache_sha256']==d.file_hash(t/'cache/vectors.npy'))
        unlabeled=copy.deepcopy(spec);unlabeled['dataset_id']='cc_news'
        news_group=copy.deepcopy(group);news_group['corpus']='cc_news';news_group['request_family']='cc_news/primary-10000'
        with patch.object(d,'_learner_parameters',side_effect=AssertionError('No label choice on News')):
            news=d.assemble_bundle([news_group],{'groups':[{**news_group,'allocations':{'R':1,'S':1,'U':0,'A':0}}]},unlabeled,rows,train,evaluation,{'e5':curator},{'e5':features},design)
        check('unlabeled_corpus_gets_no_invented_targets',news['task']=='unlabeled' and 'train_y' not in news and 'evaluation_y' not in news and not news['learners'])
        # This branch-routing fixture reuses Civil rows. It is not a News corpus.

        serial=t/'serial';serial.mkdir();desc=d.dump_value(bundle,serial/'candidate.json')
        check('frozen_loader_consumes_generated_descriptors',d.task.fingerprint(d.task.load_bound_value(d.strict_json((serial/'candidate.json').read_text()),serial))==d.task.fingerprint(bundle))
        check('array_artifacts_deduplicated',len(list((serial/'artifacts').glob('*.npy'))) < 6)
        rejects('candidate_without_private_guard_cannot_advance',lambda:d.candidate_root(serial/'candidate.json'))
        (serial/'.gitignore').write_text('*\n');d.write(serial/'receipt.json',{'status':'blocked','candidate':desc})
        rejects('blocked_stage_receipt_cannot_advance',lambda:d.candidate_root(serial/'candidate.json'))
        (serial/'receipt.json').unlink();d.write(serial/'receipt.json',{'status':'unsigned_candidate_ready','candidate':desc})
        check('successful_bound_candidate_stage_can_advance',d.candidate_root(serial/'candidate.json')[1]==serial)
        (serial/'candidate.json').write_text((serial/'candidate.json').read_text()+'\n')
        rejects('changed_candidate_file_cannot_advance',lambda:d.candidate_root(serial/'candidate.json'))
        rejects('unregistered_group_refused',lambda:d.selected_groups(['not-a-registered-group']))
        rejects('multiple_families_not_silently_merged',lambda:d.selected_groups([group['group_id'],'C_full_methods/askubuntu/primary-10000/native']))

    check('all_bound_implementation_sources_unchanged',all(d.file_hash(ROOT/name)==sha for name,sha in codes.items()))
    result={'schema':'ccu-phase8-dossier-checks-1','status':'passed','check_count':len(checks),'checks':checks,
        'source_sha256':codes,'natural_input_sha256':d.file_hash(source),
        'scope':'Existing Civil rows/labels and lexical features. Positive replay/task-choice boundaries are explicit transient software mocks.',
        'actual_original_archive_accepted':False,'actual_semantic_cache_accepted':False,'actual_human_responses_created':0,
        'external_review_created':False,'primary_study_started':False,'private_fixture_outputs_retained':False}
    out.parent.mkdir(parents=True,exist_ok=True);d.write(out,result);return result


if __name__=='__main__':
    path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'phase8/results/dossier_checks.json'
    print(json.dumps(run(path),indent=2))
