#!/usr/bin/env python3
"""Check registry coverage and fail-closed artifact inventory; no generated data."""
from pathlib import Path
import argparse,copy,gzip,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase5.study_registry import build_registry,preflight,export,digest,sha,write,CORE,LABELED

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,
        default=ROOT/'phase5/results/study_registry');args=parser.parse_args()
    registry,jobs=build_registry();checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    check('unique_job_ids',len({j['job_id'] for j in jobs})==len(jobs))
    check('job_content_digest',registry['jobs_content_sha256']==digest(jobs))
    check('no_execution_implicitly_enabled',all(j['execution_allowed'] is False for j in jobs))
    check('preregistration_not_fabricated',registry['preregistration_submitted'] is False)
    for block,expected in [('A_structure',12),('B_relevance',2),('C_full_methods',2),('D_replication',5),
                           ('E_full_refit',2),('F_convex',2),('G_boundary',18)]:
        check(block+'_group_count',sum(g['block']==block for g in registry['groups'])==expected)
    for corpus in LABELED:
        relevance=[j for j in jobs if j.get('group_id')==f'B_relevance/{corpus}/primary-10000/native' and j.get('arm') in ('R','S','U','A')]
        check(corpus+'_B_unclipped_releases_2688',sum(len(j['planned_checkpoint_units']) for j in relevance)==2688)
        core=[j for j in jobs if j.get('group_id')==f'C_full_methods/{corpus}/primary-10000/native' and j.get('arm') in ('R','S','U','A') and j.get('kind')!='timing_repeat']
        check(corpus+'_C_unclipped_releases_704',sum(len(j['planned_checkpoint_units']) for j in core)==704)
        check(corpus+'_C_subset_parent_links',all('subset_of' in j for j in core))
        repeats=[j for j in jobs if j.get('kind')=='timing_repeat' and j.get('group_id')==f'C_full_methods/{corpus}/primary-10000/native']
        check(corpus+'_80_timing_repetitions',len(repeats)==80)
        refit=[j for j in jobs if j.get('group_id')==f'E_full_refit/{corpus}/separate-refit-5000/native']
        check(corpus+'_four_all_step_refit_paths',sum(j.get('additional_oracle_every_integer_service_step',False) for j in refit)==4)
        check(corpus+'_refit_five_same_seed',sum(j.get('kind')=='same_seed_no_deletion' for j in refit)==5)
        check(corpus+'_refit_three_alternative_seeds',sum(j.get('kind')=='alternative_seed_no_deletion' for j in refit)==3)
    check('symbolic_actual_counts_preserved',all(j.get('actual_horizon') is None for j in jobs if 'actual_horizon' in j))
    check('wcep_source_condition_preserved',all(g['source_condition']=='verified_original_urls_and_native_sources' for g in registry['groups'] if g['corpus']=='wcep100'))
    wcep=next(g for g in registry['groups'] if g['corpus']=='wcep100')
    check('wcep_inherits_news_calibration','cc_news.e5.selection_lock' in wcep['required_artifact_keys'] and 'wcep100.e5.selection_lock' not in wcep['required_artifact_keys'])
    check('wcep_no_new_split_guard','wcep100.e5.semantic_guard' not in wcep['required_artifact_keys'])
    check('wcep_URL_requirement_only_source_arm','wcep100.original_URL_verification' not in wcep['required_artifact_keys'] and wcep['conditional_artifact_keys']['S']==['wcep100.original_URL_verification'])
    thresholds={g['configuration']['threshold'] for g in registry['groups'] if g['variant'].startswith('threshold-')}
    check('threshold_sensitivities_clipped_as_narrative',thresholds=={'max(0,calibrated-0.02)','min(1,calibrated+0.02)'})
    check('unresolved_obligations_retained',len([j for j in jobs if j.get('kind')=='unresolved_protocol_obligation'])==len(registry['unresolved_obligations']))
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);artifact=p/'actual-file.txt';artifact.write_text('Artifact hashing software check, not corpus data or ratings.\n')
        key=registry['required_artifact_keys'][0]
        inv={'artifacts':{key:{'path':artifact.name,'sha256':sha(artifact)}},'capabilities':{'primary_ready':True,'human_ratings':100000}}
        passed=preflight(registry,inv,p)
        check('actual_file_hash_verified',passed['byte_verified_count']==1)
        check('declared_humans_never_count_as_verified',all(g['actual_human_ratings_verified']==0 for g in passed['group_status']))
        check('capability_declaration_cannot_enable_execution',passed['execution_allowed'] is False)
        check('no_jobs_disappear_on_missing_inputs',passed['planned_jobs_retained']==len(jobs))
        bad=copy.deepcopy(inv);bad['artifacts'][key]['sha256']='0'*64
        check('tampered_hash_rejected',preflight(registry,bad,p)['byte_verified_count']==0)
        bad=copy.deepcopy(inv);bad['artifacts'][key]['path']='../outside.txt'
        check('path_escape_rejected',preflight(registry,bad,p)['byte_verified_count']==0)
        bad=copy.deepcopy(inv);bad['artifacts'][key]['sha256']=None
        check('missing_expected_hash_rejected',preflight(registry,bad,p)['byte_verified_count']==0)
    saved,report=export(args.output_dir)
    expanded=[json.loads(line) for line in gzip.decompress((args.output_dir/'jobs.jsonl.gz').read_bytes()).splitlines()]
    check('compressed_jobs_roundtrip',expanded==jobs)
    check('empty_real_inventory_report_honest',report['byte_verified_count']==0)
    audit=dict(status='passed',check_count=len(checks),checks=checks,groups=len(saved['groups']),
        planned_jobs=len(jobs),required_artifacts=report['artifact_count'],actual_assets_verified=0,
        primary_study_started=False,execution_allowed=False,registry_code_sha256=sha(ROOT/'phase5/study_registry.py'),
        check_code_sha256=sha(Path(__file__)))
    write(args.output_dir/'registry_checks.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
