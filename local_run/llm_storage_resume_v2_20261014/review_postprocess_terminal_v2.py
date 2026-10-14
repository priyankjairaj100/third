"""Local independent byte/receipt review and exact selection-statistic recomputation."""
import argparse,json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from review_family_terminal import REPO,WORK,REPORTS,require,read,binding,seal

def root_name(mode):
    return 'llm-only-v3-selection-statistics-a01' if mode=='statistics' else 'llm-only-v3-output-audit-'+mode+'-a01'
def fj(x):return None if x is None else {'numerator':x.numerator,'denominator':x.denominator}

def recompute_statistics(report):
    sample=REPORTS/'llm-selection-sampler-v1-20261014/selection-a01/run/output/selection.private.jsonl'
    rows=[json.loads(s) for s in sample.read_text().splitlines()]
    observations={a:read(REPORTS/root_name(a)/'run/report.json') for a in ('qwen','mistral')}
    panels={a:{v['opaque_pair_id']:v for v in x['canonical_observations']} for a,x in observations.items()}
    require(len(rows)==90 and all(len(v)==90 for v in panels.values()),'complete_selection_panels')
    positive={'exact_copy','substantially_same_meaning'}
    labels={};possible={}
    for row in rows:
        pair=row['opaque_pair_id'];values=[p[pair] for p in panels.values()]
        labels[pair]=int(all(v['status']=='valid' and v['category'] in positive for v in values))
        possible[pair]=int(all(v['status']!='valid' or v['category'] in positive|{'uncertain'} for v in values))
    expected=[];selected=None
    for cut in [Fraction(i,100) for i in range(50,100)]+[Fraction(199,200)]:
        included=[r for r in rows if float.fromhex(r['score_hex'])>float(cut)]
        weights=[Fraction(r['inclusion_weight']['numerator'],r['inclusion_weight']['denominator']) for r in included]
        denominator=sum(weights,Fraction());numerator=sum((weight*labels[r['opaque_pair_id']] for r,weight in zip(included,weights)),Fraction())
        upper=sum((weight*possible[r['opaque_pair_id']] for r,weight in zip(included,weights)),Fraction())
        passes=bool(denominator and numerator*20>=19*denominator)
        if passes and selected is None:selected=cut
        expected.append({'tau':fj(cut),'tau_float_hex':float(cut).hex(),'sample_support':len(included),'weighted_denominator':fj(denominator),'weighted_positive_numerator':fj(numerator),'conservative_precision':fj(numerator/denominator if denominator else None),'missing_uncertain_sensitivity_upper':fj(upper/denominator if denominator else None),'passes_selection_rule':passes})
    result=report['selection_statistics']
    require(result['grid_ledger']==expected and result['selected_tau']==fj(selected) and result['selection_passed'] is (selected is not None),'independent_exact_selection_recompute')
    require(result['failed_diagnostic_tau']==(fj(Fraction(199,200)) if selected is None else None),'failed_diagnostic_cut')
    models={a:next(iter(p.values()))['model_id'] for a,p in panels.items()}
    diagnostic={'pairs':90,'status_counts':{models[a]:dict(Counter(v['status'] for v in p.values())) for a,p in panels.items()},'category_counts':{models[a]:dict(Counter(v['category'] for v in p.values() if v['status']=='valid')) for a,p in panels.items()},'valid_six_way_disagreements':sum(all(p[pair]['status']=='valid' for p in panels.values()) and len({p[pair]['category'] for p in panels.values()})>1 for pair in labels),'conservative_positives':sum(labels.values())}
    require(result['diagnostics']==diagnostic,'independent_panel_diagnostics')
    for alias,source in observations.items():
        diags=source['diagnostic_observations'];canonical=panels[alias]
        changes=sum((canonical[v['opaque_pair_id']]['status'],canonical[v['opaque_pair_id']]['category'])!=(v['status'],v['category']) for v in diags)
        binary=lambda v:v['status']=='valid' and v['category'] in positive
        flips=sum(binary(canonical[v['opaque_pair_id']])!=binary(v) for v in diags)
        require(report['swap_diagnostics'][alias]=={'pairs':9,'status_or_category_changes':changes,'binary_positive_changes':flips,'used_in_threshold_estimator':False},'independent_swap_diagnostics')
    return {'all51_exact_weighted_cuts_recomputed':True,'selected_tau':fj(selected),'selection_passed':selected is not None,'conservative_positives_of90':sum(labels.values()),'independent_validation_performed':False}

def review(mode, launch_receipt=None):
    root=REPORTS/root_name(mode);export=read(root/'terminal_export.json');job=export['job_id']
    require(not export['queue'].strip(),'active_audit_job')
    rows=[s.split('|') for s in export['accounting'].splitlines() if s]
    parent=[r for r in rows if r[0]==job];require(len(parent)==1 and parent[0][1:3]==['COMPLETED','0:0'],'audit_terminal_success')
    require(parent[0][5] in ('csis.cn1','cn1'),'CPU_audit_node')
    require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(export['files'])|{'terminal_export.json'},'exact_export_inventory')
    for name,b in export['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'safe_export_name')
        require(not (root/name).is_symlink() and binding(root/name)=={k:b[k] for k in ('bytes','sha256')},'actual_export_bytes')
    launch=read(root/'launch_receipt.json');intent=read(root/'launch_intent.json');proc=read(root/'run/process_completion.json');wrapped=read(root/'run/postprocess_result.json');software=read(root/'run/software_validation.json');report=read(root/'run/report.json');seal(report)
    require(launch==read(launch_receipt or WORK/('postprocess_'+mode+'_launch_receipt.json')),'local_launch_receipt')
    require(all(launch.get(k)==v for k,v in intent.items()) and launch['job_id']==job and launch['returncode']==0,'intent_submission')
    require(proc['schema']=='third-llm-postprocess-process-v1' and proc['command']==launch['child_command'] and proc['slurm_job_id']==job and proc['mode']==mode,'exact_audit_child_identity')
    require(proc['returncode']==0 and proc['console_cap_exceeded'] is False and proc['retry_attempts']==proc['model_inference_calls']==0,'audit_process_success')
    require(software['tests']==12 and software['passed'] is True and software['failures']==software['errors']==software['skipped']==0,'actual12tests')
    package=binding(REPO/'local_run/llm_annotation_postprocess_v1_20261014/package_manifest.json')
    require(wrapped['package']==proc['package']==package and wrapped['bindings']==launch['bindings'],'audit_bound_package_inputs')
    require(wrapped['schema']=='third-llm-postprocess-result-v1' and wrapped['passed'] is True and wrapped['mode']==mode and wrapped['execution_eligible'] is True,'audit_wrapper_accepted')
    require(wrapped['report']==binding(root/'run/report.json') and wrapped['integrity_error'] is None and wrapped['source_and_result_bytes_initial_final_equal'] is True,'audited_report_binding')
    require(report['passed'] is True and report['human_annotation_calls']==0 and report['semantic_validity_established'] is False,'report_scope')
    active=binding(REPORTS/'llm-only-v3-annotation-activation-a01/activation.json');resource=read(WORK/'inference_resource_plan.json')
    require(report['activation']==active and report['request_manifest']==resource['request_manifest'] and report['integration_package']==resource['packages']['integration'],'report_actual_admission')
    extra={}
    if mode!='statistics':
        family=REPORTS/('llm-only-v3-selection-'+mode+'-a01/run/output');result=read(family/'result.json');responses=read(family/'responses.json')
        require(report['schema']=='third-llm-output-independent-audit-v1' and report['alias']==mode and report['execution_result']==binding(family/'result.json'),'audited_family_identity')
        require(report['planned_calls']==99 and report['canonical_calls']==90 and report['diagnostic_calls']==9 and report['status_counts']==result['status_counts'],'audited_counts')
        require(report['all_call_identities_denominators_and_tokens_verified'] is True,'full_token_and_output_replay')
        for orientation,key in [('canonical','canonical_observations'),('swapped_diagnostic','diagnostic_observations')]:
            observations=[{'opaque_pair_id':v['opaque_pair_id'],'model_id':v['model_id'],'status':v['status'],'category':v['judgment']['category'] if v['status']=='valid' else None} for v in responses if v['orientation']==orientation]
            require(observations==report[key],'raw_responses_match_auditor')
        extra={'status_counts':report['status_counts'],'execution_result':report['execution_result'],'all_actual_output_replays_passed':True}
    else:
        require(report['schema']=='third-llm-selection-ingestion-v1' and report['statistics_package']==resource['packages']['statistics'],'statistics_identity')
        require(report['complete_canonical_pair_denominator']==90 and report['canonical_model_observations']==180 and report['missing_outputs_excluded'] is False,'statistics_full_denominator')
        require(report['model_audits']=={a:binding(REPORTS/root_name(a)/'run/report.json') for a in ('qwen','mistral')},'actual_audit_pair')
        extra=recompute_statistics(report)
    return {'schema':'third-llm-postprocess-local-terminal-review-v1','passed':True,'mode':mode,'job_id':job,'state':'COMPLETED','exit_code':'0:0','scheduler_elapsed_seconds':int(parent[0][3]),'transport':binding(root/'terminal_export.json'),'transport_files':len(export['files']),'transport_bytes':export['total_bytes'],'report':binding(root/'run/report.json'),'process_receipt':binding(root/'run/process_completion.json'),'review_source':binding(Path(__file__)),'helper_source':binding(WORK/'review_family_terminal.py'),'actual_software_tests':12,'activation':active,'all_exported_bytes_verified':True,'human_annotation_calls':0,'new_model_calls':0,'semantic_validity_established':False,**extra}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qwen','mistral','statistics'));p.add_argument('--output',type=Path,required=True);p.add_argument('--launch-receipt',type=Path);a=p.parse_args()
    result=review(a.mode,a.launch_receipt)
    if a.launch_receipt is not None:result['explicit_launch_receipt']={'path':str(a.launch_receipt),'binding':binding(a.launch_receipt)}
    with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','mode','job_id','selection_passed','selected_tau','conservative_positives_of90','status_counts')}))
