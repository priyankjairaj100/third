"""Join exactly two independently audited canonical model panels into frozen statistics."""
import argparse
from collections import Counter
from pathlib import Path
from common import Watch,binding,jsonl,load,new_json,package,pins,read,require,seal,sealed
from activation import read_activation

def observed_from_responses(responses,diagnostic=False):
    wanted='swapped_diagnostic' if diagnostic else 'canonical';out=[];seen=set()
    for r in responses:
        if r['orientation']!=wanted:continue
        require(r['stage']==('selection_swap' if diagnostic else 'selection') and r['attempt']==0,'ingestion_phase_attempt')
        key=(r['opaque_pair_id'],r['model_id']);require(key not in seen,'duplicate_observation');seen.add(key)
        out.append({'opaque_pair_id':r['opaque_pair_id'],'model_id':r['model_id'],'status':r['status'],
                    'category':r['judgment']['category'] if r['status']=='valid' else None})
    return out

def validate_panel_reports(reports,models,request_binding,activation_binding,counts):
    require(set(reports)==set(models),'exact_two_model_reports')
    seen_pairs=None
    for alias,report in reports.items():
        require(report['schema']=='third-llm-output-independent-audit-v1' and report['passed'] is True and report['alias']==alias,'audit_status_alias')
        require(report['all_call_identities_denominators_and_tokens_verified'] is True and report['request_manifest']==request_binding and report['activation']==activation_binding,'audit_request_activation')
        require(report['planned_calls']==counts['per_model'] and report['canonical_calls']==counts['canonical_pairs'] and report['diagnostic_calls']==counts['swap_pairs'],'audit_counts')
        canonical=report['canonical_observations'];swaps=report['diagnostic_observations']
        require(len(canonical)==counts['canonical_pairs'] and len(swaps)==counts['swap_pairs'],'panel_lengths')
        pairs=[r['opaque_pair_id'] for r in canonical];require(len(set(pairs))==len(pairs),'duplicate_canonical_pair')
        require(all(r['model_id']==models[alias]['model_id'] for r in canonical+swaps),'wrong_model_report')
        require(len({r['opaque_pair_id'] for r in swaps})==len(swaps) and {r['opaque_pair_id'] for r in swaps}<=set(pairs),'swap_pair_subset')
        require(seen_pairs is None or pairs==seen_pairs,'model_pair_order_or_coverage');seen_pairs=pairs
    return seen_pairs

def selection_statistics(rows,populations,reports,models,request_binding,activation_binding,counts,core):
    pairs=validate_panel_reports(reports,models,request_binding,activation_binding,counts)
    require([r['opaque_pair_id'] for r in rows]==pairs,'sample_canonical_identity')
    observed=[x for a in models for x in reports[a]['canonical_observations']]
    result=core.selection(rows,populations,observed)
    swaps={};positive={'exact_copy','substantially_same_meaning'}
    for alias,report in reports.items():
        canonical={r['opaque_pair_id']:r for r in report['canonical_observations']};diags=report['diagnostic_observations']
        changes=sum((canonical[r['opaque_pair_id']]['status'],canonical[r['opaque_pair_id']]['category'])!=(r['status'],r['category']) for r in diags)
        binary_changes=sum((canonical[r['opaque_pair_id']]['status']=='valid' and canonical[r['opaque_pair_id']]['category'] in positive)!=(r['status']=='valid' and r['category'] in positive) for r in diags)
        swaps[alias]={'pairs':len(diags),'status_or_category_changes':changes,'binary_positive_changes':binary_changes,'used_in_threshold_estimator':False}
    return result,swaps

def ingest(admission_path,admission_binding,activation_dir,activation_binding,report_paths,report_bindings,run_roots):
    activation,state=read_activation(admission_path,admission_binding,activation_dir,activation_binding)
    w=state['watch'];resource=state['resource_plan'];paths=state['admission']['paths'];cp=pins()['component_expectations'];reports={}
    require(set(report_paths)==set(report_bindings)==set(run_roots)=={'qwen','mistral'},'two_report_sources')
    for alias,path in report_paths.items():
        w.file(path,report_bindings[alias],4*1024**2);report=read(path);seal(report)
        require(report['integration_package']==resource['packages']['integration'] and report['runner_package']==resource['packages']['runner'] and report['resource_plan']==activation.contract['resource_plan'],'audit_package_bindings')
        root=Path(run_roots[alias]);w.file(root/'result.json',report['execution_result'],4*1024**2);execution=read(root/'result.json');seal(execution)
        require(execution['passed_execution_integrity'] is True and execution['activation']==activation_binding and execution['request_manifest']==activation.contract['request_manifest'],'audited_execution')
        w.file(root/'responses.json',execution['files']['responses.json']);responses=read(root/'responses.json')
        require(observed_from_responses(responses)==report['canonical_observations'] and observed_from_responses(responses,True)==report['diagnostic_observations'],'audited_observations_match_raw_responses')
        reports[alias]=report
    request_dir=Path(paths['request_dir']);rows=jsonl(request_dir/'sample.private.jsonl');selection=read(paths['selection_result'])
    source_sample=Path(paths['selection_result']).parent/'selection.private.jsonl';w.file(source_sample,selection['files']['selection.private.jsonl'])
    require(rows==jsonl(source_sample),'original_sealed_selection_sample')
    core=load('third_bound_statistics_core',Path(paths['statistics_package'])/'statistics_core.py')
    result,swaps=selection_statistics(rows,selection['bin_populations'],reports,cp['models'],activation.contract['request_manifest'],activation_binding,resource['planned'],core)
    w.recheck()
    return sealed({'schema':'third-llm-selection-ingestion-v1','passed':True,'activation':activation_binding,
      'request_manifest':activation.contract['request_manifest'],'selection_result':activation.contract['selection_result'],
      'model_audits':report_bindings,'integration_package':resource['packages']['integration'],'statistics_package':resource['packages']['statistics'],
      'complete_canonical_pair_denominator':len(rows),'canonical_model_observations':2*len(rows),'missing_outputs_excluded':False,
      'selection_statistics':result,'swap_diagnostics':swaps,'model_defined_target_only':True,'semantic_validity_established':False,
      'human_annotation_calls':0,'validation_performed':False,'confirmatory_study_ready':False})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('admission','activation-dir','output','qwen-audit','mistral-audit','qwen-run','mistral-run'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('admission-sha256','activation-sha256','qwen-audit-sha256','mistral-audit-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();ab=binding(a.admission);act=binding(a.activation_dir/'activation.json');require(ab['sha256']==a.admission_sha256 and act['sha256']==a.activation_sha256,'external_ingestion_pins')
    reports={k:getattr(a,k+'_audit') for k in ('qwen','mistral')};rb={k:binding(v) for k,v in reports.items()}
    require(all(v['sha256']==getattr(a,k+'_audit_sha256') for k,v in rb.items()),'external_audit_report_pins')
    result=ingest(a.admission,ab,a.activation_dir,act,reports,rb,{k:getattr(a,k+'_run') for k in reports});new_json(a.output,result)
    print(__import__('json').dumps({'passed':True,'result':binding(a.output),'selection_passed':result['selection_statistics']['selection_passed']}))
if __name__=='__main__':main()
