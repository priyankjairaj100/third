"""Fresh v4 validation: exact finite-population bound and overlap stability."""
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import json
import os
import socket
import common as x
import contract as C
import statistics_core as S

SELECTION=x.W/'qualification/llm-only-v4-selection-a01/run'
PRIOR=x.W/'qualification/llm-only-v4-inference-a01'
PRIOR_PACKAGE_SHA='ef7449595cb5a90e8fc8a009103f958eb92a73abc7943d52c19a552e816c4207'

def main():
    p=argparse.ArgumentParser()
    for name in ('package-sha256','request-result-sha256','activation-sha256','resource-plan-sha256'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args();pins,controls,prior,h,protocol=x.setup(args.package_sha256)
    C.require(socket.gethostname()=='csis.cn1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_statistics')
    _,requests=x.request_contract(args.request_result_sha256,pins)
    C.require(C.binding(x.ROOT/'activation/activation.json')['sha256']==args.activation_sha256 and
              C.binding(x.ROOT/'resource_plan.json')['sha256']==args.resource_plan_sha256,'activation_resources')
    source=C.strict_json((x.SELECTION/'run/result.json').read_bytes());C.verify_seal(source)
    rows=C.read_jsonl(x.SELECTION/'run/validation.private.jsonl',source['files']['validation.private.jsonl'])
    C.require(source['selected_pairs']==len(rows)==min(200,source['population']) and source['population']>0,
              'complete_nonempty_validation')
    for row in rows:
        C.require(row['validation_population']==source['population'] and row['validation_sample_size']==len(rows) and
            S.rational(row['inclusion_weight'])==Fraction(source['population'],len(rows)) and
            S.rational(row['inclusion_probability'])==Fraction(len(rows),source['population']),'exact_validation_design')
    selected=C.strict_json(C.checked_file(PRIOR/'statistics/run/result.json',source['selection_statistics']).read_bytes())
    C.verify_seal(selected)
    C.require(selected['passed'] is True and selected['package_sha256']==PRIOR_PACKAGE_SHA and
        selected['statistics']['selection_passed'] is True and
        selected['statistics']['selected_tau']==source['locked_tau'],'locked_positive_selection')
    sample_report=C.strict_json(C.checked_file(SELECTION/'result.json',selected['sampling_result']).read_bytes())
    C.verify_seal(sample_report)
    selection_rows=C.read_jsonl(SELECTION/'selection.private.jsonl',sample_report['files']['selection.private.jsonl'])
    observed=[];earlier=[];audits={};swaps={}
    for alias in pins['models']:
        old_path=PRIOR/('audit_'+alias)/'run/result.json'
        old=C.strict_json(C.checked_file(old_path,selected['audits'][alias]).read_bytes());C.verify_seal(old)
        C.require(old['passed'] is True and old['package_sha256']==PRIOR_PACKAGE_SHA,'selection_audit_passed')
        earlier.extend(old['canonical_observations'])
        path=x.ROOT/('audit_'+alias)/'run/result.json';audit=C.strict_json(path.read_bytes());C.verify_seal(audit)
        C.require(audit['passed'] is True and audit['package_sha256']==args.package_sha256 and
            audit['activation']['sha256']==args.activation_sha256 and
            audit['request_manifest']==C.binding(x.REQUESTS/'result.json'),'fresh_audited_validation')
        C.require(audit['activation']!=old['activation'] and audit['request_manifest']!=old['request_manifest'],
                  'fresh_calls_and_activation_required')
        C.checked_file(x.ROOT/alias/'output/result.json',audit['execution_result'])
        observed.extend(audit['canonical_observations']);audits[alias]=C.binding(path)
        canonical={r['opaque_pair_id']:r for r in audit['canonical_observations']}
        changed=binary=0
        for row in audit['diagnostic_observations']:
            base=canonical[row['opaque_pair_id']]
            changed+=int((row['status'],row['category'])!=(base['status'],base['category']))
            positive=lambda r:r['status']=='valid' and r['category'] in S.POSITIVE
            binary+=int(positive(row)!=positive(base))
        swaps[alias]={'pairs':len(audit['diagnostic_observations']),'status_or_category_changes':changed,
                      'binary_positive_changes':binary,'diagnostic_only':True}
    statistics=S.validation(rows,source['population'],observed,selection_rows,sample_report['populations'],
                            earlier,selected['statistics'],S.frozen_bound(x.REPO))
    C.require(statistics['overlap_pairs']==source['selection_overlap_pairs'],'overlap_population_replay')
    out=x.ROOT/'statistics/run';out.mkdir(mode=0o700,exist_ok=False)
    report=x.sealed({'schema':'third-v4-independent-validation-analysis-v1','passed':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),'package_sha256':args.package_sha256,
        'validation_sample':C.binding(x.SELECTION/'run/result.json'),'selection_statistics':source['selection_statistics'],
        'audits':audits,'statistics':statistics,'swap_diagnostics':swaps,'human_annotation_calls':0,
        'independent_validation_performed':True,'semantic_validity_established':False,'confirmatory_study_ready':False,
        'next_stage':'Versioned model-defined development effect study only, after remaining task and resource checks'
            if statistics['model_validation_passed'] else 'Preserve failure; stop this configuration without retuning'})
    x.publish(out/'result.json',report)
    print(json.dumps({'passed':True,'model_validation_passed':statistics['model_validation_passed'],
        'lower_bound':statistics['finite_population_bound'],'overlap_changes':len(statistics['changed_overlap_observations'])}),flush=True)

if __name__=='__main__':main()
