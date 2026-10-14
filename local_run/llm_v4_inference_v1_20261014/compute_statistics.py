"""Full-denominator v4 selection after both independent output audits."""
import argparse
from datetime import datetime,timezone
import json
import os
import socket
import common as x
import contract as C
import statistics_core as S

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
    rows=C.read_jsonl(x.SELECTION/'run/selection.private.jsonl',source['files']['selection.private.jsonl'])
    observed=[];audits={};swaps={}
    for alias in pins['models']:
        path=x.ROOT/('audit_'+alias)/'run/result.json';audit=C.strict_json(path.read_bytes());C.verify_seal(audit)
        C.require(audit['passed'] is True and audit['package_sha256']==args.package_sha256 and
                  audit['activation']['sha256']==args.activation_sha256 and
                  audit['request_manifest']==C.binding(x.REQUESTS/'result.json'),'audited_source_identity')
        C.checked_file(x.ROOT/alias/'output/result.json',audit['execution_result'])
        observed.extend(audit['canonical_observations']);audits[alias]=C.binding(path)
        canonical={r['opaque_pair_id']:r for r in audit['canonical_observations']}
        changed=binary=0
        for r in audit['diagnostic_observations']:
            earlier=canonical[r['opaque_pair_id']]
            changed+=int((r['status'],r['category'])!=(earlier['status'],earlier['category']))
            positive=lambda t:t['status']=='valid' and t['category'] in S.POSITIVE
            binary+=int(positive(r)!=positive(earlier))
        swaps[alias]={'pairs':len(audit['diagnostic_observations']),'status_or_category_changes':changed,
                      'binary_positive_changes':binary,'diagnostic_only':True}
    statistics=S.selection(rows,source['populations'],observed)
    out=x.ROOT/'statistics/run';out.mkdir(mode=0o700,exist_ok=False)
    report=x.sealed({'schema':'third-v4-selection-analysis-v1','passed':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),'package_sha256':args.package_sha256,
        'sampling_result':C.binding(x.SELECTION/'run/result.json'),'audits':audits,'statistics':statistics,
        'swap_diagnostics':swaps,'human_annotation_calls':0,'independent_validation_performed':False,
        'semantic_validity_established':False,'confirmatory_study_ready':False,
        'next_stage':'Fresh independent validation of locked threshold' if statistics['selection_passed'] else
                     'Preserve failed v4; no validation or retuning of this configuration'})
    x.publish(out/'result.json',report)
    print(json.dumps({'passed':True,'selection_passed':statistics['selection_passed'],
                      'selected_tau':statistics['selected_tau'],'diagnostics':statistics['diagnostics']}),flush=True)

if __name__=='__main__':main()
