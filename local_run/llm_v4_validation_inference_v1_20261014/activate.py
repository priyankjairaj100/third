"""Metadata-only v4 activation after actual CPU token qualification."""
import argparse
from datetime import datetime,timezone
import json
import os
import common as x
import contract as C
from run_model import expected_contract

def main():
    parser=argparse.ArgumentParser()
    for name in ('package-sha256','request-result-sha256','qualification-result-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();pins,controls,prior,h,protocol=x.setup(args.package_sha256)
    C.require(not os.path.lexists(x.ROOT),'occupied_activation_root_reconcile')
    c,requests=x.request_contract(args.request_result_sha256,pins)
    qb=C.binding(x.QUAL/'run/result.json');C.require(qb['sha256']==args.qualification_result_sha256,'qualification_pin')
    qualified=C.strict_json((x.QUAL/'run/result.json').read_bytes());C.verify_seal(qualified)
    C.require(qualified['passed'] is True and qualified['request_contract']==c and
              qualified['package_sha256']==args.package_sha256,'qualification_identity')
    process=h.load(x.QUAL/'process_completion.json')
    C.require(process['returncode']==0 and not process['console_cap_exceeded'],'qualification_process')
    job=h.load(x.QUAL/'launch_receipt.json')['job_id']
    accounting=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(accounting,job)
    C.require(requests['planned_requests']==2*sum((requests['selected_pairs'],requests['swap_pairs'])),'planned_denominator')
    for alias in pins['models']:
        C.require(qualified['models'][alias]['requests']*2==requests['planned_requests'] and
                  qualified['models'][alias]['all_full_token_sequences_exact'] is True,'actual_token_replay')
    os.umask(0o077);x.ROOT.mkdir(mode=0o700,exist_ok=False)
    resources=x.sealed({'schema':'third-v4-inference-resource-plan-v1','package_sha256':args.package_sha256,
        'created_utc':datetime.now(timezone.utc).isoformat(),'qualification':qb,
        'qualification_accounting':accounting,'request_manifest':c['request_manifest'],
        'canonical_pairs':requests['selected_pairs'],'swap_pairs':requests['swap_pairs'],
        'per_model_requests':requests['planned_requests']//2,'total_requests':requests['planned_requests'],
        'families':{alias:{'cpus':4,'host_memory_bytes':(48 if alias=='qwen' else 64)*1024**3,
            'gpu_count':1,'gpu_model':'A10080GB','node':'csis.cn3','slurm_time_seconds':6*3600,
            'output_dir':str(x.ROOT/alias/'output'),'output_limit_bytes':256*1024**2,
            'console_limit_bytes':16*1024**2} for alias in pins['models']},
        'quota_reserve_bytes':8*1024**3,'per_stage_allowance_bytes':288*1024**2,
        'no_retries':True,'human_annotation_calls':0,'stop_on_runtime_error':True})
    rbind=x.publish(x.ROOT/'resource_plan.json',resources)
    c,_=expected_contract(args.request_result_sha256,pins,rbind)
    activation=x.ROOT/'activation';activation.mkdir(mode=0o700,exist_ok=False)
    intent=x.publish(activation/'intent.json',{'schema':'third-llm-only-validation-activation-intent-v4',
        'contract_sha256':C.sha(C.canonical(c)),'entropy_draws':0,'attempt':0})
    value=x.sealed({'schema':'third-llm-only-validation-activation-v4','status':'activated','contract':c,'intent':intent})
    abind=x.publish(activation/'activation.json',value)
    C.validate_activation(activation,abind,c)
    print(json.dumps({'activated':True,'activation':abind,'resource_plan':rbind,
                      'total_requests':resources['total_requests'],'model_calls_started':0}),flush=True)

if __name__=='__main__':main()
