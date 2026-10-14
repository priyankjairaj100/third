"""Actual v4 producer/consumer and full prompt replay on CPU; no activation."""
import argparse
from datetime import datetime,timezone
import json
import os
import resource
import socket
import time
from types import SimpleNamespace
import common as x
import contract as C
import kernel as K

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package-sha256',required=True)
    parser.add_argument('--request-result-sha256',required=True);args=parser.parse_args()
    pins,controls,prior,h,protocol=x.setup(args.package_sha256)
    C.require(socket.gethostname()=='csis.cn1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_worker_only')
    c,result=x.request_contract(args.request_result_sha256,pins)
    watch=K.Watch()
    for folder in (x.HERE,x.PROTOCOL,x.REQUEST_PACKAGE,x.OLD_REQUESTS):
        watch.add(folder/'package_manifest.json')
        for name,b in C.strict_json((folder/'package_manifest.json').read_bytes())['files'].items():watch.add(folder/name,b)
    for name,b in result['files'].items():watch.add(x.REQUESTS/name,b)
    watch.add(x.REQUESTS/'result.json',c['request_manifest'])
    # Recheck the actual source and preparation inputs, not only declarations.
    inputs=C.strict_json((x.REQUESTS/'input_bindings.json').read_bytes())
    for path,b in inputs['files'].items():watch.add(path,b)
    prep=x.module('v4_qualification_request_primitives',x.OLD_REQUESTS/'prepare_requests.py')
    rpin=prep.read(x.OLD_REQUESTS/'expectations.json')
    torch,transformers,runtime=prep.runtime(SimpleNamespace(workspace=x.W),rpin,None,watch)
    system=(x.REPO/'local_run/model_annotation_v2_20261011/judge_system_prompt.txt').read_text()
    task=(x.REPO/'local_run/model_annotation_v2_20261011/judge_task_definition.txt').read_text()
    os.umask(0o077);out=x.QUAL/'run';out.mkdir(exist_ok=False);started=time.perf_counter();reports={}
    for alias,model in pins['models'].items():
        bundle=C.read_request_bundle(x.REQUESTS,c,alias,protocol)
        tokenizer=transformers.AutoTokenizer.from_pretrained(getattr(controls,alias.upper()),
            local_files_only=True,trust_remote_code=False,use_fast=True)
        pilot=prep.load('v4_qualification_'+alias+'_pilot',x.REPO/model['pilot_code_relative'])
        identities={'tokenizer_class':type(tokenizer).__module__+'.'+type(tokenizer).__name__,
            'tokenizer_length':len(tokenizer),'tokenizer_vocab_size':tokenizer.vocab_size,
            'tokenizer_special_ids':tokenizer.all_special_ids,
            'tokenizer_backend_json_sha256':C.sha(tokenizer.backend_tokenizer.to_str().encode()),
            'chat_template_sha256':prep.template_sha256(tokenizer,pilot)}
        C.require(identities==model['tokenizer_identity'],'qualified_tokenizer')
        for row,token in zip(bundle['rows'],bundle['tokens']):
            replay=pilot.tokenize_call({'system_prompt':system,'task_definition':task,'tokenizer':tokenizer},row)
            C.require(replay['encoded']['input_ids'][0].tolist()==token['input_token_ids'] and
                replay['input_tokens']==token['input_tokens'] and
                C.sha(replay['rendered'].encode())==token['rendered_prompt_sha256'],'actual_full_token_replay')
            C.require(replay['encoded']['attention_mask'][0].tolist()==[1]*token['input_tokens'],'mask_replay')
        reports[alias]={'requests':len(bundle['tokens']),'all_full_token_sequences_exact':True,
            'overflow_requests':sum(not t['context_fit_with_output_reserve'] for t in bundle['tokens'])}
        del tokenizer
    watch.recheck();C.require(not torch.cuda.is_initialized(),'no_GPU_context')
    x.publish(out/'input_bindings.json',{'files':watch.files})
    report=x.sealed({'schema':'third-v4-producer-consumer-qualification-v1','passed':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),'package_sha256':args.package_sha256,
        'request_contract':c,'models':reports,'runtime':runtime,'human_annotation_calls':0,
        'model_inference_calls':0,'model_weights_loaded':False,'activation_created':False,
        'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'elapsed_after_preflight_seconds':time.perf_counter()-started})
    x.publish(out/'result.json',report);print(json.dumps({'passed':True,'models':reports}),flush=True)

if __name__=='__main__':main()
