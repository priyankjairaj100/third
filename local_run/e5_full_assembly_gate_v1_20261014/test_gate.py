"""Synthetic completion metadata only; no payload read, model, scheduler or SSH."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import gate
import verify_local

REPO = Path('/Users/priyankjairaj/Downloads/ACL/Three')
TASK = Path('/Users/priyankjairaj/Documents/Codex/2026-10-11/see-i-am-working-on-a')


def item(raw):
    return {**gate.binding(raw),'base64':base64.b64encode(raw).decode(),
        'stat':{'dev':1,'ino':1,'size':len(raw),'mtime_ns':1,'ctime_ns':1}}


def put(bundle,path,value):
    bundle['metadata'][path]=item(value if isinstance(value,bytes) else gate.canonical(value)+b'\n')


def get(bundle,path):return gate.strict_json(base64.b64decode(bundle['metadata'][path]['base64']))
def bound(bundle,path):return {k:bundle['metadata'][path][k] for k in ('bytes','sha256')}


def fixture():
    pins=gate.expectations();names=gate.paths(pins);root=pins['root'];remote_repo=pins['repository']
    local_plan=REPO/'local_workspace/reports/e5-full-sharded-v2-20261013/a01-plan'
    local_qual=REPO/'local_workspace/reports/e5-sharded-cpu-v2-20261013/a01/qualification'
    def local(path):
        if path.startswith(remote_repo+'/'):return REPO/path[len(remote_repo)+1:]
        if path.startswith(root+'/'):return local_plan/path[len(root)+1:]
        if path==names['qualification']:return local_qual/'result.json'
        if path==names['repair']:return REPO/'local_workspace/reports/e5-scipy-import-repair-v2-20261011/a01/repair/result.json'
        raise ValueError(path)
    bundle={'schema':gate.BUNDLE_SCHEMA,'root':root,'array_job_id':'13498','code_bindings':gate.code_bindings(),
        'metadata':{},'payload_stats':{},'log_stats':{},'inventories':{},
        'scheduler':{'observed_utc':'SYNTHETIC','queue_stdout':'','accounting_stdout':''.join(f'13498_{i}|{20000+i}|COMPLETED|0:0|100||csis.cn1|\n' for i in range(32))}}
    for path in pins['pinned_files']:
        p=local(path)
        if path==names['repair']:p=REPO/'local_workspace/reports/e5-scipy-import-repair-v2-20261011/a01/repair/result.json'
        put(bundle,path,p.read_bytes())
    for name in ('plan-run/launch.json','plan-run/dispatch_result.json','plan-run/plan.process.json','plan-run/output/result.json'):
        put(bundle,root+'/'+name,(local_plan/name).read_bytes())
    launch=json.loads((TASK/'work/resume-20261013/v3_array_launch_transport.json').read_text())
    put(bundle,root+'/array_launch_receipt.json',launch)
    put(bundle,root+'/array_launch_intent.json',{k:v for k,v in launch.items() if k not in ('returncode','stdout','stderr','job_id')})
    for key in ('core','dispatch'):
        for name in get(bundle,names[key])['files']:
            path=str(Path(names[key]).parent/name);put(bundle,path,local(path).read_bytes())
    plan=get(bundle,names['plan']);budget=get(bundle,names['budget']);qualification=get(bundle,names['qualification']);prior=get(bundle,root+'/plan_launch_receipt.json')
    bindings={'core_package':bound(bundle,names['core']),'dispatch_package':bound(bundle,names['dispatch']),'qualification_result':bound(bundle,names['qualification']),'full_plan':bound(bundle,names['plan'])}
    common=[v for k,value in zip(('repo','workspace','view','replay-result','census-dir','budget-dir'),prior['common_arguments'][:6]) for v in ('--'+k,value)]
    common+=['--package-sha256',bindings['core_package']['sha256'],'--qualification-result',names['qualification'],'--qualification-sha256',bindings['qualification_result']['sha256']]
    shardroot=root+'/shards';bundle['inventories'][shardroot]=[f'shard-{i:04d}' for i in range(32)]
    for i,shard in enumerate(plan['shards']):
        folder=shardroot+f'/shard-{i:04d}';counts=budget['per_shard_payload'][i]
        bundle['inventories'][folder]=sorted(['launch.json','dispatch_result.json','encode.process.json','replay.process.json','encode.stdout.txt','encode.stderr.txt','replay.stdout.txt','replay.stderr.txt','encode','replay'])
        common_receipt={'mode':'shard','shard_index':i,'bindings':bindings,**{k:False for k in gate.DISPATCH_FALSE_FLAGS}}
        put(bundle,folder+'/launch.json',{**common_receipt,'hostname':'csis.cn1','slurm_job_id':str(20000+i)})
        children=[]
        for j,mode in enumerate(('encode','replay')):
            directory=folder+'/'+mode;bundle['inventories'][directory]=['chunk_log.jsonl','completion.json','vectors.npy']
            files={'vectors.npy':{'bytes':counts['vectors_npy_bytes'],'sha256':'a'*64},'chunk_log.jsonl':{'bytes':counts['chunk_log_bytes'],'sha256':'b'*64}}
            completion={'schema':'third-e5-cpu-shard-completion-v1','kind':mode,'passed':True,'shard_index':i,'start':shard['start'],'stop':shard['stop'],'rows':counts['rows'],
                'content_tokens':counts['content_tokens'],'chunks':counts['chunks'],'model_forward_batches':counts['batches'],'plan_sha256':plan['sha256'],'fingerprint':pins['fingerprint'],
                'qualification_result':bindings['qualification_result'],'runtime':qualification['runtime'],'numerical_contract':gate.CONTRACT,'files':files,**{k:False for k in gate.FALSE_FLAGS}}
            if mode=='replay':completion.update(encoded_completion=bound(bundle,folder+'/encode/completion.json'),exact_vector_bytes_equal=True,complete_chunk_log_equal=True)
            put(bundle,directory+'/completion.json',completion)
            cmd=[pins['workspace']+'/envs/encoder-cpu/bin/python',pins['repository']+'/local_run/e5_sharded_cpu_v2_20261013/worker.py',*common,'--plan',names['plan'],'--shard-index',str(i),'--mode',mode,'--out',directory]
            if mode=='replay':cmd+=['--encoded-dir',folder+'/encode']
            process={'schema':'third-e5-sharded-dispatch-child-v1','returncode':0,'command':cmd,'child_pid':30000+i*2+j,'launcher_pid':40000+i,
                'child_wall_seconds_through_exit':1.0,'timer_includes_child_artifact_publication_and_exit':True,**{k:False for k in gate.DISPATCH_FALSE_FLAGS}}
            put(bundle,folder+f'/{mode}.process.json',process)
            children.append({'name':mode,'artifact':bound(bundle,directory+'/completion.json'),'process_receipt':bound(bundle,folder+f'/{mode}.process.json'),'child_wall_seconds_through_exit':1.0})
            for name,value in files.items():bundle['payload_stats'][directory+'/'+name]={'dev':1,'ino':1,'size':value['bytes'],'mtime_ns':1,'ctime_ns':1}
            for channel in ('stdout','stderr'):bundle['log_stats'][folder+f'/{mode}.{channel}.txt']={'dev':1,'ino':1,'size':0,'mtime_ns':1,'ctime_ns':1}
        put(bundle,folder+'/dispatch_result.json',{**common_receipt,'passed':True,'children':children})
    return bundle


def rebind(bundle,path,value):
    put(bundle,path,value)
    folder=path.split('/shards/')[0]+'/shards/'+path.split('/shards/')[1].split('/')[0]
    dispatch=get(bundle,folder+'/dispatch_result.json')
    for child in dispatch['children']:
        if path==folder+'/'+child['name']+'/completion.json':child['artifact']=bound(bundle,path)
        if path==folder+'/'+child['name']+'.process.json':child['process_receipt']=bound(bundle,path)
    put(bundle,folder+'/dispatch_result.json',dispatch)


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixture();cls.root=gate.expectations()['root'];cls.folder=cls.root+'/shards/shard-0000'
    def test_valid_synthetic_all32_metadata(self):
        result=gate.verify_bundle(self.base);self.assertEqual(result['completion_receipt_count'],64);self.assertFalse(result['heavy_vector_or_log_hashes_verified']);self.assertFalse(result['full_population_cache_acceptance'])
    def test_scheduler_duplicate_missing_wrongnode_failed_and_queued(self):
        original=self.base['scheduler']
        cases=[]
        for old,new in [('13498_1|','13498_0|'),('csis.cn1','csis.cn2'),('COMPLETED','FAILED')]:
            value=copy.deepcopy(original);value['accounting_stdout']=value['accounting_stdout'].replace(old,new);cases.append(value)
        value=copy.deepcopy(original);value['accounting_stdout']='\n'.join(value['accounting_stdout'].splitlines()[:-1]);cases.append(value)
        value=copy.deepcopy(original);value['queue_stdout']='13498_[0-31%8]|PENDING|0|Priority';cases.append(value)
        for value in cases:
            with self.assertRaises(ValueError):gate.verify_scheduler(value)
    def test_missing_and_extra_metadata_refused(self):
        for extra in (False,True):
            b=copy.deepcopy(self.base)
            if extra:b['metadata'][self.folder+'/unexpected.json']=item(b'{}')
            else:del b['metadata'][self.folder+'/encode/completion.json']
            with self.assertRaises((ValueError,KeyError)):gate.verify_bundle(b)
    def test_wrong_file_hash_or_size_refused(self):
        b=copy.deepcopy(self.base);b['metadata'][self.folder+'/encode/completion.json']['sha256']='0'*64
        with self.assertRaises(ValueError):gate.verify_bundle(b)
        b=copy.deepcopy(self.base);b['payload_stats'][self.folder+'/encode/vectors.npy']['size']-=1
        with self.assertRaises(ValueError):gate.verify_bundle(b)
    def test_failed_wrongrange_wrongcount_wrongruntime_refused(self):
        for key,value in [('passed',False),('start',1),('chunks',1),('runtime',{}),('numerical_contract',{**gate.CONTRACT,'threads':2})]:
            b=copy.deepcopy(self.base);path=self.folder+'/encode/completion.json';d=get(b,path);d[key]=value;rebind(b,path,d)
            with self.assertRaises(ValueError):gate.verify_bundle(b)
    def test_replay_must_bind_encode_and_equal_file_claims(self):
        for key,value in [('encoded_completion',{'bytes':1,'sha256':'0'*64}),('exact_vector_bytes_equal',False)]:
            b=copy.deepcopy(self.base);path=self.folder+'/replay/completion.json';d=get(b,path);d[key]=value;rebind(b,path,d)
            with self.assertRaises(ValueError):gate.verify_bundle(b)
    def test_wrong_physical_job_and_missing_directory_refused(self):
        b=copy.deepcopy(self.base);path=self.folder+'/launch.json';d=get(b,path);d['slurm_job_id']='999';put(b,path,d)
        with self.assertRaises(ValueError):gate.verify_bundle(b)
        b=copy.deepcopy(self.base);b['inventories'][self.folder+'/encode'].append('failure.json')
        with self.assertRaises(ValueError):gate.verify_bundle(b)
    def test_process_command_pid_and_claims_refused(self):
        for key,value in [('command',['wrong']),('child_pid',30000),('automatic_job_submission',True)]:
            b=copy.deepcopy(self.base);path=self.folder+'/replay.process.json';d=get(b,path);d[key]=value;rebind(b,path,d)
            with self.assertRaises(ValueError):gate.verify_bundle(b)
    def test_symlink_read_and_oversized_metadata_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a').write_bytes(b'abc');(p/'link').symlink_to(p/'a')
            with self.assertRaises(ValueError):gate.read_small(p/'link')
            with self.assertRaises(ValueError):gate.read_small(p/'a',cap=2)
    def test_recheck_does_not_read_payloads(self):
        b=copy.deepcopy(self.base)
        with patch.object(gate,'read_small',side_effect=lambda p:b['metadata'][p]) as read,patch.object(gate,'file_stat',side_effect=lambda p:{**b['payload_stats'],**b['log_stats']}[p]),patch.object(gate,'directory_inventory',side_effect=lambda p:b['inventories'][p]):
            gate.recheck_observations(b)
        self.assertEqual(set(c.args[0] for c in read.call_args_list),set(b['metadata']))
    def test_local_export_revalidation(self):
        bundle=self.base;summary=gate.verify_bundle(bundle);raw=gate.canonical(bundle)+b'\n'
        result={**summary,'bundle_file':{'name':'bundle.json',**gate.binding(raw)}}
        value={'schema':'third-e5-array-metadata-export-v1','array_job_id':'13498','report_dir':self.root+'/terminal-metadata-audit-v1-a01',
            'passed_metadata_gate':True,'files':{'bundle.json':item(raw),'metadata_gate_result.json':item(gate.canonical(result)+b'\n')}}
        decoded,result=verify_local.validate(value);self.assertTrue(result['passed'])
        value['files']['bundle.json']['sha256']='0'*64
        with self.assertRaises(ValueError):verify_local.validate(value)
    def test_strict_json(self):
        for raw in ('{"a":1,"a":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):gate.strict_json(raw)


if __name__=='__main__':unittest.main()
