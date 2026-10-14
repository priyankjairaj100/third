"""Synthetic assembly metadata fixtures only; no cluster or payload access."""
import base64
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import read_remote

PRIOR=Path(__file__).resolve().parent.parent/'assembly_gate_v1'
sys.path.insert(0,str(PRIOR))
import test_gate
G=read_remote.load_gate(PRIOR)


def fixture():
    prior_evidence=PRIOR.parent/'terminal-metadata-audit-v1-a01'
    oldraw=(prior_evidence/'bundle.json').read_bytes();old=G.strict_json(oldraw)
    reviewraw=(prior_evidence/'metadata_gate_result.json').read_bytes()
    pins=G.expectations();root=pins['root'];stage=root+'/assembly-run';audit=root+'/terminal-metadata-audit-v1-a01';names=G.paths(pins)
    get=lambda path:test_gate.get(old,path);bind=lambda path:test_gate.bound(old,path)
    plan=get(names['plan']);qualification=get(names['qualification']);budget=get(names['budget']);original=get(root+'/plan_launch_receipt.json')
    metadata={audit+'/bundle.json':test_gate.item(oldraw),audit+'/metadata_gate_result.json':test_gate.item(reviewraw)}
    def put(path,data):metadata[path]=test_gate.item(G.canonical(data)+b'\n')
    bound=lambda path:{k:metadata[path][k] for k in ('bytes','sha256')}
    dependency={'core_package':bind(names['core']),'dispatch_package':bind(names['dispatch']),'qualification_result':bind(names['qualification']),'full_plan':bind(names['plan'])}
    common=original['common_arguments'];dispatchfolder=pins['repository']+'/local_run/e5_sharded_dispatch_v2_20261013'
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-e5-assembly-13498','--output',root+'/assembly-%j.out','--error',root+'/assembly-%j.err',dispatchfolder+'/assembly.sbatch',*common,names['plan'],bind(names['plan'])['sha256'],bind(names['dispatch'])['sha256']]
    intent={'schema':'third-canonical-assembly-launch-v1','source_array_job_id':'13498','command':command,'gate_package_manifest':{'bytes':read_remote.GATE_BYTES,'sha256':read_remote.GATE_SHA},
        'launcher_package_manifest':read_remote.LAUNCHER_BINDING,'stage_budget_report':read_remote.STAGE_BUDGET_BINDING,'budget_policy':'third-assembly-only-capacity-policy-v1',
        'quota_before_launch':{'remaining_output_allowance_bytes':832*1024**2,'reserve_bytes':8*1024**3,'parsed':{'available_soft_bytes':10*1024**3}},
        'materialized_output_budget':{'remaining_allowance_bytes':832*1024**2,'original_output_allowance_bytes':9*1024**3//4,'directory_and_stat_slack_bytes':16*1024**2,'sum_max_logical_or_allocated_bytes':1024**3},
        'actual_payload_validation_deferred_to_sealed_Slurm_assembly_before_publication':True,'full_population_cache_acceptance':False,'confirmatory_study_ready':False,'automatic_retry':False,
        'reviewed_metadata_bundle':G.binding(oldraw),'reviewed_metadata_result':G.binding(reviewraw),**dependency,'scipy_repair_result':bind(names['repair'])}
    launch={**intent,'returncode':0,'stdout':'99999\n','stderr':'','job_id':'99999'}
    put(root+'/assembly_launch_receipt.json',launch);put(root+'/assembly_launch_intent.json',intent)
    put(stage+'/launch.json',{'mode':'assembly','shard_index':None,'slurm_job_id':'99999','hostname':'csis.cn1','bindings':dependency,**{k:False for k in G.DISPATCH_FALSE_FLAGS}})
    files={'vectors.npy':{'bytes':budget['exact_known_payload']['assembly_vectors_npy_bytes'],'sha256':'f'*64},'chunk_log.jsonl':{'bytes':budget['exact_known_payload']['one_complete_chunk_log_bytes'],'sha256':budget['exact_known_payload']['complete_chunk_log_sha256_if_passed']}}
    manifest={'schema':'third-e5-sharded-cpu-cache-v1','passed':True,'plan_sha256':plan['sha256'],'fingerprint':pins['fingerprint'],'rows':140250,'shape':[140250,768],'dtype':'float32',
        'frame_ordered_ids_sha256':plan['frame_ordered_ids_sha256'],'full_source_frame_sha256':plan['full_source_frame_sha256'],'runtime':qualification['runtime'],'all_rows_fresh_process_replayed_under_this_version':True,
        **{k:False for k in ('original_monolithic_full_writer_executed','original_monolithic_full_replay_executed','source_authenticity_verified','primary_input_acceptance','full_population_cache_acceptance','confirmatory_study_ready')},'files':files,
        'shard_receipts':[{'index':i,'encoded_completion':bind(root+f'/shards/shard-{i:04d}/encode/completion.json'),'replay_completion':bind(root+f'/shards/shard-{i:04d}/replay/completion.json')} for i in range(32)]}
    manifest['sha256']=G.binding(G.canonical(manifest))['sha256'];put(stage+'/cache/manifest.json',manifest)
    args=[v for name,arg in zip(('repo','workspace','view','replay-result','census-dir','budget-dir'),common[:6]) for v in ('--'+name,arg)]
    args+=['--package-sha256',dependency['core_package']['sha256'],'--qualification-result',names['qualification'],'--qualification-sha256',dependency['qualification_result']['sha256']]
    cmd=[pins['workspace']+'/envs/encoder-cpu/bin/python',pins['repository']+'/local_run/e5_sharded_cpu_v2_20261013/driver.py','assemble',*args,'--plan',names['plan'],'--out',stage+'/cache','--encoded-dirs']+[root+f'/shards/shard-{i:04d}/encode' for i in range(32)]+['--replay-dirs']+[root+f'/shards/shard-{i:04d}/replay' for i in range(32)]
    put(stage+'/assembly.process.json',{'schema':'third-e5-sharded-dispatch-child-v1','command':cmd,'returncode':0,'child_pid':1,'launcher_pid':2,'child_wall_seconds_through_exit':10.0,'timer_includes_child_artifact_publication_and_exit':True,**{k:False for k in G.DISPATCH_FALSE_FLAGS}})
    put(stage+'/dispatch_result.json',{'mode':'assembly','shard_index':None,'passed':True,'bindings':dependency,**{k:False for k in G.DISPATCH_FALSE_FLAGS},'children':[{'name':'assembly','artifact':bound(stage+'/cache/manifest.json'),'process_receipt':bound(stage+'/assembly.process.json'),'child_wall_seconds_through_exit':10.0}]})
    stat=lambda size:{'dev':1,'ino':1,'size':size,'mtime_ns':1,'ctime_ns':1}
    return {'schema':read_remote.SCHEMA,'root':root,'source_array_job_id':'13498','assembly_job_id':'99999','assembly_launch_sha256':bound(root+'/assembly_launch_receipt.json')['sha256'],
        'reader_sha256':G.binding(Path(read_remote.__file__).read_bytes())['sha256'],
        'scheduler':{'queue_stdout':'','accounting_stdout':'99999|COMPLETED|0:0|20|1G|csis.cn1|\n99999.batch|COMPLETED|0:0|20|1G|csis.cn1|\n'},'metadata':metadata,
        'payload_stats':{stage+'/cache/'+name:stat(value['bytes']) for name,value in files.items()},'log_stats':{stage+'/'+name:stat(0) for name in ('assembly.stdout.txt','assembly.stderr.txt')},
        'inventories':{stage:sorted(['launch.json','dispatch_result.json','assembly.process.json','assembly.stdout.txt','assembly.stderr.txt','cache']),stage+'/cache':['chunk_log.jsonl','manifest.json','vectors.npy']}}


def get(value,path):return G.strict_json(base64.b64decode(value['metadata'][path]['base64']))
def put(value,path,data):value['metadata'][path]=test_gate.item(G.canonical(data)+b'\n')
def rebind(value,path,data):
    stage=value['root']+'/assembly-run';put(value,path,data);dispatch=get(value,stage+'/dispatch_result.json')
    key='artifact' if path.endswith('/manifest.json') else 'process_receipt'
    dispatch['children'][0][key]={k:value['metadata'][path][k] for k in ('bytes','sha256')};put(value,stage+'/dispatch_result.json',dispatch)


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixture();cls.stage=cls.base['root']+'/assembly-run'
    def check(self,value):return read_remote.verify(value,G,value['assembly_launch_sha256'])
    def test_valid_exact_three_file_cache_metadata(self):
        report=self.check(self.base);self.assertTrue(report['passed_metadata_checks']);self.assertFalse(report['independent_cache_payload_hashes_verified']);self.assertFalse(report['full_population_cache_acceptance'])
    def test_requires_exact_parent_success_node_and_absent_queue(self):
        for old,new in [('COMPLETED','FAILED'),('csis.cn1','csis.cn2')]:
            v=copy.deepcopy(self.base);v['scheduler']['accounting_stdout']=v['scheduler']['accounting_stdout'].replace(old,new)
            with self.assertRaises(ValueError):self.check(v)
        for extra in ('99999|RUNNING|1|csis.cn1',''):
            v=copy.deepcopy(self.base)
            if extra:v['scheduler']['queue_stdout']=extra
            else:v['scheduler']['accounting_stdout']='99999.batch|COMPLETED|0:0|1||csis.cn1|\n'
            with self.assertRaises(ValueError):self.check(v)
    def test_wrong_manifest_seal_counts_or_shard_order_refused(self):
        for key,val in [('sha256','0'*64),('rows',140249),('shape',[1,768]),('full_population_cache_acceptance',True)]:
            v=copy.deepcopy(self.base);p=self.stage+'/cache/manifest.json';d=get(v,p);d[key]=val
            if key!='sha256':d['sha256']=G.binding(G.canonical({k:x for k,x in d.items() if k!='sha256'}))['sha256']
            rebind(v,p,d)
            with self.assertRaises(ValueError):self.check(v)
        v=copy.deepcopy(self.base);p=self.stage+'/cache/manifest.json';d=get(v,p);d['shard_receipts'].reverse();d['sha256']=G.binding(G.canonical({k:x for k,x in d.items() if k!='sha256'}))['sha256'];rebind(v,p,d)
        with self.assertRaises(ValueError):self.check(v)
    def test_result_json_is_not_cache_success_artifact(self):
        v=copy.deepcopy(self.base);v['inventories'][self.stage+'/cache']=['chunk_log.jsonl','result.json','vectors.npy']
        with self.assertRaises(ValueError):self.check(v)
        v=copy.deepcopy(self.base);v['inventories'][self.stage+'/cache'].append('result.json')
        with self.assertRaises(ValueError):self.check(v)
    def test_child_exit_or_changed_command_refused(self):
        for key,val in [('returncode',2),('command',['wrong'])]:
            v=copy.deepcopy(self.base);p=self.stage+'/assembly.process.json';d=get(v,p);d[key]=val;rebind(v,p,d)
            with self.assertRaises(ValueError):self.check(v)
    def test_stage_launcher_budget_and_capacity_identity_required(self):
        for key,val in [('launcher_package_manifest',{'bytes':787,'sha256':'0'*64}),('stage_budget_report',{'bytes':6117,'sha256':'0'*64}),('budget_policy','unknown')]:
            v=copy.deepcopy(self.base);path=v['root']+'/assembly_launch_receipt.json';d=get(v,path);d[key]=val;put(v,path,d);v['assembly_launch_sha256']=v['metadata'][path]['sha256']
            with self.assertRaises(ValueError):self.check(v)
        for field,val in [('reserve_bytes',7*1024**3),('remaining_output_allowance_bytes',512*1024**2)]:
            v=copy.deepcopy(self.base);path=v['root']+'/assembly_launch_receipt.json';d=get(v,path);d['quota_before_launch'][field]=val;put(v,path,d);v['assembly_launch_sha256']=v['metadata'][path]['sha256']
            with self.assertRaises(ValueError):self.check(v)
    def test_wrong_payload_size_or_metadata_hash_refused(self):
        v=copy.deepcopy(self.base);v['payload_stats'][self.stage+'/cache/vectors.npy']['size']-=1
        with self.assertRaises(ValueError):self.check(v)
        v=copy.deepcopy(self.base);v['metadata'][self.stage+'/cache/manifest.json']['sha256']='0'*64
        with self.assertRaises(ValueError):self.check(v)
    def test_metadata_recheck_never_reads_payloads(self):
        v=self.base
        with patch.object(G,'read_small',side_effect=lambda p,**kw:v['metadata'][p]) as reads,patch.object(G,'file_stat',side_effect=lambda p:{**v['payload_stats'],**v['log_stats']}[p]),patch.object(G,'directory_inventory',side_effect=lambda p:v['inventories'][p]),patch.object(G,'recheck_observations') as prior:
            read_remote.recheck(v,G)
        self.assertEqual(set(c.args[0] for c in reads.call_args_list),set(v['metadata']));prior.assert_called_once()


if __name__=='__main__':unittest.main()
