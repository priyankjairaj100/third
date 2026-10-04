"""WCEP bridge lineage/refusal checks; no semantic vectors or ratings fabricated."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import copy,hashlib,importlib.util,json,tempfile
from phase3 import calibration as cal
from phase4 import embeddings as emb
from phase5 import replication as rep,replication_encoding as bridge


def main():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError,FileExistsError,RuntimeError,KeyError):checks.append(name)
        else:raise AssertionError(name)
    def save(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n')
    with tempfile.TemporaryDirectory(prefix='ccu_WCEP_BRIDGE_SOFTWARE_FIXTURE_') as td:
        p=Path(td);events=[]
        for i in range(3):
            events.append({'id':'SOFTWARE_EVENT_'+str(i),'date':'2018-01-01','collection':'train',
              'summary':'SOFTWARE SUMMARY MUST NOT BE ENCODED','category':'SOFTWARE CATEGORY NOT A TARGET',
              'articles':[{'title':'SOFTWARE TITLE','text':'SOFTWARE TITLE\nSOFTWARE article body NOT EMPIRICAL DATA','url':'https://example.invalid/'+str(j)} for j in range(2)]})
        original=p/'software_events.jsonl';original.write_text(''.join(json.dumps(e)+'\n' for e in events))
        rep.parse_wcep([original],p/'parsed',snapshot_revision='SOFTWARE_SCHEMA_FIXTURE_NOT_AUTHENTIC_WCEP',collection_scope='train')
        rep.event_panel(p/'parsed',p/'panel',target=3)
        rows,panel,audit=bridge.load_wcep_panel(p/'parsed',p/'panel')
        check('whole_boundary_event_preserved',len(rows)==4 and panel['overshoot']==1)
        check('duplicate_article_instances_preserved',len({r['record_id'] for r in rows})==4)
        check('only_article_text_encoded',all(r['text']=='SOFTWARE TITLE SOFTWARE article body NOT EMPIRICAL DATA' for r in rows))
        check('event_labels_excluded',all(not r['labels'] and 'SUMMARY' not in r['text'] and 'CATEGORY' not in r['text'] for r in rows))
        check('record_only_without_invented_domains',all(r['source_kind']=='unknown_singleton' and r['source_unit_id']=='unknown:'+r['record_id'] for r in rows))
        check('wrapper_reuses_frozen_encoder_functions',bridge.emb.encode_document is emb.encode_document and bridge.emb.LocalTransformer is emb.LocalTransformer)
        check('E5_same_prefix',bridge.preprocessing(emb.E5)['prefix']=='query: ')
        check('MPNet_same_prefix',bridge.preprocessing(emb.MPNET)['prefix']=='')
        check('all_chunks_contract',bridge.preprocessing(emb.E5)['chunking']=='nonoverlapping_all_chunks')
        panelpath=p/'panel/panel.json';recordpath=p/'panel/records.jsonl'
        originalpanel=panelpath.read_bytes();originalrecords=recordpath.read_bytes()
        changed=copy.deepcopy(panel);changed['record_ids']=list(reversed(changed['record_ids']));save(panelpath,cal._seal({k:v for k,v in changed.items() if k!='sha256'}))
        reject('resealed_reordered_IDs_rejected',lambda:bridge.load_wcep_panel(p/'parsed',p/'panel'));panelpath.write_bytes(originalpanel)
        for key,value in [('selected_events',panel['selected_events'][::-1]),('target',True),('threshold_policy','retune_on_WCEP'),('source_arm_available',True),('code_sha256','0'*64)]:
            changed={k:v for k,v in copy.deepcopy(panel).items() if k!='sha256'};changed[key]=value;save(panelpath,cal._seal(changed))
            reject('resealed_'+key+'_rejected',lambda:bridge.load_wcep_panel(p/'parsed',p/'panel'));panelpath.write_bytes(originalpanel)
        mutated=copy.deepcopy(rows);mutated[0]['text']='ALTERED SOFTWARE TEXT';recordpath.write_text(''.join(json.dumps(r)+'\n' for r in mutated))
        reject('altered_record_bytes_rejected',lambda:bridge.load_wcep_panel(p/'parsed',p/'panel'))
        changed={k:v for k,v in panel.items() if k!='sha256'};changed['records_sha256']=emb.file_hash(recordpath);save(panelpath,cal._seal(changed))
        reject('resealed_text_still_rejected_against_adapter',lambda:bridge.load_wcep_panel(p/'parsed',p/'panel'))
        recordpath.write_bytes(originalrecords);panelpath.write_bytes(originalpanel)
        (p/'parsed/FAILED.json').write_text('{}');reject('failed_adapter_rejected',lambda:bridge.load_wcep_panel(p/'parsed',p/'panel'));(p/'parsed/FAILED.json').unlink()
        for batch,threads in [(True,1),(0,1),(1,False),(1,0)]:
            target=p/('bad_controls_'+str(batch)+'_'+str(threads))
            reject('invalid_runtime_controls_'+str((batch,threads)),lambda:bridge.encode_wcep(p/'parsed',p/'panel',p/'missing_assets',p/'missing_model',p/'missing_tokenizer',target,batch_size=batch,threads=threads))
            check('invalid_controls_no_output_'+str((batch,threads)),not target.exists())
        target=p/'missing_assets_output'
        reject('missing_assets_blocked',lambda:bridge.encode_wcep(p/'parsed',p/'panel',p/'missing_assets',p/'missing_model',p/'missing_tokenizer',target))
        check('missing_assets_before_output',not target.exists())
        # Negative-only placeholder byte inventories: runtime versions deliberately
        # impossible. No backend model forward or successful semantic cache exists.
        model=p/'negative_software_model';tokenizer=p/'negative_software_tokenizer';model.mkdir();tokenizer.mkdir()
        save(model/'config.json',{});(model/'model.safetensors').write_text('SOFTWARE NEGATIVE FIXTURE NOT MODEL WEIGHTS');save(tokenizer/'tokenizer.json',{})
        for encoder in emb.ENCODERS:
            short='e5' if encoder==emb.E5 else 'mpnet'
            assets=emb.asset_inventory(model,tokenizer,encoder,'f'*40,'e'*40,{v:'SOFTWARE_RUNTIME_INTENTIONALLY_UNAVAILABLE' for v in emb.VERSIONS})
            assetpath=p/(short+'_negative_assets.json');save(assetpath,assets);target=p/(short+'_should_not_exist')
            reject(short+'_runtime_mismatch_before_backend',lambda:bridge.encode_wcep(p/'parsed',p/'panel',assetpath,model,tokenizer,target))
            check(short+'_runtime_failure_before_output',not target.exists())
        reject('absent_cache_rejected',lambda:bridge.load_wcep_cache(p/'parsed',p/'panel',p/'missing_cache'))
        reject('absent_inheritance_cannot_enable_graph',lambda:bridge.load_inherited_graph_inputs(p/'parsed',p/'panel',p/'missing_cache',p/'missing_lock'))
        target=p/'inherited_curator_must_not_exist.json'
        kwargs={k:p/('missing_'+k) for k in ['news_records','news_preparation','news_vectors','news_cache_manifest','news_assets','selection_manifest','selection_responses','selection_lock','validation_manifest','validation_responses','quality_report']}
        reject('missing_news_dossier_cannot_be_inherited',lambda:bridge.inherit_news_threshold(p/'parsed',p/'panel',p/'missing_cache',out_path=target,**kwargs))
        check('failed_inheritance_no_lock',not target.exists())
        check('no_successful_semantic_cache_generated',not list(p.rglob('cache_manifest.json')))
        check('no_human_responses_generated',not list(p.rglob('*responses*.json')))
    result={'status':'passed','check_count':len(checks),'checks':checks,'code_bindings':bridge.code_bindings(),
      'check_code_sha256':emb.file_hash(__file__),'real_transformer_backend_executed':False,
      'real_WCEP_cache_created':False,'real_News_threshold_inheritance_executed':False,
      'human_ratings_created':0,'synthetic_empirical_data_created':False,
      'scope':'temporary WCEP schema/lineage and negative missing-runtime checks only; no successful fake semantic cache or quality gate'}
    output=Path(__file__).parent/'results/replication_encoding_checks.json';output.parent.mkdir(exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
