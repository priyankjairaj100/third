"""WCEP whole-event offline encoding and inherited CC-News threshold binding.

This is a WCEP-specific workflow, not a manufactured phase3 fixed-guard audit.
Event summaries, categories and membership are never transformer inputs.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,platform,resource,time
from collections import Counter
import numpy as np
from phase3 import calibration as cal,panels,run_preparation
from phase4 import embeddings as emb,calibrate_cache as news_workflow
from phase5 import replication as rep

SCHEMA='ccu-wcep-local-encoding-1'
INHERIT_SCHEMA='ccu-wcep-inherited-news-curator-1'


def code_bindings():
    return {name:emb.file_hash(module.__file__) for name,module in
      [('replication',rep),('embeddings',emb),('panels',panels),('calibration',cal),
       ('replication_encoding',sys.modules[__name__])]}


def load_wcep_panel(adapter_dir,panel_dir):
    adapter=Path(adapter_dir);paneldir=Path(panel_dir)
    audit=emb.read_json(adapter/'audit.json');manifest=emb.read_json(paneldir/'panel.json')
    cal._verify(audit);cal._verify(manifest)
    if (audit.get('schema')!=rep.SCHEMA or audit.get('stage')!='parsed_original_events'
       or manifest.get('schema')!=rep.SCHEMA or manifest.get('stage')!='whole_event_panel'
       or audit.get('dataset_id')!='wcep100' or manifest.get('dataset_id')!='wcep100'
       or (adapter/'FAILED.json').exists()):raise ValueError('Successful original-event WCEP parse and panel required')
    if (audit.get('code_sha256')!=emb.file_hash(rep.__file__)
        or manifest.get('code_sha256')!=emb.file_hash(rep.__file__)
        or audit.get('text_adapter_code_sha256')!=emb.file_hash(sys.modules[rep.news_text.__module__].__file__)
        or manifest.get('adapter_audit_sha256')!=audit['sha256']):raise ValueError('Changed WCEP parser/panel lineage')
    for name in ('records','events','exclusions'):
        if audit.get(name+'_sha256')!=emb.file_hash(adapter/(name+'.jsonl')):raise ValueError('Changed adapter '+name)
    if manifest.get('records_sha256')!=emb.file_hash(paneldir/'records.jsonl'):raise ValueError('Changed panel record bytes')
    events=[r for _,r in rep._events(adapter/'events.jsonl')]
    lookup={r['event_id']:r for r in events}
    if len(lookup)!=len(events):raise ValueError('Repeated event IDs')
    salt=manifest.get('salt');target=manifest.get('target')
    if not isinstance(salt,str) or not salt or isinstance(target,bool) or not isinstance(target,int) or target<1:raise ValueError('Valid frozen event sampling rule required')
    ordered=sorted(events,key=lambda r:(panels.hash_integer(salt,r['event_id']),r['event_id']))
    chosen=[];total=0
    for event in ordered:
        if total>=target:break
        chosen.append(event['event_id']);total+=event['retained_articles']
    if (chosen!=manifest.get('selected_events') or total!=manifest.get('selected_articles')
       or manifest.get('overshoot')!=max(0,total-target) or manifest.get('shortfall')!=max(0,target-total)
       or manifest.get('whole_boundary_event_included') is not True
       or manifest.get('threshold_policy')!='reuse_frozen_same_encoder_CC_News_threshold_no_WCEP_retuning'
       or manifest.get('source_arm_available') is not False):raise ValueError('Whole-event prefix or record-only contract changed')
    chosen_set=set(chosen);expected=[];actual_counts=Counter()
    for _,row in rep._events(adapter/'records.jsonl'):
        actual_counts[row.get('event_id')]+=1
        if row.get('event_id') in chosen_set:expected.append(row)
    rows=[r for _,r in rep._events(paneldir/'records.jsonl')]
    if not rows or rows!=expected:raise ValueError('Panel must exactly retain whole selected events in adapter row order')
    ids=[r.get('record_id') for r in rows]
    if len(ids)!=len(set(ids)) or ids!=manifest.get('record_ids') or len(rows)!=total:raise ValueError('Panel ordered instance IDs/count mismatch')
    if any(actual_counts[eid]!=event['retained_articles'] for eid,event in lookup.items()) or set(actual_counts)-set(lookup):raise ValueError('Event ledger does not match article population')
    for row in rows:
        event=lookup[row['event_id']];original_event=event['original_event_fields'];original=row.get('original_fields',{})
        ordinal=row.get('article_ordinal')
        if isinstance(ordinal,bool) or not isinstance(ordinal,int) or not 0<=ordinal<event['supplied_articles']:raise ValueError('Native within-event article ordinal required')
        instance='wcep:'+cal.digest([event['input_sha256'],row['event_id'],ordinal])
        text,repeated=rep.news_text(original['title'],original['text'])
        if (row['record_id']!=instance or row['text']!=text or text!=panels.normalized_text(text)
           or row.get('event_date')!=original_event['date'] or row.get('collection')!=original_event['collection']
           or row.get('labels')!=[] or row.get('source_unit_id')!='unknown:'+instance
           or row.get('source_kind')!='unknown_singleton'
           or row.get('adapter_input',{}).get('file_sha256')!=event['input_sha256']
           or row.get('adapter_input',{}).get('event_line')!=event['input_line']
           or row.get('adapter_input',{}).get('exact_repeated_title_removed')!=repeated):
            raise ValueError('WCEP instance/source/text lineage inconsistent')
    return rows,manifest,audit


def preprocessing(encoder):
    return {'chunk_content_tokens':256,'chunking':'nonoverlapping_all_chunks',
      'pooling':'content_token_weighted_mean_then_normalize','prefix':emb.ENCODERS[encoder],
      'storage_dtype':'float32','learner_normalization':'no_renormalization_after_FP32_storage'}


def implementation(batch_size,threads):
    # Intentionally identical to the frozen phase4 transformer execution contract.
    return {'tokenization':'content_once_without_specials; prefix_tokenized_separately_per_chunk; add_model_specials; no_decode_or_truncation',
      'chunk_pooling':'FP32_attention_masked_mean_including_prefix_and_specials_then_FP64_coordinate_order_L2_normalize',
      'document_pooling':'FP64_content_count_weighted_sum_in_chunk_order_then_FP64_coordinate_order_L2_normalize_cast_FP32',
      'graph':'phase3.panels.graph_normalize_and_reference_cosines_on_frozen_FP32',
      'learner':'exact_promotion_of_stored_FP32_without_renormalization','batch_scope':'chunks_of_one_record_only',
      'batch_size':batch_size,'padding':'fixed_CONTENT_LIMIT_plus_prefix_and_specials','device':'cpu','model_dtype':'float32',
      'threads':threads,'interop_threads':1,'seed':0,'mkldnn':False,'deterministic_algorithms':True,
      'attention_implementation':'eager','library_versions':emb.installed_versions(),'python':platform.python_version(),
      'platform':platform.platform(),'cross_machine_bit_identity_claimed':False,
      'empirical_model_execution_tested_in_current_development_workspace':False}


def encode_wcep(adapter_dir,panel_dir,asset_manifest_path,model_dir,tokenizer_dir,out_dir,*,batch_size=8,threads=1):
    out=Path(out_dir)
    if out.exists():raise FileExistsError('Frozen output exists')
    for v in (batch_size,threads):
        if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError('Positive thread/chunk batch counts required')
    rows,panel,audit=load_wcep_panel(adapter_dir,panel_dir)
    assets=emb.verify_assets(emb.read_json(asset_manifest_path),model_dir,tokenizer_dir)
    encoder=assets['encoder_id']
    # Genuine local backend only. Missing assets/runtime/model load fail before output.
    backend=emb.LocalTransformer(model_dir,tokenizer_dir,encoder,threads=threads)
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        cache=np.lib.format.open_memmap(out/'vectors.npy',mode='w+',dtype=np.float32,shape=(len(rows),768))
        token_count=chunk_count=0;maximum_tokens=0
        with (out/'chunk_log.jsonl').open('x',encoding='utf-8') as stream:
            for index,row in enumerate(rows):
                vector,detail=emb.encode_document(backend.tokenizer,backend,row['text'],encoder,backend.max_length,batch_size=batch_size)
                cache[index]=vector
                stream.write(json.dumps({'row_index':index,'record_id':row['record_id'],
                  'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),**detail},sort_keys=True)+'\n')
                token_count+=detail['content_tokens'];chunk_count+=detail['chunks'];maximum_tokens=max(maximum_tokens,detail['content_tokens'])
        cache.flush();del cache
        emb.verify_assets(assets,model_dir,tokenizer_dir)
        again,panel_again,audit_again=load_wcep_panel(adapter_dir,panel_dir)
        if (again,panel_again,audit_again)!=(rows,panel,audit):raise ValueError('WCEP input changed while encoding')
        emb.write_json(out/'row_ids.json',[r['record_id'] for r in rows]);emb.write_json(out/'assets.json',assets)
        summary={'rows':len(rows),'dimension':768,'content_tokens':token_count,'chunks':chunk_count,
          'maximum_record_content_tokens':maximum_tokens,'seconds':time.perf_counter()-started,
          'model_parameter_bytes':backend.model_bytes,'model_buffer_bytes':backend.model_buffer_bytes,
          'output_npy_bytes':(out/'vectors.npy').stat().st_size,
          'peak_process_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'rss_units':'KiB on Linux; bytes on macOS; includes earlier process work',
          'memory_scope':'Selected adapter rows, panel rows and the event ledger reside in memory; unselected adapter records are streamed, plus model/tokenizer, one complete document token/chunk list, one intradocument forward batch, mapped output resident pages and OS cache; not constant total memory',
          'event_summary_category_or_event_label_encoded':False,'semantic_quality_established':False}
        emb.write_json(out/'encoding_summary.json',summary)
        manifest=cal._seal({'schema':SCHEMA,'dataset_id':'wcep100','population_scope':'whole_event_article_instances_record_only',
          'panel_sha256':panel['sha256'],'adapter_audit_sha256':audit['sha256'],
          'records_file_sha256':emb.file_hash(Path(panel_dir)/'records.jsonl'),'prepared_rows_sha256':cal.digest(rows),
          'normalized_records_sha256':run_preparation.text_binding(rows),'row_ids':[r['record_id'] for r in rows],
          'shape':[len(rows),768],'dtype':'float32','cache_file_sha256':emb.file_hash(out/'vectors.npy'),
          'asset_manifest_sha256':assets['sha256'],'code_sha256':code_bindings(),
          'encoder_id':encoder,'encoder_revision':assets['encoder_revision'],'tokenizer_revision':assets['tokenizer_revision'],
          'preprocessing':preprocessing(encoder),'implementation_lock':implementation(batch_size,threads),
          'chunk_log_sha256':emb.file_hash(out/'chunk_log.jsonl'),'encoding_summary_sha256':emb.file_hash(out/'encoding_summary.json'),
          'threshold_policy':'must_bind_same_encoder_CC_News_quality_dossier_before_graph_execution',
          'threshold_retuning_on_WCEP_permitted':False,'source_arm_available':False,
          'input_authenticity_verified_by_software':False,'confirmatory_study_ready':False})
        emb.write_json(out/'cache_manifest.json',manifest);return manifest
    except Exception as error:
        emb.write_json(out/'FAILED.json',{'successful_cache':False,'exception_type':type(error).__name__,'message':str(error)})
        raise


def load_wcep_cache(adapter_dir,panel_dir,cache_dir):
    rows,panel,audit=load_wcep_panel(adapter_dir,panel_dir);directory=Path(cache_dir)
    manifest=emb.read_json(directory/'cache_manifest.json');assets=emb.read_json(directory/'assets.json')
    cal._verify(manifest);cal._verify(assets)
    encoder=manifest.get('encoder_id')
    if (manifest.get('schema')!=SCHEMA or manifest.get('dataset_id')!='wcep100'
       or encoder not in emb.ENCODERS or manifest.get('population_scope')!='whole_event_article_instances_record_only'
       or (directory/'FAILED.json').exists()):raise ValueError('Successful WCEP cache contract required')
    expected={'panel_sha256':panel['sha256'],'adapter_audit_sha256':audit['sha256'],
      'records_file_sha256':emb.file_hash(Path(panel_dir)/'records.jsonl'),'prepared_rows_sha256':cal.digest(rows),
      'normalized_records_sha256':run_preparation.text_binding(rows),'row_ids':[r['record_id'] for r in rows],
      'shape':[len(rows),768],'dtype':'float32','code_sha256':code_bindings(),'preprocessing':preprocessing(encoder)}
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('WCEP cache lineage/preprocessing mismatch')
    if assets.get('schema')!=emb.ASSET_SCHEMA or manifest.get('asset_manifest_sha256')!=assets['sha256']:raise ValueError('Cache asset mismatch')
    for k in ('encoder_id','encoder_revision','tokenizer_revision'):
        if manifest.get(k)!=assets.get(k):raise ValueError('Cache/asset '+k+' mismatch')
    for k in ('encoder_revision','tokenizer_revision'):
        if not isinstance(manifest[k],str) or not emb.REVISION.fullmatch(manifest[k]):raise ValueError('Full immutable revisions required')
    lock=manifest.get('implementation_lock',{})
    versions=assets.get('library_versions',{})
    if (set(versions)!=set(emb.VERSIONS) or any(not isinstance(v,str) or not v for v in versions.values())
        or lock.get('library_versions')!=versions):raise ValueError('Runtime/asset mismatch')
    for k in ('batch_size','threads'):
        if isinstance(lock.get(k),bool) or not isinstance(lock.get(k),int) or lock[k]<1:raise ValueError('Invalid frozen runtime controls')
    expected_lock=implementation(lock['batch_size'],lock['threads'])
    for k in ('library_versions','python','platform'):expected_lock[k]=lock.get(k)
    if lock!=expected_lock or any(not isinstance(lock.get(k),str) or not lock[k] for k in ('python','platform')):
        raise ValueError('Altered encoder implementation contract')
    if (manifest.get('threshold_retuning_on_WCEP_permitted') is not False
       or manifest.get('source_arm_available') is not False
       or manifest.get('threshold_policy')!='must_bind_same_encoder_CC_News_quality_dossier_before_graph_execution'):
        raise ValueError('WCEP inherited-threshold/record-only policy altered')
    for key in ('model_files','tokenizer_files'):
        files=assets.get(key)
        if not isinstance(files,list) or not files:raise ValueError('Complete pinned asset inventories required')
        names=[]
        for item in files:
            name=item.get('path');size=item.get('bytes');digest=item.get('sha256')
            if (not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts
                or isinstance(size,bool) or not isinstance(size,int) or size<0
                or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Malformed pinned asset inventory')
            names.append(name)
        if len(names)!=len(set(names)):raise ValueError('Repeated asset path')
        if key=='model_files' and ('config.json' not in names or not any(n.endswith('.safetensors') for n in names)):
            raise ValueError('Required model config/safetensors inventory absent')
    for key,name in [('cache_file_sha256','vectors.npy'),('chunk_log_sha256','chunk_log.jsonl'),('encoding_summary_sha256','encoding_summary.json')]:
        if manifest.get(key)!=emb.file_hash(directory/name):raise ValueError('Changed derived cache '+name)
    if emb.read_json(directory/'row_ids.json')!=expected['row_ids']:raise ValueError('Changed cached row IDs')
    cache=np.load(directory/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if cache.dtype!=np.float32 or cache.shape!=(len(rows),768):raise ValueError('Aligned stored N-by-768 FP32 required')
    for start in range(0,len(rows),256):panels.graph_normalize(cache[start:start+256])
    return rows,cache,manifest,assets


def inherit_news_threshold(adapter_dir,panel_dir,cache_dir,*,news_records,news_preparation,news_vectors,
      news_cache_manifest,news_assets,selection_manifest,selection_responses,selection_lock,
      validation_manifest,validation_responses,quality_report,out_path):
    """Recompute News gates; WCEP offers no threshold selection operation.

    Artifact consistency is checked, while authenticity/human independence still
    require external review. Semantic quality on News is not WCEP truth coverage.
    """
    if Path(out_path).exists():raise FileExistsError('Inherited curator lock exists')
    rows,cache,wcep,assets=load_wcep_cache(adapter_dir,panel_dir,cache_dir)
    nrows,nvectors,provenance=news_workflow.load_aligned_cache(news_records,news_preparation,news_vectors,news_cache_manifest,news_assets)
    nmanifest=emb.read_json(news_cache_manifest);nassets=emb.read_json(news_assets)
    if provenance.get('dataset_id')!='cc_news' or provenance.get('evidence_role')!='confirmatory_calibration':raise ValueError('Confirmed-scope source-disjoint CC-News calibration required')
    if (assets['sha256']!=nassets['sha256'] or wcep['encoder_id']!=provenance['encoder_id']
        or wcep['encoder_revision']!=provenance['encoder_revision']
        or wcep['preprocessing']!=nmanifest['preprocessing']
        or wcep['implementation_lock']!=nmanifest['implementation_lock']):raise ValueError('WCEP must reuse exactly the same encoder assets/runtime/chunk contract as News')
    selection=emb.read_json(selection_manifest);lock=emb.read_json(selection_lock);validation=emb.read_json(validation_manifest);quality=emb.read_json(quality_report)
    for value in (selection,lock,validation,quality):cal._verify(value)
    frame=cal._check_inputs(nvectors,nrows,provenance,selection['frame']['scorer']['block_size'])
    if selection['frame']!=frame or lock['frame']!=frame or validation['frame']!=frame:raise ValueError('News calibration frame differs from actual bound News cache')
    wanted_lock=cal.select_threshold(selection,emb.read_json(selection_responses))
    if wanted_lock!=lock:raise ValueError('Selection responses do not reproduce inherited News threshold')
    wanted_quality=cal.quality_gate(validation,emb.read_json(validation_responses),lock)
    if wanted_quality!=quality or not quality.get('statistical_and_declared_scope_eligible'):raise ValueError('News human quality gate did not pass')
    result=cal._seal({'schema':INHERIT_SCHEMA,'dataset_id':'wcep100','cache_manifest_sha256':wcep['sha256'],
      'panel_sha256':wcep['panel_sha256'],'encoder_id':wcep['encoder_id'],'encoder_revision':wcep['encoder_revision'],
      'news_cache_manifest_sha256':nmanifest['sha256'],'news_selection_lock_sha256':lock['sha256'],
      'news_quality_report_sha256':quality['sha256'],'news_validation_manifest_sha256':validation['sha256'],
      'news_selection_responses_file_sha256':emb.file_hash(selection_responses),
      'news_validation_responses_file_sha256':emb.file_hash(validation_responses),
      'threshold':lock['threshold'],'scorer':frame['scorer'],'threshold_policy':'frozen_same_encoder_CC_News_threshold_no_WCEP_retuning',
      'code_sha256':code_bindings(),'news_workflow_code_sha256':emb.file_hash(news_workflow.__file__),
      'source_arm_available':False,'event_membership_is_duplicate_gold':False,
      'WCEP_precision_or_semantic_truth_guaranteed':False,'external_provenance_verified_by_software':False,
      'confirmatory_study_ready':False})
    emb.write_json(out_path,result);return result


def load_inherited_graph_inputs(adapter_dir,panel_dir,cache_dir,inherited_lock_path):
    rows,cache,manifest,_=load_wcep_cache(adapter_dir,panel_dir,cache_dir)
    lock=emb.read_json(inherited_lock_path);cal._verify(lock)
    if (lock.get('schema')!=INHERIT_SCHEMA or lock.get('cache_manifest_sha256')!=manifest['sha256']
       or lock.get('panel_sha256')!=manifest['panel_sha256'] or lock.get('encoder_id')!=manifest['encoder_id']
       or lock.get('encoder_revision')!=manifest['encoder_revision'] or lock.get('code_sha256')!=code_bindings()
       or lock.get('news_workflow_code_sha256')!=emb.file_hash(news_workflow.__file__)
       or lock.get('threshold_policy')!='frozen_same_encoder_CC_News_threshold_no_WCEP_retuning'):
        raise ValueError('Missing/mismatched inherited News-curator lock')
    scorer=lock.get('scorer',{})
    if (scorer.get('numpy')!=np.__version__ or scorer.get('python')!=platform.python_version()
        or scorer.get('shared_scorer_sha256')!=emb.file_hash(panels.__file__)
        or scorer.get('code_sha256')!=emb.file_hash(cal.__file__)
        or scorer.get('score_predicate')!='strict_greater_than'):
        raise ValueError('Inherited scorer runtime/code changed')
    tau=lock.get('threshold')
    if isinstance(tau,bool) or not isinstance(tau,(int,float)) or not np.isfinite(tau) or not -1<=tau<=1:raise ValueError('Invalid inherited threshold')
    return rows,cache,float(tau),lock


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('encode','check-cache','inherit-news','graph-inputs'):
        p=sub.add_parser(command)
        for name in ('adapter-dir','panel-dir'):p.add_argument('--'+name,required=True)
        if command=='encode':
            for name in ('asset-manifest','model-dir','tokenizer-dir','out'):p.add_argument('--'+name,required=True)
            p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=1)
        else:p.add_argument('--cache-dir',required=True)
        if command=='inherit-news':
            for name in ('news-records','news-preparation','news-vectors','news-cache-manifest','news-assets','selection-manifest','selection-responses','selection-lock','validation-manifest','validation-responses','quality-report','out'):
                p.add_argument('--'+name,required=True)
        if command=='graph-inputs':p.add_argument('--inherited-lock',required=True)
    a=parser.parse_args();kw=vars(a);command=kw.pop('command')
    try:
        if command=='encode':
            value=encode_wcep(kw.pop('adapter_dir'),kw.pop('panel_dir'),kw.pop('asset_manifest'),kw.pop('model_dir'),kw.pop('tokenizer_dir'),kw.pop('out'),**kw)
        elif command=='check-cache':
            value=load_wcep_cache(**kw)[2]
        elif command=='inherit-news':
            kw['out_path']=kw.pop('out');value=inherit_news_threshold(**kw)
        else:
            kw['inherited_lock_path']=kw.pop('inherited_lock');value=load_inherited_graph_inputs(**kw)[3]
        print(json.dumps({'schema':value['schema'],'sha256':value['sha256'],'confirmatory_study_ready':False}))
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError,KeyError) as error:parser.exit(2,'BLOCKED: '+str(error)+'\n')

if __name__=='__main__':main()
