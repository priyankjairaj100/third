#!/usr/bin/env python3
"""Independent Phase4 review; transient algebra/parser fixtures are not empirical data.

Never creates semantic embeddings or human labels. Natural execution artifacts
are checked against a separately accumulated scalar graph and dual ridge solve.
"""
from pathlib import Path
import argparse, copy, hashlib, itertools, json, math, sys, tempfile
from fractions import Fraction
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'empirical_execution'))
from phase4 import adapters as ad, embeddings as em, requests as rq, model_selection as ms
from ccu.core import BlockerGraph
from ccu.data import read_natural_jsonl, lexical_engineering_features

CHECKS = []
def check(name, condition):
    if not condition: raise AssertionError(name)
    CHECKS.append(name)
def rejects(name, call, exceptions=(ValueError, RuntimeError)):
    try: call()
    except exceptions: check(name, True)
    else: raise AssertionError(name + ': was accepted')
def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines()]
def filehash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parser_checks(tmp):
    # Metadata strings below expressly describe software fixtures, not a corpus.
    text, report = ad.stack_text('Q?', '<p>Do NOT erase 42.</p><blockquote>quoted &amp; clear</blockquote><pre><code>x &lt; y</code></pre><p><a href="https://fixture.invalid/path">author link</a></p><div class="post-notice">platform notice</div>')
    check('HTML retains negation number quotation code and author URL', all(x in text for x in ['NOT', '42', 'quoted & clear', 'x < y', 'author link', 'https://fixture.invalid/path']))
    check('only marked notice removed', 'platform notice' not in text and report['removed_tagged_platform_notices'] == 1)
    ambiguous, audit = ad.stack_text('Q', '<p>Possible Duplicate is my own observation.</p>')
    check('ambiguous legacy notice preserved and flagged', 'Possible Duplicate' in ambiguous and audit['ambiguous_legacy_notice_text'])
    check('News removes only exact normalized first title line', ad.news_text('A title', '\nA   title\nBody text')[0] == 'A title Body text')
    check('News does not remove title-prefix body sentence', ad.news_text('A title', 'A title continued')[0] == 'A title A title continued')
    check('News keeps later duplicate title', ad.news_text('A title', 'Body\nA title')[0] == 'A title Body A title')
    psl = tmp / 'SOFTWARE_FIXTURE_NOT_OFFICIAL_PSL.txt'
    psl.write_text('// ===BEGIN ICANN DOMAINS===\ncom\nuk\nco.uk\n*.ck\n!www.ck\n// ===END ICANN DOMAINS===\n// ===BEGIN PRIVATE DOMAINS===\nblogspot.com\n// ===END PRIVATE DOMAINS===\n')
    parsed = ad.OfflinePSL(psl, filehash(psl))
    expected = {'a.b.co.uk':'b.co.uk', 'a.b.ck':'a.b.ck', 'www.ck':'www.ck', 'a.www.ck':'www.ck', 'x.blogspot.com':'x.blogspot.com', 'a.com':'a.com', 'com':None, 'a.unknowninvalid':None}
    check('PSL exact wildcard exception private and unknown cases', all(parsed.registrable(h) == e for h,e in expected.items()))
    rejects('PSL wrong bytes rejected', lambda: ad.OfflinePSL(psl, '0'*64))
    for host in ['127.0.0.1', 'example.com:80', 'https://example.com']:
        rejects('nonhostname refused '+host, lambda host=host: ad.canonical_host(host))
    posts = tmp / 'Posts.xml'; links = tmp / 'PostLinks.xml'
    root = ET.Element('posts')
    for rid,date,owner,kind in [('1','2018-05-01T23:59:59','10','1'), ('2','2018-05-02T00:00:00','10','1'), ('3','2023-12-31T23:59:59','-1','1'), ('4','2024-01-01T00:00:00','11','1'), ('5','2020-01-01T00:00:00',None,'1'), ('6','2020-01-01T00:00:00','12','2')]:
        attrs = dict(Id=rid, CreationDate=date, PostTypeId=kind, Title='SOFTWARE FIXTURE '+rid, Body='<p>Question body '+rid+'</p>', Tags='<one>')
        if owner is not None: attrs['OwnerUserId'] = owner
        ET.SubElement(root, 'row', attrs)
    ET.ElementTree(root).write(posts, encoding='utf-8', xml_declaration=True)
    root = ET.Element('postlinks'); ET.SubElement(root,'row',dict(Id='1', LinkTypeId='3', PostId='2', RelatedPostId='3')); ET.SubElement(root,'row',dict(Id='2',LinkTypeId='1', PostId='2', RelatedPostId='5'))
    ET.ElementTree(root).write(links, encoding='utf-8', xml_declaration=True)
    audit=ad.adapt_stack(posts,links,tmp/'stack',dataset_id='askubuntu')
    rows=lines(tmp/'stack/records.jsonl')
    check('Stack inclusive date endpoints and questions only', [r['record_id'] for r in rows] == ['2','3','5'])
    check('community and missing account are distinct singletons', rows[1]['source_kind'] == rows[2]['source_kind'] == 'unknown_singleton' and rows[1]['source_unit_id'] != rows[2]['source_unit_id'])
    check('native duplicate link only preserved', lines(tmp/'stack/duplicate_links.jsonl') == [['2','3']])
    check('adapter provenance and study stay unverified', audit['input_authenticity_verified'] is False and audit['confirmatory_study_ready'] is False)
    civil=tmp/'civil.jsonl'
    civil_rows=[dict(id='1',publication_id='SOFTWARE',article_id=1,parent_id='',created_date='2020-01-01',text='Parent own text',toxicity=.1),dict(id='2',publication_id='SOFTWARE',article_id=2,parent_id='1',created_date='2020-01-02',text='Child own text',toxicity=.25)]
    civil.write_text(''.join(json.dumps(r)+'\n' for r in civil_rows))
    ad.adapt_civil(civil,tmp/'civil')
    result=lines(tmp/'civil/records.jsonl')
    check('Civil native parent matches parent record identity', str(result[1]['original_fields']['parent_id']) == result[0]['record_id'])
    check('Civil preserves fraction and own text only', result[1]['labels'] == [.25] and result[1]['text'] == 'Child own text')

def embedding_checks():
    class Tokenizer:
        def encode(self,text,**kwargs): return [991,992] if text == 'query: ' else list(range(1,514))
        def num_special_tokens_to_add(self,pair=False): return 2
        def prepare_for_model(self,ids,**kwargs): return {'input_ids':[901]+ids+[902], 'attention_mask':[1]*(len(ids)+2)}
    tok=Tokenizer(); chunks,details=em.chunk_inputs(tok,'SOFTWARE TOKEN SEQUENCE',em.E5,512)
    recovered=[token for c in chunks for token in c['input_ids'][3:-1]]
    check('all 513 content tokens exactly once in order', recovered == list(range(1,514)) and [c['content_tokens'] for c in chunks] == [256,256,1])
    check('prefix included independently in each chunk', all(c['input_ids'][1:3] == [991,992] for c in chunks))
    check('first-window loss explicitly counted', details['first_chunk_baseline_omitted_tokens'] == 257 and details['model_first_window_omitted_tokens'] == 5)
    means=np.array([[3,4],[0,2],[-1,0]],dtype=np.float32)
    def backend(cs): return np.asarray([means[c['content_start']//256] for c in cs],dtype=np.float32)
    value,_=em.encode_document(tok,backend,'SOFTWARE TOKEN SEQUENCE',em.E5,512,batch_size=2,dimension=2)
    normalized=means.astype(float)/np.linalg.norm(means.astype(float),axis=1,keepdims=True)
    expected=np.sum(normalized*np.array([256,256,1])[:,None],axis=0);expected/=np.linalg.norm(expected)
    check('content-weighted pooled vector matches independent oracle', np.max(np.abs(value.astype(float)-expected)) < 5e-8)
    one,_=em.encode_document(tok,backend,'SOFTWARE TOKEN SEQUENCE',em.E5,512,batch_size=1,dimension=2)
    check('fixture pooling independent of chunk batch size', np.array_equal(value,one))
    rejects('capacity cannot silently truncate chunks', lambda: em.chunk_inputs(tok,'fixture',em.E5,259))
    rejects('zero backend vector rejected', lambda: em.encode_document(tok,lambda cs: np.zeros((len(cs),2),np.float32),'fixture',em.E5,512,dimension=2))

def graph_fixture(edges):
    ids=('a','b','c','d','e','f');owners=('genuine1','genuine1','genuine2','unknown:d','unknown:e','proxy')
    ptr=[0];index=[]
    for i in range(6): index.extend(sorted(j for a,j in edges if a==i));ptr.append(len(index))
    return BlockerGraph(ids,owners,np.arange(6),np.asarray(ptr),np.asarray(index,dtype=np.int64),.6)

def request_checks():
    graph=graph_fixture({(1,0),(2,1),(4,2),(4,3),(5,0),(5,2)})
    kinds={'genuine1':'genuine_native','genuine2':'genuine_native','unknown:d':'unknown_singleton','unknown:e':'unknown_singleton','proxy':'engineering_proxy'}
    kwargs=dict(dataset_id='SOFTWARE_FIXTURE_ONLY',panel_id='SOFTWARE_FIXTURE_ONLY',source_kinds=kinds,allocations={'R':7,'S':7,'U':7,'A':7},record_horizon=6,source_horizon=8)
    manifest=rq.generate_manifest(graph,**kwargs);rq.verify_manifest(manifest,graph)
    paths=manifest['trajectories'];byid={p['trajectory_id']:p for p in paths};owners=dict(zip(graph.record_ids,graph.source_ids))
    check('source service universe retains unknown and proxy units', set(manifest['source_service_universe']) == set(kinds))
    source=[p for p in paths if p['arm']=='S']
    check('S samples only genuine sources and clips horizon', all(set(p['deletion_order']) == {'genuine1','genuine2'} and p['initial_horizon']==2 for p in source))
    check('all source deletions expand every owned record', all(set(o['deleted_record_ids']) == {r for r,s in owners.items() if s in p['deletion_order'][:o['checkpoint']]} for p in source for o in p['observations']))
    for p in source:
        m=byid[p['trajectory_id'].replace('S','R-volume-matched-to-S',1)]
        check('matched R '+p['trajectory_id'], m['initial_horizon']==3 and m['checkpoints']==sorted(set(o['cumulative_deleted_records'] for o in p['observations'])) and len(set(m['deletion_order']))==3)
    blocked={r:{graph.record_ids[int(j)] for j in graph.blockers[i]} for i,r in enumerate(graph.record_ids)}
    initial={r for r,b in blocked.items() if not b};observations=0
    for p in paths:
        for o in p['observations']:
            deleted=set(o['deleted_record_ids']); selected={r for r,b in blocked.items() if r not in deleted and not (b-deleted)}
            check('direct selection '+p['trajectory_id']+'/'+str(o['checkpoint']), set(o['selected_record_ids'])==selected and set(o['admitted_record_ids'])==selected-initial)
            observations+=1
    for arm,summary in manifest['arm_summaries'].items():
        actual=[p for p in paths if p['arm']==arm]
        for row in summary['per_checkpoint']:
            k=row['checkpoint'];expected=len({tuple(sorted(p['deletion_order'][:k])) for p in actual if k in p['checkpoints']})
            check('distinct sets '+arm+'/'+str(k),row['distinct_request_sets']==expected)
    alternative=rq.generate_manifest(graph_fixture({(2,0)}),**kwargs)
    alternative={p['trajectory_id']:p for p in alternative['trajectories']}
    check('R S and volume matched control paired across curator alternatives', all(p['deletion_order']==alternative[p['trajectory_id']]['deletion_order'] for p in paths if p['arm'] in {'R','S','R-volume-matched-to-S'}))
    check('requests cannot assert primary source evidence', not manifest['confirmatory_study_ready'] and not manifest['genuine_source_withdrawal_evidence'])
    return observations

def extension_checks():
    from phase4 import requests_extensions as rext
    from unittest.mock import patch
    counts={'a':0,'b':0,'c':0}
    for draw in range(6):
        class ControlledDraw:
            def __init__(self,seed):pass
            def randrange(self,total):
                if total!=6:raise AssertionError('Unexpected mass')
                return draw
        with patch.object(rext.random,'Random',ControlledDraw):
            order,records=rext.weighted_source_prefix(['c','a','b'],{'a':1,'b':2,'c':3},1,0)
        counts[order[0]]+=1
        check('PPS exact integer draw '+str(draw), records[0]['conditional_probability_numerator']=={'a':1,'b':2,'c':3}[order[0]] and records[0]['conditional_probability_denominator']==6)
    check('PPS first-step exact mass distribution',counts=={'a':1,'b':2,'c':3})
    g=graph_fixture({(1,0),(2,1)})
    kinds={'genuine1':'genuine_native','genuine2':'genuine_native','unknown:d':'unknown_singleton','unknown:e':'unknown_singleton','proxy':'engineering_proxy'}
    manifest=rext.generate_structure_extensions(g,dataset_id='SOFTWARE_FIXTURE_ONLY',panel_id='SOFTWARE_FIXTURE_ONLY',source_kinds=kinds,mass_source_paths=4,one_percent_allocations={'R':4,'U':4,'A':2})
    rext.verify_structure_extensions(manifest,g)
    check('mass sources genuine-only and uniform formulas forbidden',all(set(p['deletion_order'])=={'genuine1','genuine2'} and not p['uniform_source_admission_formula_applicable'] for p in manifest['mass_source_trajectories']))
    one=manifest['one_percent_record_manifest']
    check('one percent service uses separately fixed ceil horizon',manifest['configuration']['one_percent_record_horizon']==1 and all(p['initial_horizon']==1 for p in one['trajectories']) and manifest['one_percent_service_requires_separate_initial_state'])
    check('structure extensions preserve blocked semantic readiness',manifest['confirmatory_study_ready'] is False)

def integrity_checks():
    from phase4 import study_lock
    for supplied in [None, {}, study_lock.empty_template()]:
        got=study_lock.validate_execution_lock(supplied,{})
        check('incomplete integrity dossier cannot permit execution '+str(len(CHECKS)),got['execution_allowed'] is False and got['mechanical_integrity_passed'] is False and got['external_provenance_verified_by_software'] is False)
    check('integrity review lists substantive incomplete validations',len(got['unimplemented_validations'])>=3)

def calendar_checks():
    from phase4 import news_calendar as news
    coverage={'months':{m:{'status':'complete','basis':'SOFTWARE_FIXTURE_NO_ACQUISITION_EVIDENCE'} for m in news.MONTHS}}
    selected=news.calendar_decision({'2018-01':4,'2018-02':12,'2018-03':15},coverage,target=10)
    check('calendar earliest individually qualifying month',selected['selected_months']==['2018-02'] and selected['overshoot']==2)
    selected=news.calendar_decision({'2018-01':4,'2018-02':6,'2018-03':3},coverage,target=10)
    check('calendar concatenates whole chronological prefix when no month qualifies',selected['selected_months']==['2018-01','2018-02'] and selected['target_met'])
    gap=copy.deepcopy(coverage);gap['months']['2018-02']['status']='incomplete'
    selected=news.calendar_decision({'2018-01':4,'2018-02':6,'2018-03':3},gap,target=10)
    check('calendar prefix never crosses undeclared complete gap',selected['selected_months']==['2018-01'] and selected['shortfall']==6 and selected['prefix_stopped_at_coverage_gap']=='2018-02')
    selected=news.calendar_decision({'2018-03':12},gap,target=10)
    check('later qualifying complete month allowed despite earlier gap',selected['selected_months']==['2018-03'])
    rejects('calendar negative counts rejected',lambda:news.calendar_decision({'2018-01':-1},coverage,target=10))
    rejects('calendar excludes calibration year',lambda:news.calendar_decision({'2017-12':100},coverage,target=10))

def convex_checks():
    from phase4 import convex
    from decimal import Decimal, localcontext
    def dec(value):
        return Decimal(value.numerator)/Decimal(value.denominator)
    values=[Fraction(k,8) for k in range(-48,49)]+[Fraction(0),Fraction(1,2**60),Fraction(-1,2**60),Fraction(1000),Fraction(-1000)]
    with localcontext() as context:
        context.prec=180
        for value in values:
            lower,upper,_=convex.sigmoid_interval(value)
            sigmoid=1/(1+(-dec(value)).exp())
            if not dec(lower)<=sigmoid<=dec(upper):raise AssertionError('Decimal sigmoid outside rational interval')
        check('rational sigmoid intervals contain high precision reference',True)
        x=np.array([[1.,-.5],[-.25,2.],[.5,.75]],np.float32);y=np.array([.25,.75,.5],np.float64);w=np.array([.37,-.14],np.float64)
        cert=convex.certify_logistic(x,y,.03,w)
        exact_x=[[Decimal.from_float(float(v)) for v in row] for row in x]
        exact_y=[Decimal.from_float(float(v)) for v in y];exact_w=[Decimal.from_float(float(v)) for v in w]
        gradient=[Decimal.from_float(.03)*v for v in exact_w]
        for row,target in zip(exact_x,exact_y):
            logit=sum(a*b for a,b in zip(row,exact_w));p=1/(1+(-logit).exp())
            for j in range(2):gradient[j]+=row[j]*(p-target)/3
        def fromjson(value):return Fraction(int(value['numerator']),int(value['denominator']))
        check('exact logistic gradient interval contains independent Decimal gradient',all(dec(fromjson(lo))<=g<=dec(fromjson(hi)) for lo,hi,g in zip(cert['gradient_lower'],cert['gradient_upper'],gradient)))
    empty=convex.certify_logistic(np.empty((0,2),np.float32),np.empty(0,np.float64),.03,w)
    check('empty logistic target has certified distance to zero',fromjson(empty['parameter_error_squared_upper']) == sum(Fraction(float(v))**2 for v in w))
    zero=convex.certify_logistic(np.array([[1]],np.float32),np.array([.5],np.float64),.1,np.array([0.],np.float64))
    check('half-label symmetric optimum has exactly zero certificate',fromjson(zero['parameter_error_squared_upper'])==0 and zero['meets_parameter_tolerance'])
    rejects('exact logistic coordinate budget fails explicitly',lambda:convex.certify_logistic(x,y,.03,w,max_coordinates=5),exceptions=(convex.CertificateBudgetError,))
    oldx=np.array([[1.,-.5],[-.25,2.]],np.float32);oldy=np.array([.25,.75],np.float64)
    newx=np.vstack([oldx[1:],x[2:]]);newy=np.array([.75,.5],np.float64)
    _,g,_=convex.objective_gradient_hessian(oldx,oldy,w,.03)
    signed=convex.signed_warm_gradient(g,w,.03,2,2,oldx[:1],oldy[:1],x[2:],y[2:])
    _,actual,_=convex.objective_gradient_hessian(newx,newy,w,.03)
    check('signed logistic gradient identity after removal and admission',np.max(np.abs(signed-actual))<1e-14)
    signed=convex.signed_warm_gradient(g,w,.03,2,1,oldx[:1],oldy[:1],np.empty((0,2),np.float32),np.empty(0,np.float64))
    _,actual,_=convex.objective_gradient_hessian(oldx[1:],oldy[1:],w,.03)
    check('signed logistic gradient identity includes changed-count regularizer term',np.max(np.abs(signed-actual))<1e-14)
    return {'independent_sigmoid_interval_cases':len(values),'high_precision_reference_digits':180,'certificate_scope':'stored-value optimization only; single output; no statistical unlearning claim'}

def convex_artifact_checks():
    from decimal import Decimal, localcontext
    directory=ROOT/'empirical_execution/phase4/results/convex_engineering_release_v2'
    natural=read_natural_jsonl(ROOT/'empirical_execution/data/civil_comments_engineering_preview.jsonl',require_labels=True)
    ids=[r['record_id'] for r in natural];lookup={r:i for i,r in enumerate(ids)};x,_=lexical_engineering_features(natural,64)
    y=np.array([r['label'] for r in natural],np.float64)
    rows={(r['trajectory_id'],r['checkpoint']):r for r in lines(directory/'checkpoints.jsonl')}
    heads={(r['trajectory_id'],r['checkpoint']):r['weights'] for r in lines(directory/'heads.jsonl')}
    certs=lines(directory/'certificates.jsonl');audit_rows=[]
    for entry in certs:
        key=(entry['trajectory_id'],entry['checkpoint'])
        audit_rows.append((rows[key]['selected_record_ids'],heads[key][entry['method']],entry['certificate']))
    initial=read(directory/'initial_fit.json')
    if 'weights' not in initial:raise AssertionError('Initial certified head must be persisted for independent replay')
    audit_rows.append((initial['selected_record_ids'],initial['weights'],initial['certificate']))
    fromjson=lambda value:Fraction(int(value['numerator']),int(value['denominator']))
    todecimal=lambda value:Decimal(value.numerator)/Decimal(value.denominator)
    def ahash(value):
        a=np.ascontiguousarray(value)
        return hashlib.sha256(str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes()).hexdigest()
    checked=0
    with localcontext() as context:
        context.prec=180
        for selected,weights,certificate in audit_rows:
            ix=[lookup[r] for r in selected];z=x[ix];target=y[ix];w=np.asarray(weights,np.float64);n=len(ix)
            if (ahash(z)!=certificate['input_bindings']['features_sha256'] or ahash(target)!=certificate['input_bindings']['targets_sha256'] or ahash(w)!=certificate['input_bindings']['weights_sha256']):raise AssertionError('Saved convex input binding mismatch')
            lam=fromjson(certificate['lambda_exact']);dw=[Decimal.from_float(float(v)) for v in w]
            gradient=[todecimal(lam)*v for v in dw]
            for row,label in zip(z,target):
                dx=[Decimal.from_float(float(v)) for v in row];dy=Decimal.from_float(float(label))
                score=sum(a*b for a,b in zip(dx,dw));prob=1/(1+(-score).exp())
                for j in range(len(w)):gradient[j]+=dx[j]*(prob-dy)/n
            lower=list(map(fromjson,certificate['gradient_lower']));upper=list(map(fromjson,certificate['gradient_upper']))
            if not all(todecimal(a)<=g<=todecimal(b) for a,g,b in zip(lower,gradient,upper)):raise AssertionError('Saved exact gradient enclosure misses high precision reference')
            sq=sum(max(abs(a),abs(b))**2 for a,b in zip(lower,upper));psq=sq/lam**2;tol=fromjson(certificate['parameter_tolerance'])
            if (sq!=fromjson(certificate['gradient_norm_squared_upper']) or psq!=fromjson(certificate['parameter_error_squared_upper']) or sq/(2*lam)!=fromjson(certificate['objective_gap_upper']) or fromjson(certificate['parameter_error_norm_upper'])**2<psq or certificate['meets_parameter_tolerance']!=(psq<=tol**2)):raise AssertionError('Saved certificate bound arithmetic mismatch')
            checked+=1
    check('saved convex gradient intervals and exact bound arithmetic independently checked',checked==25)
    return {'saved_certificates_checked':checked,'initial_head_saved': 'weights' in initial,'high_precision_reference_digits':180,'scope':'numerical independent containment crosscheck plus exact rational bound arithmetic; theorem proof reviewed separately'}

def systems_checks(tmp):
    from phase4 import systems
    spec={'evidence_role':'software_harness_validation','repetitions':1,'threads':1,'timeout_seconds':.2,'termination_grace_seconds':.1,'order_seed':17,'methods':[
        {'method_id':'success','argv':[sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).joinpath("three.bin").write_bytes(b"abc"); print("ok")','{output_dir}'],'input_files':[]},
        {'method_id':'failure','argv':[sys.executable,'-c','raise SystemExit(7)'],'input_files':[]},
        {'method_id':'timeout','argv':[sys.executable,'-c','import time; time.sleep(2)'],'input_files':[]}]}
    result=systems.run_benchmark(spec,tmp,tmp/'measured')
    byid={r['method_id']:r for r in result['runs']}
    check('fresh process harness preserves success exit failure and timeout',byid['success']['status']=='success' and byid['failure']['status']=='nonzero_exit' and byid['failure']['returncode']==7 and byid['timeout']['status']=='timeout')
    check('fresh process byte ledger includes artifacts and logs',byid['success']['method_artifact_logical_bytes']==3 and byid['success']['persistent_output_logical_bytes']==6)
    check('systems excludes failed timings without dropping rows',result['successful_complete_runs']==1 and result['failed_or_incomplete_runs']==2 and len(result['runs'])==3)
    check('systems keeps measured scope qualified',not result['paper_systems_result'] and not result['confirmatory_study_ready'] and not result['limits']['filesystem_or_network_security_sandbox'])

def admission_checks():
    directory=ROOT/'empirical_execution/phase4/results/human_admission_engineering_pack'
    manifest=read(directory/'private_sampling_manifest.json')
    assignments=read(directory/'blinded_assignments.json');bundle=read(directory/'responses.template.json')
    f=lambda value:Fraction(value['numerator'],value['denominator'])
    strata={tuple(row['stratum']):row for row in manifest['strata']}
    for pair in manifest['complete_frame']:
        absent=Fraction(1)
        for key in pair['strata']:
            row=strata[tuple(key)];population=row['population_size'];sample=row['sample_size']
            if pair['pair_id'] not in row['frame_pair_ids']:raise AssertionError('Pair missing from declared stratum')
            absent*=1-Fraction(sample,population)
        if f(pair['conditional_union_inclusion_probability']) != 1-absent:raise AssertionError('Conditional overlap inclusion probability mismatch')
    check('admission exact conditional union probabilities independently recomputed',True)
    selected=manifest['selected_pairs'];selected_ids={p['pair_id'] for p in selected}
    check('admission pairs deduplicated before three assignments',len(selected_ids)==len(selected) and len(assignments)==3*len(selected) and len({a['assignment_id'] for a in assignments})==len(assignments))
    forbidden={'arm','roles','score','similarity','label','labels','method','record_id','former_blocker_id','occurrences','provenance','source_unit_id'}
    check('admission visible assignments omit hidden evaluation metadata',all(not (set(a)&forbidden) for a in assignments))
    check('all human admission responses remain blank',len(bundle['responses'])==len(assignments) and all(r['human_completed'] is False and not r['meaning_relation'] and not r['annotator_id'] for r in bundle['responses']))
    check('admission dispatch and semantic readiness remain blocked',manifest['confirmatory_study_ready'] is False and manifest['human_dispatch_ready'] is False and manifest['completed_human_responses']==0)
    check('missing corpus quotas not reassigned',all(manifest['corpus_status'][name]=='missing_corpus_no_quota_reallocation' for name in ['askubuntu','cc_news']))
    return {'complete_pair_frame':len(manifest['complete_frame']),'unique_selected_pairs':len(selected),'blank_assignments':len(assignments),'completed_human_responses':0,'probability_scope':'conditional on frozen checkpoint union and one realized former-blocker draw per record'}

def threshold_checks():
    # Exhaust every binary label vector for many tied score configurations.
    count=0
    for scores in itertools.product((0.,1.,2.),repeat=4):
        for labels in itertools.product((0,1),repeat=4):
            s=np.array(scores);y=np.array(labels);got=ms._tag_threshold(s,y)
            if sum(labels)==0: expected=None
            else:
                choices=[]
                for tau in set(scores):
                    predicted=s>=tau;tp=int(np.sum(y[predicted]));f1=Fraction(2*tp,int(predicted.sum())+sum(labels))
                    choices.append((f1,tau))
                expected=max(choices)[1]
            if got['threshold'] != expected: raise AssertionError('Exhaustive F1 threshold mismatch')
            count+=1
    check('exhaustive tied-score F1 and strictest ties',True)
    class LabelTrap(dict):
        def get(self,key,default=None):
            if key=='original_fields':raise AssertionError('Test labels were accessed')
            return super().get(key,default)
    heldout=LabelTrap(partition='test',record_id='sealed')
    rejects('vocabulary rejects test before accessing labels',lambda:ms.select_tag_vocabulary([heldout],{}))
    rejects('lambda rejects test before accessing labels',lambda:ms.select_ridge_regularization(np.zeros((1,1),np.float32),[heldout],{}))
    return count

def model_selection_checks():
    from phase3.panels import source_unit
    rows=[]
    for i in range(9):
        fields={'site':'SOFTWARE_FIXTURE_NOT_REAL_SITE','OwnerUserId':i//2+1 if i<8 else None,
                'Tags':''.join('<fixture-tag-'+str(j)+'>' for j in range(20)) if i<8 else '<rare-only>'}
        row={'record_id':'SOFTWARE_FIXTURE_'+str(i),'partition':'calibration','original_fields':fields}
        row['source_unit_id']=source_unit(row,'askubuntu')[0];rows.append(row)
    p={'dataset_id':'askubuntu','evidence_role':'software_fixture_only','population_scope':'transient_nonempirical_fixture'}
    vocab=ms.select_tag_vocabulary(rows,p);y=ms.encode_tag_targets(rows,vocab)
    check('unknown owner record retained with zero-target output',y.shape==(9,20) and not y[-1].any() and vocab['unknown_singleton_rows_retained']==1)
    sealed_before=json.dumps(vocab,sort_keys=True)
    heldout=[{**r,'partition':'test'} for r in rows]
    check('heldout applies fixed vocabulary without altering lock', np.array_equal(ms.encode_heldout_tag_targets_for_evaluation(heldout,vocab),y) and json.dumps(vocab,sort_keys=True)==sealed_before)
    rejects('normal target encoder rejects test rows',lambda:ms.encode_tag_targets(heldout,vocab))
    rows=[]
    for i in range(13):
        fields={'publication_id':'SOFTWARE_FIXTURE','article_id':i//2 if i<12 else None,'toxicity':float((i*3)%7)/6}
        row={'record_id':f'SOFTWARE_FIXTURE_{i:02d}','partition':'calibration','original_fields':fields}
        row['source_unit_id']=source_unit(row,'civil_comments')[0];rows.append(row)
    p={'dataset_id':'civil_comments','evidence_role':'software_fixture_only','population_scope':'transient_nonempirical_fixture','encoder_id':'not_a_model_algebra_only','encoder_revision':'not_a_revision_algebra_only'}
    x=np.array([[i/7,(i*i)%5/3,1.] for i in range(13)],np.float32)
    locked=ms.select_ridge_regularization(x,rows,p)
    check('unknown singleton remains in CV with five genuine source groups',len(locked['record_ids'])==13 and locked['unknown_singleton_rows_retained']==1)
    folds=np.array(locked['record_folds']);y=np.array([r['original_fields']['toxicity'] for r in rows])[:,None]
    error=0.
    for candidate in locked['selection_ledger']:
        pred=np.empty_like(y)
        for fold in range(5):
            train=folds!=fold; val=~train;lam=candidate['lambda']
            augmented=np.vstack([x[train].astype(float),math.sqrt(lam*train.sum())*np.eye(3)])
            target=np.vstack([y[train],np.zeros((3,1))]);theta=np.linalg.lstsq(augmented,target,rcond=None)[0]
            pred[val]=x[val].astype(float)@theta
        error=max(error,abs(float(np.mean((pred-y)**2))-candidate['mean_squared_loss']))
    check('independent source-fold augmented least squares risk',error<1e-11)
    check('record order leaves task lock unchanged',locked==ms.select_ridge_regularization(x[::-1].copy(),rows[::-1],p))
    check('all-zero features lambda tie picks largest',ms.select_ridge_regularization(np.zeros_like(x),rows,p)['lambda']==.1)
    return {'maximum_independent_CV_MSE_difference':error,'rows_in_software_fixture':13,'unknown_rows_retained':1}

def execution_checks(directory):
    if not directory.exists(): return {'status':'not_available_for_review'}
    natural=read_natural_jsonl(ROOT/'empirical_execution/data/civil_comments_engineering_preview.jsonl',require_labels=True)
    ids=[r['record_id'] for r in natural];ix={r:i for i,r in enumerate(ids)}
    cx,_=lexical_engineering_features(natural,128); y=np.array([r['label'] for r in natural])[:,None]
    # Independent scalar score and narrative priority; no production graph/score.
    norm={}
    for rid,row in zip(ids,cx):
        row=[float(v) for v in row];total=0.
        for v in row: total+=v*v
        norm[rid]=[v/math.sqrt(total) for v in row]
    order=sorted(ids,key=lambda r:(hashlib.sha256(f'priority-v1|0|{r}'.encode()).digest(),r))
    blockers={r:set() for r in ids}
    for p,r in enumerate(order):
        for s in order[:p]:
            score=0.
            for a,b in zip(norm[r],norm[s]):score+=a*b
            if score>.6:blockers[r].add(s)
    initial={r for r,b in blockers.items() if not b};maxdiff=0.;headcount=0;checkpointcount=0
    for d in [64,768]:
        base=directory/f'd{d}'
        if not (base/'summary.json').exists():raise ValueError('Incomplete execution results')
        x,_=lexical_engineering_features(natural,d);x=x.astype(float)
        checkpoints=lines(base/'checkpoints.jsonl');heads={(r['trajectory_id'],r['checkpoint']):r for r in lines(base/'heads.jsonl')}
        lock=read(base/'execution_lock.json')
        check('execution role and no readiness d'+str(d),lock['confirmatory_study_ready'] is False and lock['evidence_role']=='engineering_nonconfirmatory')
        for entry in checkpoints:
            if entry['status']!='completed_engineering':raise AssertionError('Execution has incomplete rows')
            checkpointcount+=1;dead=set(entry['deleted_record_ids'])
            selected={r for r,b in blockers.items() if r not in dead and not (b-dead)}
            check('independent scalar selected IDs '+str(d)+'/'+entry['trajectory_id']+'/'+str(entry['checkpoint']),selected==set(entry['selected_record_ids']))
            saved=heads[(entry['trajectory_id'],entry['checkpoint'])]['weights']
            for method,value in saved.items():
                selected_ids=initial-dead if method=='B-F' else selected
                indices=[ix[r] for r in sorted(selected_ids)];z=x[indices];target=y[indices];n=len(indices)
                theta=z.T@np.linalg.solve(z@z.T+.01*n*np.eye(n),target) if n else np.zeros((d,1))
                diff=float(np.max(np.abs(np.asarray(value)-theta),initial=0));maxdiff=max(maxdiff,diff);headcount+=1
                if diff>1e-10:raise AssertionError('Independent dual head mismatch')
    check('all saved heads agree with independent dual ridge',True)
    return {'status':'passed','checkpoint_rows':checkpointcount,'saved_heads':headcount,'maximum_independent_dual_head_difference':maxdiff,'evidence_scope':'reused natural Civil100 lexical software integration only'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execution-dir',type=Path,default=ROOT/'empirical_execution/phase4/results/execution_engineering_final');args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='phase4_independent_',dir=ROOT/'tmp') as temp:
        parser_checks(Path(temp));systems_checks(Path(temp))
    embedding_checks();observations=request_checks();extension_checks();integrity_checks();calendar_checks();convex=convex_checks();convex_artifacts=convex_artifact_checks();admission=admission_checks();thresholds=threshold_checks();selection=model_selection_checks();execution=execution_checks(args.execution_dir)
    modules={p.name:filehash(p) for p in Path(__file__).parent.glob('*.py') if p.name not in {'audit_phase4_independent.py'}}
    report={'status':'passed','checks':CHECKS,'check_count':len(CHECKS),'exhaustive_F1_cases':thresholds,'independent_request_observations':observations,'model_selection':selection,'execution':execution,'convex':convex,'convex_artifacts':convex_artifacts,'admission':admission,'module_sha256':modules,'audit_code_sha256':filehash(__file__),'synthetic_empirical_data_created':False,'fixtures':'transient explicitly named software parser/algebra fixtures; not research data','primary_semantic_study_verified':False,'actual_human_ratings_verified':0}
    destination=Path(__file__).parent/'results/independent_phase4_audit.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'checks','module_sha256'}},indent=2))

if __name__=='__main__':main()
