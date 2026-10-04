#!/usr/bin/env python3
"""Independent Phase 5 release audit.

Algebraic fixtures exercise software contracts, never empirical evidence.
Natural replay artifacts are audited separately. This audit establishes no
semantic model provenance, human judgment, privacy or physical erasure.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, sys, tempfile
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'empirical_execution'))
CHECKS=[]

def check(name,truth):
    if not truth:raise AssertionError(name)
    CHECKS.append(name)

def rejects(name,fn,exceptions=(ValueError,RuntimeError,PermissionError)):
    try:fn()
    except exceptions:check(name,True)
    else:raise AssertionError(name+' unexpectedly accepted')

def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def independent_selected(blockers,owners,deleted):
    """Raw-neighbor oracle; no production state or eligibility helper used."""
    D=set(deleted)
    return [i for i,B in enumerate(blockers)
            if owners[i] not in D and all(owners[j] in D for j in B)]

def independent_eligible(blockers,owners,deleted,remaining):
    D=set(deleted)
    out=[]
    for i,B in enumerate(blockers):
        own=owners[i]
        surviving={owners[j] for j in B if owners[j] not in D}
        if own not in D and own not in surviving and len(surviving)<=remaining:out.append(i)
    return out

def independent_moments(x,y,selected):
    y=np.asarray(y,np.float64)
    if y.ndim==1:y=y[:,None]
    d=x.shape[1];c=y.shape[1]
    G=np.zeros((d,d),np.float64);H=np.zeros((d,c),np.float64)
    for i in selected:
        # Deliberately scalar independent accumulation, unlike batched BLAS.
        for j in range(d):
            for k in range(d):G[j,k]+=float(x[i,j])*float(x[i,k])
            for k in range(c):H[j,k]+=float(x[i,j])*float(y[i,k])
    return G,H,len(selected)

def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'empirical_execution/phase5').glob('*.py'))}

# Module-specific audit functions are added as producer interfaces freeze.

def multioutput_checks(tmp):
    from phase5 import convex_multioutput as cm
    from decimal import Decimal,localcontext
    x=np.array([[1.,-.5],[-.25,2.],[.5,.75]],np.float32)
    y=np.array([[.25,1.,0.],[.75,0.,0.],[.5,1.,0.]],np.float64)
    w=np.array([[.37,-.13,.7],[-.14,.55,-.2]],np.float64);lam=.03
    # Direct vectorized audit of the independently specified summed-output loss.
    s=x.astype(float)@w;p=1/(1+np.exp(-s))
    obj=float(np.mean(np.logaddexp(0,s)-y*s,axis=0).sum()+lam/2*np.sum(w*w))
    grad=x.astype(float).T@(p-y)/len(x)+lam*w
    got,g=cm.objective_gradient(x,y,w,lam)
    check('multioutput objective uses row average and output sum',abs(got-obj)<1e-14 and np.max(np.abs(g-grad))<1e-14)
    cert=cm.certify_multioutput(x,y,lam,w)
    square=cm.fraction(cert['parameter_error_frobenius_squared_upper'])
    check('multioutput aggregates squared per-output radii',square==sum((cm.fraction(c['parameter_error_squared_upper']) for c in cert['scalar_certificates']),Fraction(0)))
    check('multioutput Frobenius tolerance not separate-column shortcut',not cert['meets_parameter_tolerance'])
    with localcontext() as ctx:
        ctx.prec=120
        D=lambda f:Decimal(f.numerator)/Decimal(f.denominator)
        xx=[[Decimal.from_float(float(z)) for z in row] for row in x]
        yy=[[Decimal.from_float(float(z)) for z in row] for row in y]
        ww=[[Decimal.from_float(float(z)) for z in row] for row in w]
        ll=Decimal.from_float(lam)
        for c,col in enumerate(cert['scalar_certificates']):
            exact=[ll*ww[j][c] for j in range(2)]
            for i,row in enumerate(xx):
                logit=sum(row[j]*ww[j][c] for j in range(2));sig=1/(1+(-logit).exp())
                for j in range(2):exact[j]+=row[j]*(sig-yy[i][c])/3
            check('independent Decimal gradient enclosed output '+str(c),all(D(cm.fraction(lo))<=g<=D(cm.fraction(hi)) for lo,hi,g in zip(col['gradient_lower'],col['gradient_upper'],exact)))
    fit=cm.solve_multioutput(x,y,lam)
    fitcert=cm.certify_multioutput(x,y,lam,fit['weights'])
    check('multioutput fitted head independently bounded',fitcert['meets_parameter_tolerance'])
    _,old=cm.objective_gradient(x[:2],y[:2],w,lam)
    repaired=cm.signed_gradient(old,w,lam,2,1,x[:1],y[:1],x[:0],y[:0])
    _,direct=cm.objective_gradient(x[1:2],y[1:2],w,lam)
    check('multioutput changed-count gradient identity',np.max(np.abs(repaired-direct))<1e-14)
    empty=cm.solve_multioutput(x[:0],y[:0],lam)
    check('empty multioutput fit returns zero head',empty['weights'].shape==(2,3) and not empty['weights'].any())
    rejects('whole multioutput certificate coordinate cap enforced',lambda:cm.certify_multioutput(x,y,lam,w,max_coordinates=x.size*3-1),exceptions=(cm.scalar.CertificateBudgetError,))
    path=tmp/'multi_resume.json';cm.save_resume(path,fit['weights'],selected_ids=['a'],deleted_ids=['b'],input_bindings={'fixture':'algebra_only'})
    restored,_=cm.load_resume(path,input_bindings={'fixture':'algebra_only'},expected_selected_ids=['a'],expected_deleted_ids=['b'])
    check('multioutput warm-head roundtrip exact',np.array_equal(restored,fit['weights']))
    rejects('multioutput resumed inputs cannot change',lambda:cm.load_resume(path,input_bindings={'fixture':'different'},expected_selected_ids=['a'],expected_deleted_ids=['b']))
    return {'scope':'algebraic software controls only; no empirical Stack labels invented','outputs':3,'independent_decimal_precision':120}

def graph_fixture(blockers,owners):
    from ccu.core import BlockerGraph
    ids=tuple('r'+str(i) for i in range(len(blockers)));ptr=[0];idx=[]
    for b in blockers:idx.extend(sorted(b));ptr.append(len(idx))
    return BlockerGraph(ids,tuple(owners),np.arange(len(ids),dtype=np.int64),np.asarray(ptr,np.int64),np.asarray(idx,np.int64),.5)

def polynomial_oracle(x,y,blockers,owners,deleted,horizon,unit_ids):
    # Integer code monomials are constructed directly from row contributions.
    lookup={u:i for i,u in enumerate(unit_ids)};D=set(deleted);out={}
    tri=np.triu_indices(x.shape[1]);y=np.asarray(y,np.float64)
    for i,B in enumerate(blockers):
        if owners[i] in D:continue
        b={owners[j] for j in B}-D
        if owners[i] in b or len(b)>horizon:continue
        z=x[i].astype(np.float64)
        v=np.concatenate([np.outer(z,z)[tri],np.outer(z,y[i]).ravel(),[1.]])
        for names,sign in [(b,1),(b|{owners[i]},-1)]:
            if len(names)>horizon:continue
            key=tuple(sorted(lookup[u] for u in names))
            out[key]=out.get(key,np.zeros_like(v))+sign*v
    return out

def service_checks(tmp):
    from phase5.payload import PayloadState
    from phase5.methods import SummaryService
    x=np.array([[1,.5,-.25],[.25,-1,2],[-.5,.75,1],[1.5,.125,-1],[2,.5,.75],[-1,1.25,.5]],np.float32)
    y=np.array([[.25,0],[.75,1],[.5,.5],[1,0],[0,1],[.125,.25]],np.float64)
    cases=[([[],[0],[1],[0,1],[0,2],[1,3,4]],['s0','s0','s1','unknown:3','s2','s2']),
           ([[],[],[0,1],[0,2],[1,2,3],[0,1,2,3,4]],['s0','s1','s1','s2','unknown:4','s3']),
           ([[],[],[],[],[],[]],['s0','s0','s1','s1','s2','unknown:5'])]
    states=0;maxerr=0.
    for ci,(blockers,sources) in enumerate(cases):
        graph=graph_fixture(blockers,sources)
        for unit in ['record','source']:
            owners=list(graph.record_ids) if unit=='record' else sources
            unit_ids=sorted(set(owners))
            for h in range(0,min(3,len(unit_ids))+1):
                for k in range(h+1):
                    for deleted in itertools.combinations(unit_ids,k):
                        selected=independent_selected(blockers,owners,deleted)
                        expect=independent_moments(x,y,selected)
                        eligible=independent_eligible(blockers,owners,deleted,h-k)
                        for method in ['B-E','B-A','P-I','P-S','P-R']:
                            state=(PayloadState(graph,x,y,h,unit=unit,retain_all=method=='B-A',compaction_fraction=.5,batch_rows=2)
                                   if method in ['B-E','B-A'] else SummaryService(graph,x,y,h,unit=unit,method=method))
                            state.delete(deleted)
                            actual=state.moments();err=max(np.max(np.abs(actual.gram-expect[0])),np.max(np.abs(actual.cross-expect[1])))
                            maxerr=max(maxerr,float(err))
                            if actual.count!=expect[2] or err>2e-12:raise AssertionError((ci,unit,h,deleted,method,'moments',err))
                            if method in ['B-E','B-A']:
                                if set(state.selected_ids())!={graph.record_ids[i] for i in selected}:raise AssertionError('payload selected membership')
                                live=[i for i in range(len(x)) if owners[i] not in deleted] if method=='B-A' else eligible
                                if set(state.logical_membership())!={graph.record_ids[i] for i in live}:raise AssertionError('payload eligibility')
                                state.check_invariants(check_moments=True)
                            else:
                                coefficient=state.coefficients()
                                desired=polynomial_oracle(x,y,blockers,owners,deleted,h-k,state.unit_ids)
                                for key in set(coefficient)|set(desired):
                                    if np.max(np.abs(coefficient.get(key,np.zeros(state.packed_dimension))-desired.get(key,np.zeros(state.packed_dimension))))>2e-12:
                                        raise AssertionError((ci,unit,h,deleted,method,'coefficient',key))
                            states+=1
            check('record/source/horizon exhaustive states case '+str(ci)+' '+unit,True)
    # Snapshot continued execution, duplicate retries, and atomic malformed input.
    blockers,sources=cases[0];graph=graph_fixture(blockers,sources)
    for method in ['B-E','B-A','P-I','P-S','P-R']:
        state=(PayloadState(graph,x,y,3,unit='source',retain_all=method=='B-A',compaction_fraction=1.,batch_rows=2)
               if method in ['B-E','B-A'] else SummaryService(graph,x,y,3,unit='source',method=method))
        state.delete(['s0']);before=state.moments();retry=state.delete(['s0','s0'])
        check(method+' duplicate retries consume no horizon',retry['fresh_deletions']==0 and np.array_equal(before.gram,state.moments().gram))
        rejects(method+' unknown units rejected atomically',lambda:state.delete(['s1','not-in-universe']))
        check(method+' invalid batch keeps state unchanged',np.array_equal(before.gram,state.moments().gram) and before.count==state.moments().count)
        p=tmp/(method+'.npz');state.snapshot(p);restored=type(state).load(p)
        restored.delete(['s1']);state.delete(['s1'])
        check(method+' save resume continued moments exact',np.array_equal(restored.moments().gram,state.moments().gram) and np.array_equal(restored.moments().cross,state.moments().cross))
        if method in ['B-E','B-A']:
            check(method+' no constructor payload memory alias',not np.shares_memory(state.x,x) and not np.shares_memory(state.y,y))
        else:
            check(method+' stored summary has no raw graph or payload attribute',not any(k in vars(state) for k in ['features','targets','graph','x','y']))
    return {'algebraic_states_checked':states,'max_independent_moment_absolute_error':maxerr,'scope':'software algebra fixtures only; no empirical source labels'}

def exact_ridge(x,y,selected,lam):
    d=x.shape[1];c=y.shape[1]
    if not selected:return [[Fraction(0) for _ in range(c)] for _ in range(d)]
    Q=lambda v:Fraction(float(v));n=len(selected)
    a=[[sum((Q(x[i,j])*Q(x[i,k]) for i in selected),Fraction(0))+(Q(lam)*n if j==k else 0) for k in range(d)] for j in range(d)]
    b=[[sum((Q(x[i,j])*Q(y[i,k]) for i in selected),Fraction(0)) for k in range(c)] for j in range(d)]
    aug=[aa+bb for aa,bb in zip(a,b)]
    for j in range(d):
        pivot=aug[j][j]
        aug[j]=[v/pivot for v in aug[j]]
        for k in range(d):
            if k==j:continue
            factor=aug[k][j];aug[k]=[u-factor*v for u,v in zip(aug[k],aug[j])]
    return [row[d:] for row in aug]

def boundary_checks():
    from phase5 import boundary as b
    from phase3.reference_graph import build_reference_graph
    tiny=np.nextafter(np.float32(0),np.float32(1));large=np.finfo(np.float32).max
    matrices=[np.array([[1,.5,-.25],[1,.5,-.25],[-1,-.5,.25],[.5,-.5,.5]],np.float32),
              np.array([[tiny,0,tiny],[0,tiny,-tiny],[tiny,-tiny,0]],np.float32),
              np.array([[large,large,-large],[large,-large,large],[large,0,0]],np.float32)]
    total=0
    for case,x in enumerate(matrices):
        normalized=[]
        for row in x:
            squares=0.
            for v in row:squares=squares+float(v)*float(v)
            norm=math.sqrt(squares);normalized.append([float(v)/norm for v in row])
        for i,j,lo,hi in b.score_intervals(x):
            score=0.
            for a,c in zip(normalized[i],normalized[j]):score=score+a*c
            if not lo<=score<=hi:raise AssertionError(('independent interval scorer escaped',case,i,j,lo,score,hi))
            total+=1
        check('uniform pair interval includes independent ordered scorer '+str(case),True)
    x=matrices[0];ids=['a','b','c','d'];g=build_reference_graph(x,ids,.8)
    candidates,config=b.lsh_candidates(x,tables=1,bits=5,seed=13)
    sample,sm=b.missed_pair_sample(4,candidates,size=100,seed=14)
    universe=set(itertools.combinations(range(4),2))
    check('ANN retrieved plus nonretrieved audit frames exactly partition all pairs',set(sample)==universe-candidates and not set(sample)&candidates)
    check('ANN full nonretrieved census carries exact inclusion',sm['inclusion_probability']==1 if sample else sm['empty_population'])
    sample,sm=b.missed_pair_sample(4,set(),size=2,seed=14)
    check('ANN nonretrieved SRS retains known full population probability',len(set(sample))==2 and sm['population_size']==6 and sm['inclusion_probability_numerator']==2 and sm['inclusion_probability_denominator']==6)
    approximate,report=b.audit_ann(x,g,tables=1,bits=5,seed=13,sample_size=100)
    check('ANN census missed edge estimate equals actual',report['metrics']['estimated_missed_edges_HT']==report['metrics']['actual_missed_edges'])
    lower,upper,report=b.graph_envelope(x,g)
    for k in range(5):
        for D in itertools.combinations(ids,k):
            lo=set(upper.selected_indices(D));truth=set(g.selected_indices(D));hi=set(lower.selected_indices(D))
            if not lo<=truth<=hi:raise AssertionError('selection containment failed')
    check('graph interval selection sandwich survives every subset deletion',True)
    # Exact rational comparison against all optional subsets, including empty.
    xx=np.array([[1.,.5],[-.25,1.],[.5,-.75]],np.float32)
    yy=np.array([[.25,1.],[.75,0.],[.5,.5]],np.float64)
    w=np.array([[.3,-.1],[.15,.2]],np.float64);lam=.07;subsets=0
    for minimum in [set(),{0},{0,1}]:
        maximum={0,1,2};r=b.model_envelope_bound(xx,yy,w,minimum,maximum,lam)
        radius=Fraction(int(r['radius_numerator']),int(r['radius_denominator']))
        optional=sorted(maximum-minimum)
        for k in range(len(optional)+1):
            for extras in itertools.combinations(optional,k):
                S=minimum|set(extras);opt=exact_ridge(xx,yy,S,lam)
                error=sum((Fraction(float(w[j,c]))-opt[j][c])**2 for j in range(2) for c in range(2))
                if error>radius**2:raise AssertionError('exact rational graph-model radius failed')
                subsets+=1
    check('model envelope exact rational radius covers every optional set',True)
    return {'pair_intervals_checked':total,'exact_rational_model_subset_checks':subsets,'scope':'algebraic software tests only; conditional IEEE arithmetic theorem assumptions'}

def refit_checks(tmp):
    from phase5 import refit as r
    manifest=r.vendor_integrity()
    check('official SemDeDup vendor matches ten pinned git blobs',len(manifest['files'])==10 and manifest['commit']==r.UPSTREAM_COMMIT)
    status=r.backend_status()
    if not status['official_backend_available']:
        rejects('missing official backend fails without reference fallback',r._official_kernels)
        check('official backend not executed by dependency inspection',status['official_runtime_executed'] is False)
    # Algebraic direction/tie fixture; not natural data and not SemDeDup evidence.
    x=np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0],[0,1]],np.float32)
    ids=[str(i) for i in range(len(x))];sources=ids
    cfg=r.Config(clusters=2,iterations=4,epsilon=.2,seed=23,backend=r.REFERENCE,development_config_id='independent_algebra_software_only')
    fitted=r.fit_curator(x,ids,cfg,tmp/'reference_fit')
    graph=r.initial_graph(x,ids,sources,fitted,cfg)
    check('refit owns its branch-specific original selection',set(fitted['selected_ids'])=={ids[i] for i in graph.selected_indices()})
    comparisons=0
    for k in range(len(x)+1):
        for dead in itertools.combinations(ids,k):
            selected,_=r.frozen_fitted_selection(x,ids,fitted,dead,cfg)
            # independently scan original fitted within-cluster order.
            expected=[]
            for order in fitted['orders']:
                alive=[i for i in order if i not in dead]
                for j,identifier in enumerate(alive):
                    i=ids.index(identifier)
                    blocked=any(float(x[i,0])*float(x[ids.index(early),0])+float(x[i,1])*float(x[ids.index(early),1])>1-cfg.epsilon for early in alive[:j])
                    if not blocked:expected.append(identifier)
            if set(expected)!=set(selected):raise AssertionError('frozen-fit differs from raw-neighbor within-cluster oracle')
            comparisons+=1
    check('frozen fitted target preserves original raw-neighbor cluster order for every deletion set',True)
    rejects('refit cannot silently reduce locked cluster count',lambda:r.fit_curator(x[:1],ids[:1],cfg,tmp/'too_few'))
    rejects('refit cannot reuse stale centroid directory',lambda:r.fit_curator(x,ids,cfg,tmp/'reference_fit'),exceptions=(FileExistsError,))
    repeat=r.fit_curator(x,ids,cfg,tmp/'reference_repeat')
    check('reference same-seed selected identities repeat',fitted['selected_ids']==repeat['selected_ids'])
    return {'pinned_upstream_files_verified':len(manifest['files']),'frozen_selection_subset_checks':comparisons,'official_backend_available':status['official_backend_available'],'official_runtime_executed_by_audit':False,'reference_scope':'software orchestration only; reference is not Faiss-equivalent'}

def confinement_checks(tmp):
    import subprocess
    private=tmp/'private';private.mkdir();outside=tmp/'forbidden';outside.write_text('audit sentinel, no corpus data')
    (private/'escape_link').symlink_to(outside)
    code='''import sys,os,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from phase5.workers import confine_files
private=Path(sys.argv[2]);outside=Path(sys.argv[3]);os.chdir(private)
# Imported code has no corpus object. This is a filesystem capability test.
old=os.open(outside,os.O_RDONLY)
for name in os.listdir('/proc/self/fd'):
 fd=int(name)
 if fd>=3:
  try:os.close(fd)
  except OSError:pass
info=confine_files(private)
checks={}
for label,path in [('outside',outside),('symlink',private/'escape_link'),('proc_fd',Path('/proc/self/fd')/str(old))]:
 try:
  with path.open('rb') as f:f.read(1)
 except PermissionError:checks[label]=True
 except FileNotFoundError:checks[label]=label=='proc_fd'
 else:checks[label]=False
try:os.read(old,1)
except OSError:checks['preopened_closed']=True
else:checks['preopened_closed']=False
(private/'allowed.txt').write_text('allowed')
checks['allowed_private_read']=(private/'allowed.txt').read_text()=='allowed'
print(json.dumps({'checks':checks,'isolation':info}))
'''
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'empirical_execution'),str(private),str(outside)],capture_output=True,text=True,timeout=30)
    if result.returncode:
        if 'Landlock ABI >=3 required; got -1, errno=38' in result.stderr:
            check('unsupported kernel confinement fails closed',True)
            return {'kernel_confinement_available':False,'actual_denied_read_tests_executed':False,
                    'status':'blocked_kernel_ENOSYS','failure_stderr':result.stderr,
                    'no_reaccess_claim_established':False}
        raise AssertionError('Independent Landlock probe failed: '+result.stderr)
    value=json.loads(result.stdout)
    for name,ok in value['checks'].items():check('kernel filesystem confinement '+name,ok)
    check('Landlock scope excludes IPC and physical erasure',value['isolation']['network_or_IPC_isolation_claim'] is False and value['isolation']['physical_erasure_claim'] is False)
    return value

def natural_inputs():
    from ccu.data import read_natural_jsonl,lexical_engineering_features
    p=ROOT/'empirical_execution/data/civil_comments_engineering_preview.jsonl'
    records=read_natural_jsonl(p,require_labels=True)
    x,_=lexical_engineering_features(records,64);curator,_=lexical_engineering_features(records,128)
    return records,x,curator

def natural_convex_artifacts():
    from decimal import Decimal,localcontext
    from phase5.convex_multioutput import fraction
    records,x,curator=natural_inputs();ids=[r['record_id'] for r in records];lookup={r:i for i,r in enumerate(ids)}
    root=ROOT/'empirical_execution/phase5/results/convex_multioutput_engineering';lock=read(root/'design_lock.json')
    y=np.array([[r['fields'][t] for t in lock['targets']] for r in records],np.float64)
    init=read(root/'initial_fit.json');rows=read(root/'checkpoints.json');heads={(r['arm'],r['checkpoint']):r for r in read(root/'heads.json')}
    examples=[(init['selected_ids'],np.array(init['weights'],np.float64),init['certificate'])]
    for row in rows:
        for method in ['cold','warm']:examples.append((row['selected_ids'],np.array(heads[(row['arm'],row['checkpoint'])][method],np.float64),row['certificates'][method]))
    ahash=lambda a:hashlib.sha256(str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+np.ascontiguousarray(a).tobytes()).hexdigest()
    scalar_count=0
    with localcontext() as ctx:
        ctx.prec=120
        D=lambda v:Decimal(v.numerator)/Decimal(v.denominator)
        for selected,w,certificate in examples:
            ix=[lookup[r] for r in selected];xx=x[ix];yy=y[ix];n=len(ix)
            if certificate['input_bindings']!={'features':ahash(xx),'targets':ahash(yy),'weights':ahash(w)}:raise AssertionError('natural multioutput certificate input binding mismatch')
            xd=[[Decimal.from_float(float(v)) for v in row] for row in xx]
            yd=[[Decimal.from_float(float(v)) for v in row] for row in yy]
            wd=[[Decimal.from_float(float(v)) for v in row] for row in w]
            lam=Decimal.from_float(lock['lambda_reg'])
            for c,cert in enumerate(certificate['scalar_certificates']):
                g=[lam*wd[j][c] for j in range(w.shape[0])]
                for i,row in enumerate(xd):
                    s=sum(row[j]*wd[j][c] for j in range(len(row)));p=1/(1+(-s).exp())
                    for j,v in enumerate(row):g[j]+=v*(p-yd[i][c])/n
                if not all(D(fraction(lo))<=v<=D(fraction(hi)) for lo,hi,v in zip(cert['gradient_lower'],cert['gradient_upper'],g)):
                    raise AssertionError('saved natural certificate misses independent Decimal gradient')
                scalar_count+=1
            square=sum((fraction(c['parameter_error_squared_upper']) for c in certificate['scalar_certificates']),Fraction(0))
            if square!=fraction(certificate['parameter_error_frobenius_squared_upper']) or square>fraction(certificate['parameter_tolerance'])**2:raise AssertionError('saved joint certificate aggregation invalid')
    check('all natural multioutput saved scalar certificates contain independent high-precision gradient',True)
    check('all natural multioutput certificate inputs and Frobenius aggregations bound',True)
    check('multioutput evidence remains lexical and native-source unavailable',not lock['semantic_encoder_used'] and not lock['source_claim'] and not lock['Stack20_actual_data_used'])
    return {'saved_matrix_certificates':len(examples),'saved_scalar_gradient_certificates':scalar_count,'independent_Decimal_digits':120,'primary_semantic_result':False}

def natural_refit_artifacts():
    records,x,curator=natural_inputs();ids=[r['record_id'] for r in records];lookup={r:i for i,r in enumerate(ids)}
    y=np.array([r['label'] for r in records],np.float64)[:,None]
    root=ROOT/'empirical_execution/phase5/results/refit_engineering_final/natural';lock=read(root/'lock.json');rows=lines(root/'checkpoints.jsonl');initial=read(root/'original_head.json')
    lam=lock['lambda_reg'];cases=[(initial['selected_ids'],initial['weights'])]
    cases += [(selected,row['weights'][method]) for row in rows if row['status']=='completed' for method,selected in row['selected_ids'].items()]
    error=0.
    for selected,weights in cases:
        ix=[lookup[r] for r in selected];z=x[ix].astype(np.float64)
        # Dual solution independent of producer primal moment/Cholesky code.
        expected=z.T@np.linalg.solve(z@z.T+lam*len(z)*np.eye(len(z)),y[ix]) if len(z) else np.zeros((x.shape[1],1))
        error=max(error,float(np.max(np.abs(expected-np.asarray(weights)))))
    check('refit saved branch-specific heads match independent dual solver',error<1e-10)
    check('refit evidence does not claim official backend or semantic confirmation',lock['backend_requested']=='separate_numpy_spherical_lloyd_reference' and not lock['official_backend_executed'] and not lock['primary_study'])
    repeats=read(root/'seed_audit.json')
    check('refit no-deletion audits retain five fixed and three alternative seed rows',len(repeats)==8 and sum(r['same_declared_seed'] for r in repeats)==5)
    return {'saved_heads_independently_checked_including_initial':len(cases),'max_dual_solver_absolute_error':error,'official_SemDeDup_execution':False}

def independent_natural_graph(curator,ids,threshold=.6,seed=0):
    normals=[]
    for row in curator:
        sq=0.
        for v in row:sq+=float(v)*float(v)
        root=math.sqrt(sq);normals.append([float(v)/root for v in row])
    priority=sorted(ids,key=lambda r:(hashlib.sha256((f'priority-v1|{seed}|'+r).encode()).digest(),r));rank={r:i for i,r in enumerate(priority)}
    blockers={r:set() for r in ids};scores={}
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            score=0.
            for a,b in zip(normals[i],normals[j]):score+=a*b
            scores[(ids[i],ids[j])]=scores[(ids[j],ids[i])]=score
            if score>threshold:
                late,early=(ids[i],ids[j]) if rank[ids[i]]>rank[ids[j]] else (ids[j],ids[i])
                blockers[late].add(early)
    return blockers,scores

def natural_context_artifacts():
    from phase5.context_audit import verify_context_manifest
    from phase4.requests import generate_manifest
    from phase3.reference_graph import build_reference_graph
    records,x,cx=natural_inputs();ids=[r['record_id'] for r in records];texts={r['record_id']:r['text'] for r in records}
    root=ROOT/'empirical_execution/phase5/results/context_audit_engineering/blank_pack'
    m=read(root/'private_sampling_manifest.json');verify_context_manifest(m)
    blockers,scores=independent_natural_graph(cx,ids);initial={r for r in ids if not blockers[r]}
    graph=build_reference_graph(cx,ids,.6,block_size=17)
    requests=generate_manifest(graph,dataset_id='civil_comments_engineering_preview',panel_id='all100_lexical128_human_pack_software_check',source_kinds={s:'unknown_singleton' for s in graph.source_ids},design={'requests':{'record_horizon':8,'record_checkpoints':[1,2,4,8]}},allocations={'R':8,'S':0,'U':8,'A':4},include_excluded_blocker_stress=False)
    check('context request frame regenerated with frozen exact manifest',requests['manifest_sha256']==m['input_bindings']['civil_comments']['request_manifest_sha256'])
    admitted=set()
    for path in requests['trajectories']:
        for k in path['checkpoints']:
            dead=set(path['deletion_order'][:k]);chosen={r for r in ids if r not in dead and blockers[r]<=dead}
            admitted|=chosen-initial
    for kind in ['former_neighborhood','nearest_surviving_selected']:
        check('context complete unique-admission frame '+kind,{u['record_id'] for u in m['complete_frame'] if u['kind']==kind}==admitted)
    count=0
    for u in m['complete_frame']:
        if not u['context_prepared']:continue
        state=m['checkpoint_states'][u['checkpoint_state_key']];D=set(state['deleted_record_ids']);S={r for r in ids if r not in D and blockers[r]<=D};r=u['record_id']
        if set(state['retained_selected_ids'])!=S or set(u['complete_former_blocker_ids'])!=blockers[r] or not blockers[r]<=D:raise AssertionError('context state has wrong actual former neighborhood')
        if u['kind']=='former_neighborhood':
            if set(u['complete_context_ids'])!=blockers[r]:raise AssertionError('missing former blocker context')
        else:
            others=sorted(S-{r});best=min(others,key=lambda v:(-scores[(r,v)],v)) if others else None
            if u['nearest_record_id']!=best or (best is not None and u['nearest_reference_score']!=scores[(r,best)]):raise AssertionError('nearest surviving selected mismatch')
        count+=1
    check('all prepared contexts match independent raw-neighbor and nearest-text oracle',True)
    for stratum in m['strata']:
        N=stratum['population_size'];n=stratum['sample_size'];pr=stratum['inclusion_probability'];actual=Fraction(pr['numerator'],pr['denominator'])
        if actual!=(Fraction(n,N) if N else 0) or stratum['shortfall']!=stratum['quota']-n:raise AssertionError('context inclusion/shortfall mismatch')
    check('context exact inclusion probabilities and unfilled corpus quotas preserved',True)
    assignments=read(root/'blinded_assignments.json');responses=read(root/'responses.template.json')['responses']
    allowed={'assignment_id','unit_id','target_text','comparison_texts','task_definition','comparison_question','context_complete','comparison_available'}
    check('context assignment export conceals methods arms scores labels and native IDs',all(set(a)==allowed for a in assignments))
    check('context human responses remain genuinely blank',len(responses)==len(assignments) and all(not r['human_completed'] and not r['annotator_id'] and not r['distinct_information'] for r in responses))
    check('context pack cannot imply dispatch or semantic readiness',not m['human_dispatch_ready'] and not m['confirmatory_study_ready'] and m['completed_human_responses']==0)
    return {'unique_natural_admissions':len(admitted),'prepared_contexts_checked':count,'blank_assignments':len(assignments),'human_responses':0,'semantic_quality_established':False}

def statistics_checks():
    from phase5 import statistics as s
    y=np.array([[0,1],[1,0],[1,1],[0,0]],float)
    before=np.array([[.1,.8],[.9,.3],[.7,.75],[.2,.1]],float)
    oracle=np.empty((2,2,4,2));frozen=np.empty_like(oracle)
    for t in range(2):
        for k in range(2):
            oracle[t,k]=before+(t+1)*.02+(k+1)*.01
            frozen[t,k]=oracle[t,k]+np.array([[.08,-.03],[-.05,.02],[.04,.1],[.03,-.07]])*(t+1)*(k+1)
    got=s.prediction_metrics(y,before,frozen[0,0],oracle[0,0],thresholds=[.5,.5])
    variance=np.mean(np.var(before,axis=0))
    check('reported prediction normalizer matches mean output population variance',abs(got['predelete_score_sd']-math.sqrt(variance))<1e-15)
    zero=s.prediction_metrics(np.zeros((2,1)),np.zeros((2,1)),np.ones((2,1)),np.zeros((2,1)))
    check('zero normalizers remain undefined without favorable floor',zero['signed_normalized_loss'] is None and zero['normalized_prediction_rms'] is None)
    cells=[dict(dataset_id='software',panel_id='p',configuration_id='c',arm='R',trajectory_id=t,checkpoint=k) for t in ['t1','t2'] for k in [1,2]]
    reg=s.lock_registry(cells,replicates=100);values=[dict(**r,status='success',metrics={'effect':(1 if r['trajectory_id']=='t1' else 3)*r['checkpoint']}) for r in cells]
    summary=s.summarize_registry(reg,values,metrics=['effect']);groups=summary['groups']
    check('whole-path bootstrap uses shared checkpoint draws',np.allclose(np.asarray(groups[1]['metrics']['effect']['interval_95']),2*np.asarray(groups[0]['metrics']['effect']['interval_95'])))
    missing=s.summarize_registry(reg,values[:-1],metrics=['effect'])
    check('missing planned trajectory remains missing and full estimand unidentified',missing['groups'][1]['statuses']['missing']==1 and missing['groups'][1]['metrics']['effect']['mean'] is None and missing['groups'][1]['metrics']['effect']['interval_95'] is None)
    rejects('unplanned analysis cells refused',lambda:s.reconcile_registry(reg,values+[dict(values[0],trajectory_id='other')]))
    check('statistical registry cannot authenticate preoutcome freeze',reg['external_preoutcome_timing_verified'] is False)
    paired_rows=[dict(**r,status='success',metrics={'a':(1 if r['trajectory_id']=='t1' else 9)*r['checkpoint'],'b':(1 if r['trajectory_id']=='t1' else 3)*r['checkpoint']}) for r in cells]
    paired=s.paired_method_bootstrap(reg,paired_rows,method_metric='a',baseline_metric='b')
    check('paired method estimate is ratio of means rather than mean ratios',all(g['ratio_of_means']['estimate']==2.5 for g in paired['groups']))
    arm=s.equal_rs_lifecycle_bootstrap([0.,2.],[10.],replicates=100,seed=19)
    check('secondary R/S lifecycle target gives arms equal weight despite unequal counts',arm['equal_RS_mean']==5.5 and not arm['arms_paired'])
    reps=120;seed=17
    cross=s.crossed_source_bootstrap(y,before,frozen,oracle,trajectory_ids=['a','b'],checkpoint_ids=[1,2],test_source_ids=['s1','s1','s2','s2'],source_kinds={'s1':'software_fixture_only','s2':'software_fixture_only'},seed=seed,replicates=reps,thresholds=[.5,.5],software_fixture_only=True)
    rng=np.random.default_rng(seed);boot=np.empty((reps,2,4))
    for b in range(reps):
        tr=np.repeat(np.arange(2),rng.multinomial(2,[.5,.5]))
        sw=rng.multinomial(2,[.5,.5]);rows=np.repeat(np.arange(4),np.repeat(sw,2))
        yy=y[rows];bb=before[rows];base=np.mean((bb-yy)**2);sd=np.sqrt(np.mean(np.var(bb,axis=0)))
        for k in range(2):
            measures=[]
            for t in tr:
                f=frozen[t,k,rows];o=oracle[t,k,rows]
                Z=(np.mean((f-yy)**2)-np.mean((o-yy)**2))/base
                D=np.sqrt(np.mean((f-o)**2));f1=[]
                for a in [f,o]:
                    p=a>=.5;tp=np.sum(p&(yy==1),axis=0);den=np.sum(p,axis=0)+np.sum(yy==1,axis=0)
                    f1.append(np.mean(np.divide(2*tp,den,out=np.zeros(2),where=den!=0)))
                measures.append([Z,D,D/sd,f1[0]-f1[1]])
            boot[b,k]=np.mean(measures,axis=0)
    names=['signed_normalized_loss','prediction_rms','normalized_prediction_rms','macro_f1_difference']
    for k in range(2):
        for j,name in enumerate(names):
            expected=np.quantile(boot[:,k,j],[.025,.975]);actual=cross['groups'][k]['metrics'][name]['interval_95']
            check('crossed whole-source explicit row-resampling agreement '+str(k)+' '+name,np.max(np.abs(expected-actual))<2e-14)
    values={'civil_comments:R':.001,'civil_comments:S':.02,'askubuntu:R':.03,'askubuntu:S':None}
    holm=s.holm_family(values,family='signed_loss')
    check('fixed Holm family retains missing test and step-down monotonicity',[holm['tests'][k]['adjusted_p_value'] for k in values]==[.004,.06,.06,1.])
    return {'crossed_explicit_replicates':reps,'conditional_bootstrap_scope_verified':True,'actual_primary_empirical_statistics':False}

def seccomp_checks(tmp):
    import subprocess
    private=tmp/'seccomp_private';private.mkdir();outside=tmp/'seccomp_forbidden';outside.write_text('independent sentinel')
    (private/'escape_link').symlink_to(outside)
    code='''import sys,os,json,ctypes,errno,threading
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from phase5.workers import confine_open_syscalls
private=Path(sys.argv[2]);outside=Path(sys.argv[3]);os.chdir(private)
libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
old=os.open(outside,os.O_RDONLY)
trigger=threading.Event();secondary={}
def other_thread():
 trigger.wait()
 try:fd=os.open(outside,os.O_RDONLY)
 except PermissionError:secondary['other_preexisting_thread_denied']=True
 else:secondary['other_preexisting_thread_denied']=False;os.close(fd)
thread=threading.Thread(target=other_thread);thread.start()
for name in os.listdir('/proc/self/fd'):
 fd=int(name)
 if fd>=3:
  try:os.close(fd)
  except OSError:pass
out=os.open(private/'allowed.txt',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
info=confine_open_syscalls();checks={}
trigger.set();thread.join(timeout=5);checks.update(secondary)
checks['other_preexisting_thread_finished']=not thread.is_alive()
for label,path in [('existing_original_path',outside),('symlink_to_original',private/'escape_link'),('proc_fd_reopen',Path('/proc/self/fd')/str(out))]:
 try:fd=os.open(path,os.O_RDONLY)
 except PermissionError:checks[label]=True
 else:checks[label]=False;os.close(fd)
try:os.read(out,1)
except OSError as e:checks['output_fd_not_readable']=e.errno==errno.EBADF
else:checks['output_fd_not_readable']=False
for nr in [2,85,257,437,304,41,425,426,427,101,310,311,438,59,322,0x40000002]:
 ctypes.set_errno(0);ret=libc.syscall(nr,0,0,0,0,0,0)
 checks['denied_syscall_'+str(nr)]=ret==-1 and ctypes.get_errno()==errno.EACCES
os.write(out,b'only output');os.fsync(out);os.close(out)
print(json.dumps({'checks':checks,'isolation':info}))
'''
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'empirical_execution'),str(private),str(outside)],capture_output=True,text=True,stdin=subprocess.DEVNULL,timeout=30)
    if result.returncode:raise AssertionError('Independent seccomp capability test failed: '+result.stderr)
    value=json.loads(result.stdout)
    for name,ok in value['checks'].items():check('kernel seccomp '+name,ok)
    check('kernel seccomp permits only preopened write-only output in this probe',(private/'allowed.txt').read_text()=='only output')
    check('seccomp filter applied synchronously to multiple preexisting threads',value['isolation']['seccomp_thread_synchronization']=='TSYNC' and value['isolation']['native_threads_at_restriction']>=2 and value['checks']['other_preexisting_thread_denied'])
    return value

def replication_checks(tmp):
    import gzip
    from phase5 import replication as r
    from phase5 import replication_encoding as e
    source=tmp/'wcep_software.jsonl.gz'
    events=[{'id':'event-one','date':'2019-01-02','collection':'train','summary':'SUMMARY MUST NOT ENTER TEXT','articles':[{'title':'Same title','text':'Same title\narticle body','url':'https://fixture.invalid/a'},{'title':'Same title','text':'Same title\narticle body','url':'https://fixture.invalid/a'}]},
            {'id':'event-two','date':'2019-01-03','collection':'train','summary':'OTHER FORBIDDEN SUMMARY','articles':[{'title':'Same title','text':'Same title\narticle body','url':'https://fixture.invalid/a'},{'title':'Other','text':'author text'}]}]
    with gzip.open(source,'wt') as f:
        for event in events:f.write(json.dumps(event)+'\n')
    adapter=tmp/'adapted';panel=tmp/'event_panel';audit=r.parse_wcep([source],adapter,snapshot_revision='SOFTWARE_FIXTURE_ONLY',collection_scope='train')
    rows=lines(adapter/'records.jsonl')
    check('WCEP article instances stay distinct despite same URL and text',len(rows)==4 and len({x['record_id'] for x in rows})==4 and sum(x['text']=='Same title article body' for x in rows)==3)
    check('WCEP event summaries not encoded as article content',all('SUMMARY' not in x['text'] for x in rows))
    check('WCEP source and duplicate gold not fabricated',all(x['source_kind']=='unknown_singleton' and x['labels']==[] for x in rows) and not audit['input_authenticity_verified'])
    selected=r.event_panel(adapter,panel,target=1)
    check('WCEP boundary event remains complete',selected['selected_articles']==2 and selected['overshoot']==1 and len(selected['selected_events'])==1)
    reread,_,_=e.load_wcep_panel(adapter,panel)
    check('WCEP encoder intake exactly matches complete event rows',reread==lines(panel/'records.jsonl'))
    p=tmp/'missing_model'
    rejects('WCEP actual encoding cannot create output without authentic pinned assets',lambda:e.encode_wcep(adapter,panel,tmp/'missing-assets.json',tmp/'missing-model',tmp/'missing-tokenizer',p),exceptions=(ValueError,RuntimeError,FileNotFoundError))
    check('failed WCEP asset preflight leaves no successful cache',not p.exists())
    changed=rows[0].copy();changed['text']='changed'
    with (panel/'records.jsonl').open('a') as f:f.write(json.dumps(changed)+'\n')
    rejects('WCEP changed panel cannot pass lineage checks',lambda:e.load_wcep_panel(adapter,panel))
    chronology=[{'record_id':'a','partition':'train','original_fields':{'date':'2018-01-01T01:00:00'}},{'record_id':'b','partition':'train','original_fields':{'date':'2018-01-01T21:00:00'}},{'record_id':'c','partition':'train','original_fields':{'date':'2018-01-02'}},{'record_id':'d','partition':'train','original_fields':{'date':'2019-03-01'}}]
    got=r.news_chronological_requests(chronology,horizon=1)
    check('News chronology never cuts date group to fit budget',got['checkpoints']==[] and got['blocked_next_group']['records']==2)
    got=r.news_chronological_requests(chronology,horizon=3)
    check('News chronology keeps nested whole-date record sets',got['checkpoints'][-1]['cumulative_record_ids']==['a','b','c'] and got['blocked_next_group']['date_group']=='2019-03-01')
    check('News descriptive sequence not counted as random replication',got['independent_random_replicates']==0 and not got['source_withdrawal_claim'])
    rejects('News chronology refuses held-out records',lambda:r.news_chronological_requests([dict(chronology[0],partition='test')]))
    rejects('News chronology excludes threshold-calibration year',lambda:r.news_chronological_requests([dict(chronology[0],original_fields={'date':'2017-01-01'})]))
    return {'WCEP_fixture_instances':4,'actual_WCEP_corpus_processed':False,'real_encoder_or_inherited_quality_pass':False,'News_scope':'descriptive whole-date record stream'}

def natural_boundary_artifacts():
    records,_,cx=natural_inputs();ids=[r['record_id'] for r in records]
    root=ROOT/'empirical_execution/phase5/results/boundary_engineering_release_v2'
    result=read(root/'boundary_audit.json');lock=read(root/'lock.json');projection=np.load(root/'fixed_projection.npy',allow_pickle=False)
    x=(cx.astype(np.float64)@projection).astype(np.float32);y=np.array([r['label'] for r in records],np.float64)[:,None]
    blockers,scores=independent_natural_graph(cx,ids)
    # Reconstruct fixed LSH from locked planes independently using scalar loops.
    config=result['ann']['config'];rng=np.random.Generator(np.random.PCG64(config['seed']))
    planes=rng.standard_normal((config['tables'],config['bits'],cx.shape[1]),dtype=np.float64)
    z=[]
    for row in cx:
        n=0.
        for v in row:n+=float(v)*float(v)
        z.append([float(v)/math.sqrt(n) for v in row])
    candidates=set()
    for table in range(config['tables']):
        buckets={}
        for i,row in enumerate(z):
            signature=0
            for bit in range(config['bits']):
                v=0.
                for a,b in zip(row,planes[table,bit]):v+=a*float(b)
                if v>=0:signature|=1<<bit
            for j in buckets.setdefault(signature,[]):candidates.add((j,i))
            buckets[signature].append(i)
    approximate={r:set() for r in ids}
    for i,j in candidates:
        if ids[j] in blockers[ids[i]]:approximate[ids[i]].add(ids[j])
        if ids[i] in blockers[ids[j]]:approximate[ids[j]].add(ids[i])
    exactedges=sum(map(len,blockers.values()));annedges=sum(map(len,approximate.values()))
    metrics=result['ann']['metrics']
    check('saved ANN recall reconstructs independent scalar graph and LSH signatures',metrics['exact_edges']==exactedges and metrics['retained_edges']==annedges and metrics['retrieved_candidates']==len(candidates))
    manifest=read(ROOT/'empirical_execution/phase4/results/execution_engineering_final/d64/requests.json');paths={p['trajectory_id']:p for p in manifest['trajectories']}
    count=0;maxerr=0.
    for row in result['rows']:
        path=paths[row['trajectory_id']];dead=set(path['deletion_order'][:row['checkpoint']])
        for key,B in [('reference_weights',blockers),('ANN_weights',approximate)]:
            ix=[i for i,r in enumerate(ids) if r not in dead and B[r]<=dead];zz=x[ix].astype(np.float64);lam=row['lambda']
            want=zz.T@np.linalg.solve(zz@zz.T+lam*len(zz)*np.eye(len(zz)),y[ix]) if len(zz) else np.zeros((x.shape[1],1))
            got=np.asarray(row[key]);err=float(np.max(np.abs(got-want)));maxerr=max(maxerr,err)
            if err>1e-10:raise AssertionError('boundary saved head disagrees with independent target')
            count+=1
        saved=np.asarray(row['reference_weights']);fp32=np.asarray(row['FP32_weights']);ann=np.asarray(row['ANN_weights'])
        if not math.isclose(float(np.max(np.abs(fp32-saved))),row['FP32_head_max_abs'],abs_tol=1e-15):raise AssertionError('FP32 frontier metric mismatch')
        if not math.isclose(float(np.max(np.abs(ann-saved))),row['ann_head_max_abs'],abs_tol=1e-15):raise AssertionError('ANN frontier metric mismatch')
    check('all boundary reference and ANN heads match independent dual solves',True)
    check('all boundary FP32 and ANN saved head error metrics reconstruct',True)
    check('boundary sensitivity preserves three prespecified lambda factors',{row['lambda_factor'] for row in result['rows']}=={.1,1.,10.})
    check('natural envelope covers complete unordered-pair universe',result['graph_envelope']['all_pairs_certified']==len(ids)*(len(ids)-1)//2)
    return {'saved_reference_and_ANN_heads_checked':count,'max_independent_dual_absolute_error':maxerr,'reference_edges':exactedges,'ANN_edges':annedges,'ANN_is_exact':annedges==exactedges,'semantic_quality_claim':False}

def isolated_natural_artifacts():
    from ccu.data import lexical_engineering_features
    root=ROOT/'empirical_execution/phase5/results/isolated_natural_engineering_final'
    lock=read(root/'prospective_lock.json');summary=read(root/'summary.json');ledger=lines(root/'job_ledger.jsonl')
    planned={j['job_id']:j for j in lock['job_order']};observed={j['job_id']:j for j in ledger}
    check('isolated natural ledger retains every prospectively planned job once',len(observed)==len(ledger)==len(planned) and set(observed)==set(planned))
    reconciliation=read(root/'journal_reconciliation.json');original=lines(root/'job_ledger_original.jsonl')
    check('original incomplete journal preserved byte exactly',sha(root/'job_ledger_original.jsonl')==reconciliation['original_sha256'] and len(original)==213)
    check('canonical reconciled journal hash agrees',sha(root/'job_ledger.jsonl')==reconciliation['canonical_sha256'])
    check('journal recovered precisely the missing three completed jobs',set(reconciliation['recovered_job_ids'])==set(observed)-{r['job_id'] for r in original} and len(reconciliation['recovered_job_ids'])==3)
    for row in ledger:
        audit_path=root/'jobs'/row['job_id']/'audit.json'
        if read(audit_path)!=row or sha(audit_path)!=reconciliation['exact_per_job_audit_sha256'][row['job_id']]:raise AssertionError('journal row differs from original per-job audit')
    check('journal rows reconstruct all exact completed per-job audits without reruns',all(observed[r['job_id']]==r for r in original) and not reconciliation['method_runs_repeated'])
    for name,h in lock['dependency_hashes'].items():
        if sha(ROOT/'empirical_execution'/name)!=h:raise AssertionError('isolated locked dependency changed '+name)
    check('isolated natural production dependencies remain locked',True)
    records,_,cx=natural_inputs();ids=[r['record_id'] for r in records];y=np.array([r['label'] for r in records],np.float64)[:,None]
    blockers,_=independent_natural_graph(cx,ids);dimensions={};cache={};maxerr=0.;nheads=0;scopes=set()
    requests={p['trajectory_id']:p for p in lock['requests']}
    for row in ledger:
        check('isolated job completed '+row['job_id'],row['status']=='passed')
        job=root/'jobs'/row['job_id'];report=read(job/'repair/repair_report.json');iso=report['isolation'];scopes.add(iso['backend'])
        if not (iso['all_new_file_opens_denied'] and iso['input_file_descriptors_closed_before_repair'] and iso['only_output_file_descriptors_opened_before_restriction'] and iso['seccomp_thread_synchronization']=='TSYNC' and iso['denied_existing_file_probes']>=1):raise AssertionError('isolated repair did not establish declared capability boundary')
        if report['selected_ID_exports'] or not report['persistent_head_only_output']:raise AssertionError('unexpected repair output capability')
        d=row['dimension']
        if d not in dimensions:dimensions[d]=lexical_engineering_features(records,d)[0]
        x=dimensions[d];D=set()
        initial_key=(d,'initial',0)
        if initial_key not in cache:
            selected=[j for j,r in enumerate(ids) if not blockers[r]];z=x[selected].astype(np.float64)
            desired=z.T@np.linalg.solve(z@z.T+lock['lambda_reg']*len(z)*np.eye(len(z)),y[selected]) if len(z) else np.zeros((d,1))
            cache[initial_key]=(desired,len(selected))
        desired,count=cache[initial_key];initialpath=job/'construction/initial_head.npy';got=np.load(initialpath,allow_pickle=False)
        creport=read(job/'construction/construction_report.json');err=float(np.max(np.abs(got-desired)));maxerr=max(maxerr,err)
        if err>lock['tolerance'] or creport['selected_count']!=count or sha(initialpath)!=creport['head_sha256']:raise AssertionError('isolated initial head/target/hash mismatch')
        nheads+=1
        for i,batch in enumerate(requests[row['path']]['requests']):
            D.update(batch);key=(d,row['path'],i)
            if key not in cache:
                selected=[j for j,r in enumerate(ids) if r not in D and blockers[r]<=D];z=x[selected].astype(np.float64)
                desired=z.T@np.linalg.solve(z@z.T+lock['lambda_reg']*len(z)*np.eye(len(z)),y[selected]) if len(z) else np.zeros((d,1))
                cache[key]=(desired,len(selected))
            desired,count=cache[key];headpath=job/f'repair/head_{i:04d}.npy';got=np.load(headpath,allow_pickle=False)
            err=float(np.max(np.abs(got-desired)));maxerr=max(maxerr,err)
            if err>lock['tolerance'] or report['releases'][i]['count']!=count or sha(headpath)!=report['releases'][i]['head_sha256']:raise AssertionError('isolated saved head/target/hash mismatch')
            nheads+=1
    check('all isolated saved heads independently reconstructed from scalar graph and dual ridge',True)
    check('all isolated jobs genuinely enforced synchronized kernel access restriction',scopes=={'seccomp_fd_only'})
    check('isolated pilot does not claim semantic/source or primary systems confirmation',not summary['source_evidence'] and not summary['semantic_evidence'] and not summary['paper_speedup_claim'])
    return {'planned_jobs':len(planned),'saved_heads_checked':nheads,'max_independent_head_absolute_error':maxerr,
            'native_dimension':768,'native_dimension_repetitions':1,'primary_systems_allocation_executed':False,
            'no_reaccess_boundary':'trusted repair process after own-state load; no new file opens; input FDs closed; TSYNC seccomp',
            'saved_state_limitation':'large service snapshots intentionally not archived; state bytes/hashes and generation code retained; restore checked separately on software states'}

def new_decoder_and_metrics_checks():
    from ccu.core import RidgeMoments
    from phase5.decoders import decode,extended_audit
    from phase5.task_metrics import binary_rank_metrics,multilabel_metrics,civil_metrics
    x=np.array([[1,.5],[-.5,1],[.25,-.25],[2,1]],np.float32)
    y=np.array([[.25,0],[.5,0],[1,0],[0,0]],np.float64)
    G,H,n=independent_moments(x,y,range(len(x)));mom=RidgeMoments(G,H,n)
    want=exact_ridge(x,y,range(n),Fraction(1,8))
    for solver in ['cg','cholesky']:
        result=decode(mom,.125,solver=solver)
        check(solver+' common decoder releases under fixed gate',result.diagnostics['release_allowed'] and result.diagnostics['normalized_residual_eta']<=1e-10)
        check(solver+' independently solved dyadic matrix target',np.max(np.abs(result.weights-np.asarray(want,float)))<1e-12)
        check(solver+' zero output RHS preserved exactly',not result.weights[:,1].any())
        report=extended_audit(x,y,result.weights,.125,computed_moments=mom)
        check(solver+' extended audit never overrides or certifies candidate',not report['replacement_head_produced'] and not report['service_failure_overridden'] and not report['numerically_certified'])
    check('empty moment drift fails closed in new common decoder',not decode(RidgeMoments(np.eye(2)*1e-300,np.zeros((2,2)),0),.125).diagnostics['release_allowed'])
    yy=np.array([1,0,1,0,1]);scores=np.array([.8,.8,.2,.1,.1]);weights=np.array([2,3,1,4,5])
    got=binary_rank_metrics(yy,scores,weights)
    auc=sum(weights[i]*weights[j]*(float(scores[i]>scores[j])+.5*float(scores[i]==scores[j])) for i in range(5) for j in range(5) if yy[i] and not yy[j])/(weights[yy==1].sum()*weights[yy==0].sum())
    ap=sum(weights[(yy==1)&(scores==s)].sum()/weights[yy==1].sum()*weights[(yy==1)&(scores>=s)].sum()/weights[scores>=s].sum() for s in set(scores))
    check('weighted tied AUROC matches independent positive-negative pair enumeration',math.isclose(got['auroc'],auc,abs_tol=1e-15))
    check('weighted tied AP matches independent threshold grouping',math.isclose(got['average_precision'],ap,abs_tol=1e-15))
    my=np.column_stack([yy,np.zeros(len(yy),int)]);ms=np.column_stack([scores,np.ones(len(yy))])
    multi=multilabel_metrics(my,ms,[.5,math.inf],weights)
    check('multilabel zero tag and zero-label rows remain in metrics',multi['zero_positive_outputs']==1 and multi['zero_target_rows_retained']==2 and multi['per_output_f1'][1]==0.)
    check('multilabel macro AP includes zero-positive output',math.isclose(multi['macro_average_precision'],ap/2,abs_tol=1e-15))
    check('Civil task scores are not clipped',civil_metrics([0.,1.],[-1.,2.])['mse']==1.)
    return {'algebraic_software_only':True,'independent_weighted_AUROC':auc,'independent_weighted_AP':ap,'common_decoder_is_not_retroactively_used_by_frozen_workers':True}

def convex_payload_checks(tmp):
    from phase5.convex_program import LogisticPayload
    blockers=[[],[0],[0,1],[],[2,3]];owners=['s0','s0','s1','s2','s3'];graph=graph_fixture(blockers,owners)
    x=np.array([[1,0],[2,.5],[3,-.5],[4,.25],[5,1]],np.float32)
    y=np.array([[0,1],[.25,0],[.5,1],[.75,0],[1,1]],np.float64)
    state=LogisticPayload(graph,x,y,3,'source',compaction_fraction=.5);lookup={r:i for i,r in enumerate(graph.record_ids)}
    check('convex payload excludes own-source-blocked records',set(state.counter.record_ids)=={graph.record_ids[i] for i in independent_eligible(blockers,owners,[],3)})
    state.delete(['s0']);path=tmp/'convex_owned.npz';receipt=state.snapshot(path,np.zeros((2,2),np.float64));restored,_,_=LogisticPayload.load(path,expected_sha256=receipt['file_sha256'])
    for service in (state,restored):
        service.delete(['s2']);xx,yy,ids=service.selected_payload();ix=[lookup[r] for r in ids]
        check('convex real payload stays aligned after source compaction and resume',np.array_equal(xx,x[ix]) and np.array_equal(yy,y[ix]) and set(ix)==set(independent_selected(blockers,owners,['s0','s2'])))
        check('convex remaining horizon eligibility preserved',set(service.counter.logical_membership())=={graph.record_ids[i] for i in independent_eligible(blockers,owners,['s0','s2'],1)})
    rejects('convex own snapshot hash cannot change',lambda:LogisticPayload.load(path,expected_sha256='0'*64))
    return {'source_software_fixture_only':True,'real_payload_snapshot_resume_checked':True,'shared_process_isolation_claim':False}

def registry_checks(tmp):
    from phase5.study_registry import build_registry,preflight
    registry,jobs=build_registry();ids=[j['job_id'] for j in jobs]
    check('full prospective registry has unique retained planned jobs',len(ids)==len(set(ids))==registry['job_count'])
    check('registry does not activate primary execution or pretend preregistration',not registry['execution_allowed'] and not registry['primary_semantic_study_started'] and not registry['preregistration_submitted'] and all(not j['execution_allowed'] for j in jobs))
    wcep=[g for g in registry['groups'] if g['corpus']=='wcep100']
    check('WCEP record replication inherits News calibration without own calibration',all('cc_news.e5.selection_lock' in g['required_artifact_keys'] and 'wcep100.e5.selection_lock' not in g['required_artifact_keys'] and 'wcep100.e5.semantic_guard' not in g['required_artifact_keys'] and 'wcep100.original_URL_verification' not in g['required_artifact_keys'] for g in wcep))
    threshold_groups=[g for g in registry['groups'] if g['variant'].startswith('threshold-')]
    check('threshold sensitivities use narrative endpoint clipping',{g['configuration']['threshold'] for g in threshold_groups}=={'max(0,calibrated-0.02)','min(1,calibrated+0.02)'})
    got=preflight(registry,{'artifacts':{},'capabilities':{'all_done':True}},tmp)
    check('empty registry capability declarations never activate or verify inputs',not got['execution_allowed'] and got['byte_verified_count']==0 and all(not g['execution_allowed'] for g in got['group_status']))
    return {'planned_groups':len(registry['groups']),'planned_jobs':len(jobs),'required_artifact_keys':len(registry['required_artifact_keys']),'unresolved_protocol_obligations':registry['unresolved_obligations'],'primary_activation_implemented':False}

def human_analysis_checks():
    from phase5.human_analysis import _weighted_binary,analyze_human_audit
    domain=['a','b','c','d'];selected={'a':{},'b':{}};prob={i:Fraction(1,2) for i in domain}
    result=_weighted_binary(domain,selected,prob,{'a':1.,'b':None})
    check('human missing outcomes retain undefined point estimator and correct weighted extremes',result['ht_prevalence'] is None and result['hajek_missing_outcome_extremes']==[.5,1.])
    check('human finite-frame identification includes all unsampled records',result['finite_frame_bounded_outcome_identification_bounds']==[.25,1.] and not result['intervals_above_are_sampling_confidence_intervals'])
    complete=_weighted_binary(domain,selected,prob,{'a':1.,'b':0.})
    check('human HT estimator uses exact inclusion weights and full domain size',complete['ht_prevalence']==.5 and complete['finite_frame_bounded_outcome_identification_bounds']==[.25,.75])
    pack=ROOT/'empirical_execution/phase5/results/context_audit_engineering/blank_pack'
    manifest=read(pack/'private_sampling_manifest.json')
    candidates=list(pack.glob('*response*.json'))
    if len(candidates)!=1:raise AssertionError('unexpected context blank-response layout')
    analysis=analyze_human_audit(manifest,read(candidates[0]))
    check('real blank context pack analysis retains zero actual responses',analysis['actual_human_response_count']==0 and not analysis['confirmatory_claims_established'])
    check('blank context categories never become observed no/uncertain judgments',all(v['ht_prevalence'] is None for row in analysis['original_summaries'] for choices in row['categories'].values() for v in choices.values()))
    return {'actual_human_responses':0,'blank_context_assignments':analysis['expected_assignment_count'],'missingness_envelopes_are_not_sampling_confidence_intervals':True}

def natural_convex_program_artifacts():
    from decimal import Decimal,localcontext
    from phase5.convex_multioutput import fraction
    root=ROOT/'empirical_execution/phase5/results/convex_program_release_v2/natural'
    lock=read(root/'lock.json');summary=read(root/'summary.json');records,x,curator=natural_inputs()
    ids=[r['record_id'] for r in records];lookup={r:i for i,r in enumerate(ids)};y=np.array([[r['label']] for r in records],np.float64)
    blockers,_=independent_natural_graph(curator,ids);cases=[]
    for path in sorted(root.glob('R*')):
        initial=read(path/'initial.json');selected=[i for i,r in enumerate(ids) if not blockers[r]]
        if set(initial['selected_ids'])!={ids[i] for i in selected}:raise AssertionError('convex initial selected set')
        for method,fit in initial['methods'].items():
            order=[lookup[r] for r in initial['selected_ids']] if method=='fresh_optimum' else selected
            cases.append((order,fit,path/(method+'_initial_head.npy'),initial['stages'][method]))
        for file in sorted(path.glob('checkpoint_*.json')):
            row=read(file);dead=set(row['deleted_ids']);selected=[i for i,r in enumerate(ids) if r not in dead and blockers[r]<=dead]
            if set(row['selected_ids'])!={ids[i] for i in selected}:raise AssertionError('convex fresh retained selected set')
            for method,fit in row['methods'].items():
                if fit['status']!='released':raise AssertionError('natural convex method did not release')
                order=sorted(selected,key=lambda i:(hashlib.sha256(('priority-v1|0|'+ids[i]).encode()).digest(),ids[i])) if method=='fresh_optimum' else selected
                cases.append((order,fit,path/(method+f'_head_{row["checkpoint"]}.npy'),fit['cost']))
    ahash=lambda a:hashlib.sha256(str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+np.ascontiguousarray(a).tobytes()).hexdigest()
    with localcontext() as ctx:
        ctx.prec=120;D=lambda v:Decimal(v.numerator)/Decimal(v.denominator)
        for selected,fit,file,cost in cases:
            xx=x[selected];yy=y[selected];w=np.asarray(fit['weights'],np.float64);cert=fit['certificate'];n=len(xx)
            if cert['input_bindings']!={'features':ahash(xx),'targets':ahash(yy),'weights':ahash(w)}:raise AssertionError('convex program certificate binding')
            if not np.array_equal(np.load(file,allow_pickle=False),w) or sha(file)!=cost['head_file_sha256'] or file.stat().st_size!=cost['head_release_bytes']:raise AssertionError('convex actual output receipt mismatch')
            wd=[Decimal.from_float(float(v)) for v in w[:,0]];g=[Decimal.from_float(lock['lambda_reg'])*v for v in wd]
            for row,target in zip(xx,yy):
                xd=[Decimal.from_float(float(v)) for v in row];score=sum(a*b for a,b in zip(xd,wd));p=1/(1+(-score).exp())
                for j,v in enumerate(xd):g[j]+=v*(p-Decimal.from_float(float(target[0])))/n
            scalar=cert['scalar_certificates'][0]
            if not all(D(fraction(lo))<=v<=D(fraction(hi)) for lo,hi,v in zip(scalar['gradient_lower'],scalar['gradient_upper'],g)):raise AssertionError('convex program high-precision gradient containment')
            if not cert['meets_parameter_tolerance'] or fraction(cert['parameter_error_frobenius_squared_upper'])>fraction(cert['parameter_tolerance'])**2:raise AssertionError('convex parameter certificate')
            logits=x.astype(float)@w
            if not np.array_equal(logits,np.asarray(fit['evaluation']['score_logits'])):raise AssertionError('convex saved task predictions')
    check('all three-method convex initial/release heads and certificate gradients independently validated',len(cases)==30)
    check('convex program preserves absent source trajectories and shared-process scope',summary['missing_trajectories']==2 and not summary['primary_study_started'] and not summary['speedup_claim'] and not summary['process_isolation_claim'])
    check('convex final source bound to authoritative engineering release',lock['code_sha256']['convex_program.py']==sha(ROOT/'empirical_execution/phase5/convex_program.py'))
    return {'initial_and_release_heads_checked':len(cases),'independent_Decimal_digits':120,'actual_source_experiment':False,'primary_process_isolation':False}

def natural_task_program_artifacts():
    from ccu.data import lexical_engineering_features
    root=ROOT/'empirical_execution/phase5/results/task_program_release'
    design=read(root/'engineering_design.json');out=root/'executed';lock=read(out/'run_lock.json');summary=read(out/'summary.json')
    records,_,_=natural_inputs();roles={r['record_id']:r['partition'] for r in read(ROOT/'empirical_execution/phase2/results/partition_manifest.json')}
    train=[i for i,r in enumerate(records) if roles[r['record_id']]=='development_train'];test=[i for i,r in enumerate(records) if roles[r['record_id']]=='development_evaluation']
    ids=[records[i]['record_id'] for i in train];y=np.array([[records[i]['label']] for i in train],np.float64);ey=np.array([[records[i]['label']] for i in test],np.float64)
    groups=design['registry']['groups'];rows=read(out/'checkpoints.json');ledger=read(out/'job_ledger.json');requests=next(iter(lock['bundle_bindings'].values()))['requests'];paths={p['trajectory_id']:p for p in requests['trajectories']}
    features={d:lexical_engineering_features(records,d)[0] for d in (64,128,256)};contexts={};heads=0;worst=0.
    for gi,g in enumerate(groups):
        folder=out/f'group_{gi:03d}'
        if not (folder/'utility.json').exists():continue
        config=g['configuration'];ld=int(config['learner_encoder'].replace('lexical',''));cd=int(config['curator_encoder'].replace('lexical',''))
        x=features[ld][train];ex=features[ld][test];utility=read(folder/'utility.json');geometry=utility['geometry'];lam=geometry['lambda']
        if config['learner_dimension']!=ld:
            d=config['learner_dimension'];z=np.random.Generator(np.random.PCG64(config['projection_seed'])).standard_normal((ld,d));p,r=np.linalg.qr(z,mode='reduced');p*=np.where(np.diag(r)<0,-1.,1.)
            if np.max(np.abs(p.T@p-np.eye(d)))>1e-12:raise AssertionError('task projection orthogonality')
            x=(x.astype(float)@p).astype(np.float32);ex=(ex.astype(float)@p).astype(np.float32)
        blockers,_=independent_natural_graph(features[cd][train],ids,geometry['threshold'],config['priority_seed']);initial={i for i,rid in enumerate(ids) if not blockers[rid]}
        hashorder=sorted(range(len(ids)),key=lambda i:(hashlib.sha256(('ccu-v1-utility-hash\0'+ids[i]).encode()).digest(),ids[i]))
        before=None
        for name,selected in [('curated_ridge',initial),('uncurated_ridge',set(range(len(ids)))),('same_size_hash_ridge',set(hashorder[:len(initial)]))]:
            saved=np.load(folder/(name+'.npz'),allow_pickle=False);ix=sorted(selected);z=x[ix].astype(float)
            expected=z.T@np.linalg.solve(z@z.T+lam*len(z)*np.eye(len(z)),y[ix]) if len(z) else np.zeros((x.shape[1],1))
            err=float(np.max(np.abs(expected-saved['weights'])));worst=max(worst,err)
            if err>1e-10 or set(saved['selected_indices'].tolist())!=selected or not np.array_equal(ex.astype(float)@saved['weights'],saved['predictions']):raise AssertionError('task initial utility target or prediction')
            heads+=1
            if name=='curated_ridge':before=saved['predictions'].copy()
        contexts[g['group_id']]=(x,ex,lam,blockers,initial,before)
    for row in rows:
        if row['status']!='completed':raise AssertionError('task checkpoint failure')
        x,ex,lam,blockers,initial,before=contexts[row['group_id']];D=set(paths[row['trajectory_id']]['deletion_order'][:row['checkpoint']]);selected={i for i,r in enumerate(ids) if r not in D and blockers[r]<=D};frozen={i for i in initial if ids[i] not in D}
        file=out/row['prediction_file'];saved=np.load(file,allow_pickle=False)
        if sha(file)!=row['prediction_sha256'] or set(row['correct_selected_ids'])!={ids[i] for i in selected}:raise AssertionError('task counterfactual IDs or file binding')
        for name,target in [('oracle',selected),('frozen',frozen)]:
            ix=sorted(target);z=x[ix].astype(float);expected=z.T@np.linalg.solve(z@z.T+lam*len(z)*np.eye(len(z)),y[ix]) if len(z) else np.zeros((x.shape[1],1))
            err=float(np.max(np.abs(expected-saved[name+'_weights'])));worst=max(worst,err)
            if err>1e-10 or not np.array_equal(ex.astype(float)@saved[name+'_weights'],saved[name]):raise AssertionError('task saved ridge prediction target')
            heads+=1
        effect=(np.mean((saved['frozen']-ey)**2)-np.mean((saved['oracle']-ey)**2))/np.mean((before-ey)**2)
        if not math.isclose(effect,row['effects']['signed_normalized_loss'],abs_tol=1e-12):raise AssertionError('task signed effect changed')
    check('task utility and all counterfactual saved heads independently reconstructed',heads==244)
    check('task complete planned cells retain missing semantic inputs and out-of-scope methods',len(ledger)==summary['planned_task_jobs']==31 and summary['completed_task_jobs']==26 and summary['blocked_or_other_jobs']==5)
    blocked=read(root/'registered_missing_inputs/summary.json')
    check('actual registered semantic task jobs remain unexecuted',blocked['completed_task_jobs']==0 and blocked['planned_task_jobs']==blocked['observed_task_jobs'])
    check('task program explicitly reuses old evaluation data with no primary promotion',not design['new_independent_empirical_evidence'] and not summary['primary_study'])
    check('task production source bound to executed release',summary['code_bindings']['phase5/task_program.py']==sha(ROOT/'empirical_execution/phase5/task_program.py'))
    return {'ridge_heads_checked':heads,'task_checkpoint_rows':len(rows),'max_independent_dual_absolute_error':worst,'reused_training_rows':len(train),'reused_evaluation_rows':len(test),'new_independent_corpus':False,'registered_task_jobs_blocked':blocked['planned_task_jobs']}

def natural_worker_v2_artifacts():
    import ast
    from ccu.data import lexical_engineering_features
    root=ROOT/'empirical_execution/phase5/results/worker_v2_engineering_final';verification=read(root/'verification.json')
    records,_,_=natural_inputs();records=records[:20];ids=[r['record_id'] for r in records];cx=lexical_engineering_features(records,32)[0];x=lexical_engineering_features(records,16)[0];y=np.array([[r['label']] for r in records],np.float64)
    blockers,_=independent_natural_graph(cx,ids);count=0;worst=0.;native=set()
    for case in verification['cases']:
        folder=root/(case['method']+'_'+case['solver']);config=read(folder/'repair/repair.json');report=read(folder/'repair/repair_report.json');initial=read(folder/'construction/construction_report.json')
        D=set();examples=[(set(),folder/'construction/initial_head.npy',initial['numerical_decoder'],initial['head_sha256'])]
        for i,batch in enumerate(config['requests']):
            D.update(batch);release=report['releases'][i];examples.append((set(D),folder/f'repair/head_{i:04d}.npy',release['numerical_decoder'],release['head_sha256']))
        for deleted,file,gate,h in examples:
            ix=[i for i,r in enumerate(ids) if r not in deleted and blockers[r]<=deleted];z=x[ix].astype(float);lam=config['lambda_reg']
            expected=z.T@np.linalg.solve(z@z.T+lam*len(z)*np.eye(len(z)),y[ix]) if len(z) else np.zeros((16,1));actual=np.load(file,allow_pickle=False)
            error=float(np.max(np.abs(actual-expected)));worst=max(worst,error)
            if error>1e-10 or sha(file)!=h or not gate['release_allowed'] or gate['normalized_residual_eta']>1e-10 or gate['fallback_performed']:raise AssertionError('v2 kernel worker target or release gate mismatch')
            count+=1
        iso=report['isolation']
        if iso['seccomp_thread_synchronization']!='TSYNC' or not iso['input_file_descriptors_closed_before_repair'] or iso['denied_existing_file_probes']<1:raise AssertionError('v2 no-reaccess boundary')
        if report['snapshot_hash_and_load_read_meter']['read_call_bytes']<report['input_snapshot_logical_bytes']:raise AssertionError('v2 snapshot I/O meter')
        for dependency in report['loaded_native_dependencies']['files']:
            if sha(Path(dependency['path']))!=dependency['sha256']:raise AssertionError('v2 loaded ELF hash changed')
            native.add(dependency['path'])
    for file,h in verification['source_sha256'].items():
        if sha(ROOT/'empirical_execution/phase5'/file)!=h:raise AssertionError('v2 source changed')
    functions=[]
    for version in ('workers.py','workers_v2.py'):
        tree=ast.parse((ROOT/'empirical_execution/phase5'/version).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='confine_open_syscalls');functions.append(ast.dump(fn,include_attributes=False))
    check('v2 uses identical independently kernel-probed seccomp installation function',functions[0]==functions[1])
    check('v2 all nine methods both solvers pass independent targets and fixed residual gate',count==54)
    check('v2 initial failures were preserved rather than silently replaced',(ROOT/'empirical_execution/phase5/results/worker_v2_engineering/verification.json').exists())
    return {'services':len(verification['cases']),'saved_initial_and_release_heads_checked':count,'max_independent_absolute_error':worst,'loaded_ELF_files_rehashed':len(native),'numeric_diagnostics_are_not_interval_certificates':True,'metadata_read_bytes_still_unknown':True}

def main():
    import time
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'empirical_execution/phase5/results/independent_review.json');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Independent audit output already exists; preserve historical audit')
    started=time.perf_counter();results={};before=source_hashes()
    with tempfile.TemporaryDirectory(prefix='ccu_independent_phase5_') as folder:
        tmp=Path(folder)
        for name in ['multioutput_checks','service_checks','boundary_checks','refit_checks','confinement_checks','seccomp_checks','natural_convex_artifacts','natural_refit_artifacts','natural_context_artifacts','statistics_checks','replication_checks','natural_boundary_artifacts','isolated_natural_artifacts','new_decoder_and_metrics_checks','convex_payload_checks','registry_checks','human_analysis_checks','natural_convex_program_artifacts','natural_worker_v2_artifacts','natural_task_program_artifacts']:
            print('independent audit: '+name,flush=True)
            fn=globals()[name];needs_tmp=name in {'multioutput_checks','service_checks','refit_checks','confinement_checks','seccomp_checks','replication_checks','convex_payload_checks','registry_checks'}
            results[name]=fn(tmp) if needs_tmp else fn()
    after=source_hashes()
    check('reviewed Phase5 code did not change during independent audit',before==after)
    report={'schema':'ccu-independent-phase5-review-1','passed_with_explicit_scope_limits':True,'check_count':len(CHECKS),'checks':CHECKS,'results':results,
      'elapsed_seconds':time.perf_counter()-started,'reviewed_code_hashes':after,
      'scope':'software contracts and reused natural lexical engineering artifacts; not new independent corpora or confirmatory evidence',
      'primary_semantic_study_started':False,'real_human_ratings_available':0,'official_SemDeDup_backend_executed':False,
      'remaining_external_requirements':['Authentic complete corpora and genuine source ownership, including primary AskUbuntu/News and replications','Pinned E5/MPNet model/tokenizer assets, supported runtime and derived verified semantic caches','Actual independent blinded human threshold selection, validation and admission/context judgments','Calibration-only task configurations, guarded frozen panels/requests and untouched evaluation data','Supported official torch/faiss SemDeDup execution and full retained-input refit audit','Primary activation acceptance and all applicable unresolved registry obligations before confirmatory execution'],
      'implementation_still_pending':['Integrated primary semantic activation/acceptance gate','Registry obligations still requiring prospective allocation, recipe or actual-frame lock','Primary-scale acceptance of v2 workers and complete binary/logical I/O accounting chain','Authentic-scale execution of all registered studies and required controls'],
      'claims_not_established':['Semantic novelty prevalence or task benefit','Primary or real-source repair speedup','Total-byte memory optimality','Physical deletion, adversarial native-code confinement or privacy','Publication readiness or reviewer-proofness']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes=args.output.parent/'reviewed_code_hashes.json'
    if hashes.exists():raise FileExistsError('reviewed code hash artifact already exists')
    hashes.write_text(json.dumps(after,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed_with_explicit_scope_limits':True,'check_count':len(CHECKS),'output':str(args.output),'seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
