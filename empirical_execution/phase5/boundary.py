"""Secondary ANN and numerical boundary audits; never the primary curator.

Scorer intervals enclose the specified stored-value FP64 scorer assuming IEEE
binary64 round-to-nearest operations, gradual underflow and correctly rounded
sqrt. They are numerical bounds, not bounds on latent semantic similarity.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib,json,math,random,platform
from fractions import Fraction
import numpy as np
from ccu.core import BlockerGraph,ridge_moments
from phase3.panels import graph_normalize,reference_cosines
from phase3.reference_graph import build_reference_graph
from phase4.execution import array_hash,canonical,common_decode,write
from phase4.requests import verify_manifest,graph_binding


def integer(v,name,minimum=0):
    if isinstance(v,bool) or not isinstance(v,int) or v<minimum:raise ValueError(name)
    return v


def validate_x(x):
    x=np.asarray(x)
    if x.dtype!=np.float32 or x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('Nonempty finite stored FP32 feature matrix required')
    graph_normalize(x)
    return x


def graph_from_pairs(reference,pairs):
    n=len(reference.record_ids);rank=np.empty(n,dtype=int);rank[reference.priority_indices]=np.arange(n)
    blockers=[[] for _ in range(n)]
    for i,j in pairs:
        if not 0<=i<j<n:raise ValueError('Unordered pairs require 0 <= i < j < N')
        a,b=(i,j) if rank[i]>rank[j] else (j,i);blockers[a].append(b)
    indptr=np.zeros(n+1,dtype=np.int64)
    for i,row in enumerate(blockers):row.sort();indptr[i+1]=indptr[i]+len(row)
    index=np.asarray([j for row in blockers for j in row],dtype=np.int64)
    order=reference.priority_indices.copy()
    for a in (indptr,index,order):a.flags.writeable=False
    return BlockerGraph(reference.record_ids,reference.source_ids,order,indptr,index,reference.threshold)


def graph_pairs(graph):
    return {(min(i,int(j)),max(i,int(j))) for i,row in enumerate(graph.blockers) for j in row}


def lsh_candidates(x,*,tables=8,bits=10,seed=202710041):
    """Union same-bucket pairs from seeded random hyperplane signatures.

    This is an actual approximate candidate scheme, not exhaustive nearest
    neighbors under an ANN name. Plane bytes, NumPy and PCG64 are bound; no
    theoretical recall promise is made for floating-point implementation.
    """
    x=validate_x(x);integer(tables,'tables',1);integer(bits,'bits',1);integer(seed,'seed')
    if bits>64:raise ValueError('At most 64 bits per signature')
    z=graph_normalize(x);rng=np.random.Generator(np.random.PCG64(seed))
    planes=rng.standard_normal((tables,bits,x.shape[1]),dtype=np.float64)
    candidates=set();signatures=np.zeros((tables,len(x)),dtype=np.uint64)
    for table in range(tables):
        for bit in range(bits):
            score=np.zeros(len(x),dtype=np.float64)
            for k in range(x.shape[1]):score+=z[:,k]*planes[table,bit,k]
            signatures[table]|=(score>=0).astype(np.uint64)<<np.uint64(bit)
        buckets={}
        for i,key in enumerate(signatures[table]):
            bucket=buckets.setdefault(int(key),[])
            for j in bucket:candidates.add((j,i))
            bucket.append(i)
    config={'scheme':'phase5_random_hyperplane_LSH_v1','tables':tables,'bits':bits,'seed':seed,
      'generator':'numpy.PCG64.standard_normal.float64','zero_sign':'nonnegative_is_one',
      'candidate_rule':'union_all_unordered_same_bucket_pairs_no_top_k',
      'planes_sha256':array_hash(planes),'signatures_sha256':array_hash(signatures),
      'candidate_pairs_sha256':hashlib.sha256(canonical(sorted(candidates))).hexdigest(),
      'input_sha256':array_hash(x),'numpy':np.__version__,'python':platform.python_version(),
      'numeric_index_bytes':int(planes.nbytes+signatures.nbytes),
      'candidate_pair_payload_bytes_if_int64':16*len(candidates),
      'not_measured':'Python buckets/set overhead, allocator overhead and peak RSS'}
    return candidates,config


def missed_pair_sample(n,candidates,*,size=200,seed=202710042):
    """Reservoir SRS of the complete nonretrieved unordered-pair population.

    Streaming frame enumeration costs N(N-1)/2 membership checks; the audit is
    an evaluative expense and supplies no ANN scalability claim.
    """
    integer(n,'N');integer(size,'size',1);integer(seed,'seed')
    if any(not 0<=i<j<n for i,j in candidates):raise ValueError('Invalid candidate pair')
    rng=random.Random(seed);sample=[];population=0
    for i in range(n):
        for j in range(i+1,n):
            if (i,j) in candidates:continue
            population+=1
            if len(sample)<size:sample.append((i,j))
            else:
                choice=rng.randrange(population)
                if choice<size:sample[choice]=(i,j)
    sample.sort()
    return sample,{'sampling':'simple_random_without_replacement_reservoir',
      'seed':seed,'population_size':population,'sample_size':len(sample),
      'inclusion_probability_numerator':len(sample),'inclusion_probability_denominator':population,
      'inclusion_probability':len(sample)/population if population else None,
      'nonretrieved_pairs_all_have_positive_probability':bool(population),
      'empty_population':population==0,'retrieved_stratum':'census','strata_disjoint_and_exhaustive':True}


def audit_ann(x,reference,*,tables=8,bits=10,seed=202710041,sample_size=200,sample_seed=202710042):
    graph_binding(reference)  # Validates complete priority, ownership and oriented CSR.
    candidates,config=lsh_candidates(x,tables=tables,bits=bits,seed=seed)
    z=graph_normalize(x)
    if len(x)!=len(reference.record_ids):raise ValueError('Reference graph row alignment')
    declared=graph_pairs(reference);verified=set()
    for i in range(len(x)-1):
        scores=reference_cosines(z[i:i+1],z[i+1:])[0]
        verified.update((i,i+1+int(j)) for j in np.flatnonzero(scores>reference.threshold))
    if declared!=verified:raise ValueError('Reference graph is not the exhaustive fixed-input graph')
    found=set()
    for i,j in sorted(candidates):
        if reference_cosines(z[i:i+1],z[j:j+1])[0,0]>reference.threshold:found.add((i,j))
    approximate=graph_from_pairs(reference,found);truth=graph_pairs(reference)
    if not found<=truth:raise AssertionError('ANN rescore edges must be a subset of exact edges')
    sample,sampling=missed_pair_sample(len(x),candidates,size=sample_size,seed=sample_seed)
    scored=[]
    for i,j in sample:
        score=float(reference_cosines(z[i:i+1],z[j:j+1])[0,0])
        scored.append({'i':i,'j':j,'score':score,'edge':bool(score>reference.threshold)})
    successes=sum(p['edge'] for p in scored);m=sampling['population_size'];k=len(sample)
    complete=[set(a)==set(b) for a,b in zip(reference.blockers,approximate.blockers)]
    nonempty=[i for i,b in enumerate(reference.blockers) if len(b)]
    from phase3.calibration import finite_population_lower_bound
    if m:
        low=finite_population_lower_bound(m,k,successes,Fraction(1,40))['lower_success_count']
        high=m-finite_population_lower_bound(m,k,k-successes,Fraction(1,40))['lower_success_count']
    else:low=high=0
    sampling['missed_edge_count_interval_95']=[low,high]
    sampling['interval_method']='two exact hypergeometric one-sided inversions, alpha=1/40 each; union bound'
    metrics={'exact_edges':len(truth),'retrieved_candidates':len(candidates),'retained_edges':len(found),
      'edge_recall':len(found)/len(truth) if truth else None,
      'complete_blocker_records':sum(complete),'complete_blocker_rate':float(np.mean(complete)),
      'nonempty_blocker_records':len(nonempty),
      'nonempty_complete_blocker_rate':sum(complete[i] for i in nonempty)/len(nonempty) if nonempty else None,
      'initial_selected_symmetric_difference':len(set(reference.selected_indices())^set(approximate.selected_indices())),
      'actual_missed_edges':len(truth-found),'sampled_nonretrieved_edges':successes,
      'estimated_missed_edges_HT':successes*m/k if k else 0.,
      'estimate_is_exact_census':k==m,'false_positive_edges':len(found-truth)}
    return approximate,{'config':config,'sampling':sampling,'sample':scored,'metrics':metrics,
      'scope':'secondary_candidate_graph_only; exhaustive graph remains target',
      'all_pair_count':len(x)*(len(x)-1)//2}


# Outward binary64 interval primitives. No fused multiply-add is used.
def down(x):return np.nextafter(np.asarray(x,dtype=np.float64),-np.inf)
def up(x):return np.nextafter(np.asarray(x,dtype=np.float64),np.inf)
def add(a,b):return down(a[0]+b[0]),up(a[1]+b[1])
def mul(a,b):
    corners=np.asarray([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]])
    return down(np.min(corners,axis=0)),up(np.max(corners,axis=0))
def divide_positive(a,b):
    if np.any(b[0]<=0):raise ValueError('Positive denominator interval required')
    corners=np.asarray([a[0]/b[0],a[0]/b[1],a[1]/b[0],a[1]/b[1]])
    return down(np.min(corners,axis=0)),up(np.max(corners,axis=0))


def normalized_intervals(x):
    x=validate_x(x).astype(np.float64);s=(np.zeros(len(x)),np.zeros(len(x)))
    for k in range(x.shape[1]):s=add(s,mul((x[:,k],x[:,k]),(x[:,k],x[:,k])))
    # A nonzero FP32 norm is far above binary64 underflow even if a zero term's
    # outward endpoint is a negative subnormal. Clamp only known sumsq >= 0.
    s=(np.maximum(s[0],0),s[1]);den=(down(np.sqrt(s[0])),up(np.sqrt(s[1])))
    if np.any(den[0]<=0):raise ValueError('Unresolved norm interval; refuse certificate')
    lo,hi=divide_positive((x,x),(den[0][:,None],den[1][:,None]))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():raise ValueError('Nonfinite normalized bounds')
    return lo,hi


def score_intervals(x):
    """Yield every pair exactly once, enclosing real and actual FP64 score.

    At each primitive, rounded endpoint then one outward nextafter contains
    both the exact operation and its round-to-nearest result. Induction covers
    the reference's fixed sequential sum and normalization, including all
    nonretrieved pairs. The assumption on IEEE/sqrt is part of the certificate.
    """
    lo,hi=normalized_intervals(x);n,d=lo.shape
    for i in range(n-1):
        s=(np.zeros(n-i-1),np.zeros(n-i-1))
        for k in range(d):s=add(s,mul((lo[i,k],hi[i,k]),(lo[i+1:,k],hi[i+1:,k])))
        for offset,(l,u) in enumerate(zip(*s)):yield i,i+1+offset,float(l),float(u)


def graph_envelope(x,reference):
    graph_binding(reference)
    if len(x)!=len(reference.record_ids):raise ValueError('Reference graph row alignment')
    lower=set();upper=set();uncertain=[];pair_count=0;max_width=0.;z=graph_normalize(x)
    for i,j,lo,hi in score_intervals(x):
        pair_count+=1;score=float(reference_cosines(z[i:i+1],z[j:j+1])[0,0])
        if not lo<=score<=hi:raise AssertionError('Specified reference escaped interval')
        if lo>reference.threshold:lower.add((i,j))
        if hi>reference.threshold:upper.add((i,j))
        if lo<=reference.threshold<hi:uncertain.append({'i':i,'j':j,'lower':lo,'upper':hi,'reference':score})
        max_width=max(max_width,hi-lo)
    truth=graph_pairs(reference)
    if not lower<=truth<=upper:raise AssertionError('Graph sandwich failed')
    return graph_from_pairs(reference,lower),graph_from_pairs(reference,upper),{
      'all_pairs_certified':pair_count,'lower_edges':len(lower),'upper_edges':len(upper),
      'uncertain_edges':uncertain,'maximum_interval_width':max_width,
      'uniform_scope':'all unordered pairs of fixed stored FP32 rows, same frozen priority and threshold',
      'arithmetic_assumptions':['IEEE754 binary64 round to nearest','gradual underflow; no flush-to-zero',
        'correctly rounded elementary +, *, /, sqrt','NumPy scalar/ufunc evaluation without contraction'],
      'certificate_target':'actual ordered FP64 graph scorer; also encloses exact real cosine of stored values',
      'latent_semantic_similarity_certificate':False,'conditional_on_arithmetic_assumptions':True}


def model_envelope_bound(x,y,w,selected_min,selected_max,lambda_reg):
    """Exact-rational, conservative Frobenius-radius bound for all set choices.

    B <= S <= T. r_B = sum_B[x(y-x'w)-lambda*w]; optional rows add r_v.
    ||w-w*_S||_F <= (||r_B||_1 + sum_(T-B)||r_v||_1)/(lambda*max(1,|B|))
    for nonempty S. If empty S possible, additionally include ||w||_1.
    Each quantity before final display conversion is rational over stored
    values. This is intentionally loose and need not establish a useful bound.
    """
    minimum=set(map(int,selected_min));maximum=set(map(int,selected_max))
    if not minimum<=maximum:raise ValueError('Selection bounds must nest')
    if not maximum<=set(range(len(x))):raise ValueError('Selection index outside rows')
    lam=Fraction(float(lambda_reg))
    if lam<=0:raise ValueError('Positive lambda')
    w=np.asarray(w);y=np.asarray(y)
    if y.ndim==1:y=y[:,None]
    wf=[[Fraction(float(v)) for v in row] for row in w];d,c=w.shape
    residual=[[Fraction(0) for _ in range(c)] for _ in range(d)];optional=Fraction(0)
    for i in sorted(maximum):
        z=[Fraction(float(v)) for v in x[i]];target=[Fraction(float(v)) for v in y[i]]
        err=[target[k]-sum(z[j]*wf[j][k] for j in range(d)) for k in range(c)]
        term=[[z[j]*err[k]-lam*wf[j][k] for k in range(c)] for j in range(d)]
        if i in minimum:
            for j in range(d):
                for k in range(c):residual[j][k]+=term[j][k]
        else:optional+=sum(abs(v) for row in term for v in row)
    numerator=sum(abs(v) for row in residual for v in row)+optional
    radius=numerator/(lam*max(1,len(minimum))) if maximum else Fraction(0)
    if not minimum:radius=max(radius,sum(abs(v) for row in wf for v in row))
    return {'radius_numerator':str(radius.numerator),'radius_denominator':str(radius.denominator),
      'radius_upper_display':float(np.nextafter(float(radius),np.inf)) if radius else 0.,
      'minimum_selected':len(minimum),'maximum_selected':len(maximum),
      'optional_records':len(maximum-minimum),'bound_norm':'Frobenius upper bounded by entrywise L1 residual',
      'empty_target_included':not minimum,'exact_rational_arithmetic':True}


def deleted_indices(path,checkpoint,reference):
    units=set(path['deletion_order'][:checkpoint]);ids=reference.record_ids
    if path['unit']=='record':return {i for i,r in enumerate(ids) if r in units}
    if path['unit']=='source':return {i for i,s in enumerate(reference.source_ids) if s in units}
    raise ValueError('Unknown request service unit')


def moments_ordered(x,y,selected,dtype):
    d=x.shape[1];c=y.shape[1];gram=np.zeros((d,d),dtype=dtype);cross=np.zeros((d,c),dtype=dtype)
    for i in sorted(selected):
        row=np.asarray(x[i],dtype=dtype);target=np.asarray(y[i],dtype=dtype)
        gram+=np.outer(row,row);cross+=np.outer(row,target)
    return gram,cross


def run_frontiers(curator,learner,y,reference,request_manifest,*,lambda_reg=.01,envelope=False):
    """Same locked requests/features; no tuning of lambda or feature projection.

    FP32 frontier is a signed-moment payload reference implementation, not a
    polynomial-summary FP32 implementation or a fast service/no-reaccess claim.
    Evaluation owns all rows. State numeric bytes exclude shared inputs/graph.
    """
    from ccu.core import RidgeMoments
    validate_x(curator);learner=validate_x(learner);y=np.asarray(y)
    if y.ndim==1:y=y[:,None]
    if y.dtype!=np.float64 or y.shape[0]!=len(curator) or not np.isfinite(y).all():raise ValueError('Aligned original FP64 targets required')
    if len(learner)!=len(curator):raise ValueError('Feature alignment')
    if not math.isfinite(lambda_reg) or lambda_reg<=0:raise ValueError('Positive finite lambda')
    verify_manifest(request_manifest,reference)
    ann,ann_audit=audit_ann(curator,reference)
    if envelope:lower,upper,envelope_audit=graph_envelope(curator,reference)
    else:lower=upper=None;envelope_audit={'instantiated':False}
    initial=set(map(int,reference.selected_indices()));rows=[]
    for path in request_manifest['trajectories']:
        previous=initial;gram,cross=moments_ordered(learner,y,initial,np.float32)
        for checkpoint in path['checkpoints']:
            dead=deleted_indices(path,checkpoint,reference);deleted=[reference.record_ids[i] for i in dead]
            exact=set(map(int,reference.selected_indices(deleted)));approx=set(map(int,ann.selected_indices(deleted)))
            for sign,indices in [(-1,previous-exact),(1,exact-previous)]:
                for i in sorted(indices):
                    gram+=np.float32(sign)*np.outer(learner[i],learner[i]);cross+=np.float32(sign)*np.outer(learner[i],y[i].astype(np.float32))
            previous=exact;ix=np.asarray(sorted(exact),dtype=int);ai=np.asarray(sorted(approx),dtype=int)
            exact_mom=ridge_moments(learner[ix],y[ix]);approx_mom=ridge_moments(learner[ai],y[ai])
            for factor in [.1,1.,10.]:
                lam=lambda_reg*factor;w=common_decode(exact_mom,lam).weights;wa=common_decode(approx_mom,lam).weights
                record={'trajectory_id':path['trajectory_id'],'arm':path['arm'],'checkpoint':checkpoint,
                  'dimension':learner.shape[1],'lambda_factor':factor,'lambda':lam,
                  'selected_count':len(exact),'ann_selected_count':len(approx),
                  'ann_selected_symmetric_difference':len(exact^approx),'ann_head_max_abs':float(np.max(np.abs(w-wa),initial=0)),
                  'FP32_numeric_moment_bytes':int(gram.nbytes+cross.nbytes),
                  'FP64_numeric_moment_bytes':int(gram.nbytes+cross.nbytes)*2,
                  'shared_learner_FP32_and_original_target_FP64_bytes':int(learner.nbytes+y.nbytes),
                  'gram_max_abs_error':float(np.max(np.abs(gram.astype(float)-exact_mom.gram),initial=0)),
                  'cross_max_abs_error':float(np.max(np.abs(cross.astype(float)-exact_mom.cross),initial=0))}
                try:
                    if exact:
                        a32=gram+np.eye(len(gram),dtype=np.float32)*np.float32(lam*len(exact))
                        wf=np.linalg.solve(a32,cross).astype(np.float64)
                        af=exact_mom.gram+np.eye(len(gram))*lam*len(exact)
                        residual=af@wf-exact_mom.cross
                        den=np.linalg.norm(af)*np.linalg.norm(wf)+np.linalg.norm(exact_mom.cross)
                        record.update(condition_number_FP64=float(np.linalg.cond(af)),
                          FP32_normalized_residual=float(np.linalg.norm(residual)/den) if den else 0.)
                    else:
                        wf=np.zeros_like(w);record.update(condition_number_FP64=None,FP32_normalized_residual=0.)
                    record.update(status='completed',reference_weights=w.tolist(),ANN_weights=wa.tolist(),FP32_weights=wf.tolist(),FP32_head_max_abs=float(np.max(np.abs(wf-w),initial=0)),
                      FP32_head_fro_error=float(np.linalg.norm(wf-w)),empty_target_explicit_zero=not exact)
                except np.linalg.LinAlgError as error:record.update(status='failed',error=str(error))
                if envelope and factor==1.:
                    sm=set(map(int,upper.selected_indices(deleted)));sx=set(map(int,lower.selected_indices(deleted)))
                    if not sm<=exact<=sx:raise AssertionError('Selection envelope violated')
                    record['graph_envelope_model_bound']=model_envelope_bound(learner,y,w,sm,sx,lam)
                rows.append(record)
    return {'ann':ann_audit,'graph_envelope':envelope_audit,'rows':rows,
      'bindings':{'curator_sha256':array_hash(curator),'learner_sha256':array_hash(learner),'targets_sha256':array_hash(y),
        'request_manifest_sha256':request_manifest['manifest_sha256'],'lambda_reference':lambda_reg},
      'scope':'secondary boundary software integration; no semantic claim, test-set utility estimate, speedup or memory-optimality claim',
      'FP32_variant':'signed dense Gram/cross moments; original FP64 target is rounded at each FP32 contribution',
      'primary_study':False}
