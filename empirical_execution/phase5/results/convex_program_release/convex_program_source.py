"""Three-arm convex program with an actual horizon-eligible payload service.

Fresh retained curation, eligible-payload cold retraining and eligible-payload
warm certified repair share the frozen multioutput solver/certificate. Explicit
local inputs are supported; this module never self-authenticates provenance or
unlocks the absent study-wide primary acceptance gate.
"""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,io,json,time,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from phase3.reference_graph import build_reference_graph
from phase4.requests import verify_manifest,digest
from phase5 import payload as packed
from phase5 import convex_multioutput as convex

METHODS=('fresh_optimum','eligible_payload_retraining','certified_convex_repair')

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def code_hashes():
    return {Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (packed,convex,convex.scalar)}|{'convex_program.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

class LogisticPayload:
    """Packed eligibility with genuine retained features/labels and no ridge moments.

    The frozen CSR kernel receives one zero dummy coordinate so that its signed
    moment routine never builds d-by-d logistic-irrelevant moments. The real
    eligible payload is owned separately and moves with the kernel's compaction.
    Dummy arrays, transient compaction copies and actual selected gathers count.
    """
    def __init__(self,graph,features,targets,horizon,unit,compaction_fraction=.5):
        x,y,_=convex._inputs(features,targets)
        self.counter=packed.PayloadState(graph,np.zeros((len(x),1),np.float32),np.zeros((len(x),1),np.float64),
          horizon,unit=unit,retain_all=False,compaction_fraction=compaction_fraction)
        lookup={rid:i for i,rid in enumerate(graph.record_ids)}
        ix=np.asarray([lookup[r] for r in self.counter.record_ids],dtype=int)
        self.x=x[ix].copy();self.y=y[ix].copy()
        self.meter=dict(real_payload_gathered_rows=0,real_payload_gathered_bytes=0,
          real_payload_compaction_read_bytes=0,real_payload_compaction_written_bytes=0,
          construction_dummy_array_bytes=len(x)*12,constructor_payload_read_bytes=self.x.nbytes+self.y.nbytes)

    def delete(self,units):
        old_ids=self.counter.record_ids;result=self.counter.delete(units)
        if old_ids!=self.counter.record_ids:
            index={rid:i for i,rid in enumerate(old_ids)};ix=np.asarray([index[r] for r in self.counter.record_ids],dtype=int)
            self.meter['real_payload_compaction_read_bytes']+=len(ix)*(4*self.x.shape[1]+8*self.y.shape[1])
            self.x=self.x[ix].copy();self.y=self.y[ix].copy()
            self.meter['real_payload_compaction_written_bytes']+=self.x.nbytes+self.y.nbytes
        return result

    def selected_payload(self):
        ix=np.flatnonzero(self.counter.selected);x=self.x[ix].copy();y=self.y[ix].copy()
        self.meter['real_payload_gathered_rows']+=len(ix);self.meter['real_payload_gathered_bytes']+=x.nbytes+y.nbytes
        return x,y,[self.counter.record_ids[i] for i in ix]

    def accounting(self):
        base=self.counter.accounting();live=int(np.count_nonzero(self.counter.live));width=4*self.x.shape[1]+8*self.y.shape[1]
        return dict(eligibility_counter=base,real_payload_allocated_bytes=int(self.x.nbytes+self.y.nbytes),
          real_payload_live_bytes=live*width,real_payload_stale_bytes=(len(self.x)-live)*width,
          real_feature_dimension=self.x.shape[1],real_outputs=self.y.shape[1],meter=dict(self.meter),
          real_payload_touches_required=True,raw_constructor_arrays_retained=False,
          graph_or_curator_vectors_retained=False,logical_deletion_not_physical_erasure=True,
          peak_rss_measured=False)

    def snapshot(self,path,weights):
        w=np.asarray(weights)
        if w.dtype!=np.float64 or w.shape!=(self.x.shape[1],self.y.shape[1]):raise ValueError('aligned head required')
        start=time.perf_counter();counter=self.counter.snapshot_bytes();buf=io.BytesIO()
        np.savez(buf,counter=np.frombuffer(counter,dtype=np.uint8),features=self.x,targets=self.y,weights=w,
          metadata=np.frombuffer(json.dumps(dict(schema='ccu-convex-payload-snapshot-v1',meter=self.meter),sort_keys=True).encode(),dtype=np.uint8))
        raw=buf.getvalue();path=Path(path)
        with path.open('xb') as handle:handle.write(raw)
        return dict(file_sha256=hashlib.sha256(raw).hexdigest(),serialized_bytes=len(raw),write_seconds=time.perf_counter()-start,
          serialization_buffer_bytes=len(raw)+len(counter),includes_tombstones=True)

    @classmethod
    def load(cls,path,*,expected_sha256):
        start=time.perf_counter();raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('changed payload snapshot')
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            if set(archive.files)!={'counter','features','targets','weights','metadata'}:raise ValueError('snapshot members mismatch')
            meta=json.loads(archive['metadata'].tobytes())
            if meta.get('schema')!='ccu-convex-payload-snapshot-v1':raise ValueError('snapshot schema mismatch')
            obj=cls.__new__(cls);obj.x=archive['features'].copy();obj.y=archive['targets'].copy();weights=archive['weights'].copy();obj.meter=meta['meter']
            counter_raw=archive['counter'].tobytes()
        with np.load(io.BytesIO(counter_raw),allow_pickle=False) as archive:
            if set(archive.files)!=set(packed._ARRAYS)|{'metadata'}:raise ValueError('counter snapshot members mismatch')
            metadata=json.loads(archive['metadata'].tobytes())
            if metadata.pop('schema',None)!=packed.SCHEMA:raise ValueError('counter schema mismatch')
            counter=packed.PayloadState.__new__(packed.PayloadState)
            for key,value in metadata.items():setattr(counter,key,value)
            counter.unit_ids=tuple(counter.unit_ids);counter.record_ids=tuple(counter.record_ids)
            counter._unit_lookup={rid:i for i,rid in enumerate(counter.unit_ids)}
            for name in packed._ARRAYS:setattr(counter,name,archive[name].copy())
        counter.check_invariants(check_moments=True);obj.counter=counter
        convex._inputs(obj.x,obj.y,weights)
        if len(obj.x)!=len(counter.record_ids):raise ValueError('real payload and eligibility rows differ')
        return obj,weights,dict(serialized_bytes=len(raw),read_seconds=time.perf_counter()-start,
          deserialization_buffer_bytes=len(raw)+len(counter_raw),original_inputs_reopened=False)


def _fit(x,y,lam,warm,eval_x,eval_y,tolerance,certificate_budget):
    result=convex.solve_multioutput(x,y,lam,warm_start=warm)
    cert=convex.certify_multioutput(x,y,lam,result['weights'],parameter_tolerance=tolerance,max_coordinates=certificate_budget)
    pred=eval_x.astype(np.float64)@result['weights'];p=convex.scalar.expit(pred)
    signed=np.where(pred>=0,np.logaddexp(0,-pred)+(1-eval_y)*pred,np.logaddexp(0,pred)-eval_y*pred)
    if not np.isfinite(pred).all() or not np.isfinite(signed).all():raise FloatingPointError('nonfinite declared evaluation output')
    return dict(solver={k:v for k,v in result.items() if k!='weights'},weights=result['weights'].tolist(),
      certificate=cert,released=bool(cert['meets_parameter_tolerance']),
      evaluation=dict(fractional_bce_mean_over_rows_and_outputs=float(np.mean(signed)),
        fractional_target_probability_mse=float(np.mean((p-eval_y)**2)),
        score_logits=pred.tolist(),probabilities=p.tolist(),rows=len(eval_x),
        forward_coordinate_output_operations=len(eval_x)*eval_x.shape[1]*eval_y.shape[1]))

def run_convex_program(curator_fp32,learner_fp32,targets_fp64,record_ids,source_ids,source_kinds,
                       request_manifest,out_dir,*,lambda_reg,threshold,eval_features,eval_targets,
                       allocations=None,evidence_role='engineering_nonconfirmatory',primary_dossier=None,
                       evaluation_scope='declared_evaluation_not_automatically_held_out',
                       parameter_tolerance=Fraction(1,10**8),certificate_budget=2_000_000,
                       compaction_fraction=.5):
    """Execute first32R/32S by stable trajectory ID unless an engineering allocation is explicit.

    IDs/whole-source coverage, request seals, graph identity and every current
    selected set are checked. Missing arms/paths are status rows, never replacements.
    Primary execution remains refused until an external full-study activator exists.
    """
    if evidence_role!='engineering_nonconfirmatory':
        raise ValueError('Primary study activation/acceptance is not established by this driver; primary execution refused')
    if primary_dossier is not None:raise ValueError('No dossier bypass of missing primary activation is accepted')
    x,y,_=convex._inputs(learner_fp32,targets_fp64);ex,ey,_=convex._inputs(eval_features,eval_targets)
    cx=np.asarray(curator_fp32);ids=list(record_ids);sources=list(source_ids)
    if not len(ex) or ex.shape[1]!=x.shape[1] or ey.shape[1]!=y.shape[1]:raise ValueError('nonempty aligned declared evaluation inputs required')
    if len(ids)!=len(x) or len(sources)!=len(ids) or set(source_kinds)!=set(sources):raise ValueError('complete IDs/source-kind map required')
    if any(v not in {'genuine_native','unknown_singleton','engineering_proxy'} for v in source_kinds.values()):raise ValueError('source kind invalid')
    lam=convex.scalar._positive(lambda_reg,'lambda_reg');tol=convex.scalar.rational(parameter_tolerance)
    if tol<0:raise ValueError('nonnegative tolerance required')
    if type(certificate_budget)!=int or certificate_budget<0:raise ValueError('nonnegative integer certificate budget required')
    allocations=dict({'R':32,'S':32} if allocations is None else allocations)
    if set(allocations)-{'R','S'} or any(type(n)!=int or n<0 for n in allocations.values()):raise ValueError('explicit R/S counts required')
    for arm in ('R','S'):allocations.setdefault(arm,0)
    start=time.perf_counter();graph=build_reference_graph(cx,ids,threshold,source_ids=sources);verify_manifest(request_manifest,graph)
    graph_seconds=time.perf_counter()-start;out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    bound=dict(curator=convex.scalar._hash_array(cx),learner=convex.scalar._hash_array(x),targets=convex.scalar._hash_array(y),
      evaluation_features=convex.scalar._hash_array(ex),evaluation_targets=convex.scalar._hash_array(ey),
      record_ids=digest(ids),source_ids=digest(sources),source_kinds=digest(source_kinds),requests=request_manifest['manifest_sha256'])
    lock=dict(schema='ccu-convex-program-1',evidence_role=evidence_role,input_bindings=bound,code_sha256=code_hashes(),
      allocations=allocations,methods=list(METHODS),lambda_reg=lam,threshold=threshold,objective='sum_output_row_mean_BCE_plus_lambda_half_Frobenius_squared',
      parameter_tolerance=convex.scalar._fraction_json(tol),certificate_budget=certificate_budget,
      evaluation_scope=evaluation_scope,shared_graph_seconds=graph_seconds,
      compaction_fraction=compaction_fraction,persistence='final_state_per_trajectory_each_payload_method',
      process_isolation=False,paper_systems_claim=False,primary_study_started=False,
      source_provenance_authenticated=False,all_local=True)
    write(out/'lock.json',lock);write(out/'requests.json',request_manifest)
    initial=sorted(ids[i] for i in graph.selected_indices());lookup={rid:i for i,rid in enumerate(ids)}
    selected_paths=[];missing=[]
    for arm,count in allocations.items():
        paths=sorted((p for p in request_manifest['trajectories'] if p['arm']==arm),key=lambda p:p['trajectory_id'])[:count]
        if arm=='S':
            for path in paths:
                if path['unit']!='source' or any(source_kinds[s]!='genuine_native' for s in path['deletion_order']):
                    raise ValueError('S paths require declared genuine source IDs; unknown/proxy source substitution refused')
        selected_paths.extend(paths)
        missing.extend(dict(arm=arm,planned_index=i,status='not_executed_missing_native_sources' if arm=='S' and not any(v=='genuine_native' for v in source_kinds.values()) else 'not_executed_missing_frozen_trajectory')
                       for i in range(len(paths),count))
    write(out/'missing_trajectories.json',missing);results=[];failures=[];certificates=[]
    for path in selected_paths:
        run=out/path['trajectory_id'];run.mkdir();unit=path['unit'];horizon=len(path['deletion_order'])
        if unit not in ('record','source'):raise ValueError('request unit invalid')
        stages={};states={};warm={};initial_reports={}
        for method in METHODS:
            stage=time.perf_counter()
            if method=='fresh_optimum':
                indices=np.asarray([lookup[r] for r in initial],dtype=int);fx,fy=x[indices],y[indices]
                stages[method]=dict(state_build_seconds=0.,retained_raw_payload_bytes=int(cx.nbytes+x.nbytes+y.nbytes))
            else:
                states[method]=LogisticPayload(graph,x,y,horizon,unit,compaction_fraction=compaction_fraction)
                built=time.perf_counter();fx,fy,sel=states[method].selected_payload()
                assert set(sel)==set(initial)
                stages[method]=dict(state_build_seconds=built-stage,accounting=states[method].accounting())
            try:
                fit=_fit(fx,fy,lam,None,ex,ey,tol,certificate_budget);initial_reports[method]=fit
                certificates.append(fit['certificate']);warm[method]=np.asarray(fit['weights'],dtype=np.float64)
                if not fit['released']:failures.append(dict(trajectory=path['trajectory_id'],method=method,checkpoint=0,status='initial_certificate_failed'))
            except Exception as error:
                initial_reports[method]=dict(released=False,error_type=type(error).__name__,error=str(error))
                failures.append(dict(trajectory=path['trajectory_id'],method=method,checkpoint=0,status='initial_fit_failed'))
            stages[method]['initial_total_seconds']=time.perf_counter()-stage
        write(run/'initial.json',dict(selected_ids=initial,methods=initial_reports,stages=stages))
        blocked={m:not initial_reports[m].get('released',False) for m in METHODS};last=0
        for checkpoint in path['checkpoints']:
            prefix=set(path['deletion_order'][:checkpoint]);dead=prefix if unit=='record' else {r for r,s in zip(ids,sources) if s in prefix}
            alive=np.asarray([i for i,r in enumerate(ids) if r not in dead],dtype=int)
            # Independent retained graph, outside the two payload services.
            stage=time.perf_counter();fresh=build_reference_graph(cx[alive],[ids[i] for i in alive],threshold,
              source_ids=[sources[i] for i in alive]);sel_ix=alive[fresh.selected_indices()]
            selected_ids=[ids[i] for i in sel_ix];fresh_seconds=time.perf_counter()-stage
            oracle_gather_bytes=int(cx[alive].nbytes+x[sel_ix].nbytes+y[sel_ix].nbytes)
            row=dict(trajectory=path['trajectory_id'],arm=path['arm'],checkpoint=checkpoint,
              selected_ids=sorted(selected_ids),deleted_ids=sorted(dead),methods={})
            for method in METHODS:
                if blocked[method]:row['methods'][method]=dict(status='not_executed_after_method_failure');continue
                stage=time.perf_counter();cost={}
                try:
                    if method=='fresh_optimum':
                        fx,fy=x[sel_ix],y[sel_ix];cost=dict(curator_rebuild_seconds=fresh_seconds,oracle_gather_bytes=oracle_gather_bytes,
                          retained_raw_records=len(alive),selected_records=len(fx))
                    else:
                        cost['maintenance']=states[method].delete(path['deletion_order'][last:checkpoint])
                        fx,fy,selected=states[method].selected_payload()
                        if set(selected)!=set(selected_ids):raise AssertionError('eligible service differs from fresh retained curation')
                        cost['accounting']=states[method].accounting()
                    fit=_fit(fx,fy,lam,warm[method] if method=='certified_convex_repair' else None,ex,ey,tol,certificate_budget)
                    certificates.append(fit['certificate']);cost['method_seconds_before_output']=time.perf_counter()-stage
                    row['methods'][method]=dict(status='released' if fit['released'] else 'certificate_failed',**fit,cost=cost)
                    if fit['released']:warm[method]=np.asarray(fit['weights'],dtype=np.float64)
                    else:blocked[method]=True;failures.append(dict(trajectory=path['trajectory_id'],method=method,checkpoint=checkpoint,status='certificate_failed'))
                except Exception as error:
                    blocked[method]=True;row['methods'][method]=dict(status='failed',error_type=type(error).__name__,error=str(error),cost=cost)
                    failures.append(dict(trajectory=path['trajectory_id'],method=method,checkpoint=checkpoint,status='failed'))
            successful=[v for v in row['methods'].values() if v.get('status')=='released']
            for first in range(len(successful)):
                for second in range(first):
                    a,b=successful[first],successful[second]
                    squared=sum(((Fraction.from_float(float(u))-Fraction.from_float(float(v)))**2
                      for u,v in zip(np.asarray(a['weights']).flat,np.asarray(b['weights']).flat)),Fraction(0))
                    radius=convex.fraction(a['certificate']['parameter_error_frobenius_upper'])+convex.fraction(b['certificate']['parameter_error_frobenius_upper'])
                    if squared>radius**2:raise AssertionError('same-target candidate distance exceeds certified sum')
            row['released_head_distance_checks_passed']=True if len(successful)>=2 else None
            write(run/f'checkpoint_{checkpoint}.json',row);results.append(row);last=checkpoint
        snapshots={}
        for method,state in states.items():
            if method in warm:
                pathout=run/(method+'.npz');receipt=state.snapshot(pathout,warm[method]);reloaded,rw,read=LogisticPayload.load(pathout,expected_sha256=receipt['file_sha256'])
                if reloaded.counter.selected_ids()!=state.counter.selected_ids() or not np.array_equal(rw,warm[method]):raise AssertionError('payload/head resume mismatch')
                snapshots[method]=dict(write=receipt,read=read,accounting=reloaded.accounting(),
                  warm_head_released_for_current_target=not blocked[method],
                  candidate_only_if_method_failed=bool(blocked[method]))
        write(run/'final_snapshots.json',snapshots)
    summary=dict(status='completed_engineering' if not failures else 'completed_with_preserved_failures',
      planned_trajectories=sum(allocations.values()),executed_trajectories=len(selected_paths),missing_trajectories=len(missing),
      checkpoint_rows=len(results),method_releases=sum(v.get('status')=='released' for r in results for v in r['methods'].values()),
      failures=failures,joint_certificates=len(certificates),scalar_certificates=sum(c['outputs'] for c in certificates),
      max_certificate_radius_display=max((c['radius_float_for_display_only'] for c in certificates),default=None),
      primary_study_started=False,source_authenticity_verified=False,semantic_quality_established=False,
      speedup_claim=False,process_isolation_claim=False,evaluation_scope=evaluation_scope,lock_sha256=hashlib.sha256((out/'lock.json').read_bytes()).hexdigest())
    write(out/'summary.json',summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    bundle=json.loads(args.bundle.read_text());base=args.bundle.resolve().parent
    def load(name):
        spec=bundle[name];path=(base/spec['path']).resolve()
        if not path.is_relative_to(base) or hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('array path/hash binding mismatch')
        return np.load(path,allow_pickle=False)
    result=run_convex_program(load('curator'),load('learner'),load('targets'),bundle['record_ids'],bundle['source_ids'],bundle['source_kinds'],
      bundle['request_manifest'],args.out,lambda_reg=bundle['lambda_reg'],threshold=bundle['threshold'],
      eval_features=load('evaluation_features'),eval_targets=load('evaluation_targets'),allocations=bundle.get('allocations'),
      evidence_role=bundle.get('evidence_role','engineering_nonconfirmatory'),evaluation_scope=bundle['evaluation_scope'])
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
