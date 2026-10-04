"""Exhaustive tiny mathematical controls, not NLP data or benchmarks."""
from fractions import Fraction as Q
from itertools import combinations,permutations,product
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from phase10 import finite_word_codec as codec
from phase3.reference_graph import build_reference_graph,stable_priority


def source_hashes():
    names=['phase10/finite_word_codec.py','phase10/check_finite_word_codec.py',
           'phase3/reference_graph.py','phase3/panels.py','ccu/core.py']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def independently_materialize(contract,blocks):
    """Hand-build the stated rows; do not call the codec's vector helper."""
    d=2+contract.k+contract.D;rows=[]
    for index in range(len(contract.record_ids)):
        row=[Q(0)]*d
        if index==0: row[0]=Q(1)
        elif index<=contract.b: row[0]=Q(1,2);row[1]=Q(7,8)
        elif index<=contract.b+contract.m:
            i=index-1-contract.b;row[0]=Q(1,2)
            for j in range(contract.k): row[2+j]=Q(7*contract.signature_signs[i][j],8*(1<<((contract.k.bit_length()-1)//2)))
        else:
            i=index-1-contract.b-contract.m;row[1]=Q(1,2)
            for j in range(contract.k): row[2+j]=Q(7*contract.signature_signs[i][j],8*(1<<((contract.k.bit_length()-1)//2)))
            for j in range(contract.D): row[2+contract.k+j]=Q(blocks[contract.candidate_ids[i]][j],128*(1<<((contract.D.bit_length()-1)//2)))
        rows.append(tuple(row))
    return np.asarray([[float(q) for q in row] for row in rows],np.float32),rows


def exact_ridge_normal_equations(x,y,selected,lam):
    """Independent full Gram/cross construction and rational elimination."""
    d=x.shape[1];n=len(selected)
    if not n:return (Q(0),)*d
    z=[[Q.from_float(float(x[i,j])) for j in range(d)] for i in selected]
    target=[Q.from_float(float(y[i])) for i in selected]
    gram=[[sum((row[i]*row[j] for row in z),Q(0))+(n*lam if i==j else 0) for j in range(d)] for i in range(d)]
    rhs=[sum((row[i]*t for row,t in zip(z,target)),Q(0)) for i in range(d)]
    if not any(rhs):return (Q(0),)*d
    a=[row+[v] for row,v in zip(gram,rhs)]
    for col in range(d):
        pivot=next(i for i in range(col,d) if a[i][col])
        a[col],a[pivot]=a[pivot],a[col];scale=a[col][col]
        a[col]=[v/scale for v in a[col]]
        for i in range(d):
            if i==col:continue
            scale=a[i][col]
            if scale:a[i]=[v-scale*w for v,w in zip(a[i],a[col])]
    answer=tuple(row[-1] for row in a)
    if any(sum((gram[i][j]*answer[j] for j in range(d)),Q(0))!=rhs[i] for i in range(d)):
        raise AssertionError('Independent exact normal equations failed')
    return answer


def oracle(contract,x,y,forgotten):
    keep=[i for i,r in enumerate(contract.record_ids) if r not in forgotten]
    ids=[contract.record_ids[i] for i in keep]
    graph=build_reference_graph(x[keep],ids,0.4,seed=contract.priority_seed,block_size=2)
    selected=[keep[int(i)] for i in graph.selected_indices()]
    return exact_ridge_normal_equations(x,y,selected,contract.lambda_reg),graph,selected


def independent_state_bytes(contract,forgotten,eligible,blocks):
    indices=[i for i,r in enumerate(contract.record_ids) if r in forgotten]
    width=max(1,((len(contract.record_ids)-1).bit_length()+7)//8)
    bits=[int(v==1) for i in eligible for v in blocks[contract.candidate_ids[i]]]
    payload=bytearray((len(bits)+7)//8)
    for i,v in enumerate(bits):payload[i//8]+=v*(2**(i%8))
    header=b'CCUFW01\x00'+hashlib.sha256(contract.public_bytes()).digest()+len(indices).to_bytes(8,'little')
    return header+b''.join(i.to_bytes(width,'little') for i in indices)+payload


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    start=source_hashes();checks=[];counts={'families':0,'retained_sets':0,'exact_heads':0,'sequential_transitions':0,'fresh_graphs':0}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def refuses(name,fn):
        try:fn()
        except (ValueError,TypeError):checks.append(name);return
        raise AssertionError(name)
    report={'schema':'ccu-finite-word-codec-checks-1','status':'failed','source_sha256':start,
            'evidence_role':'tiny_exhaustive_mathematical_software_controls_not_empirical_datasets',
            'natural_corpora_created':0,'human_responses_created':0,'benchmarks_run':0,'primary_study_started':False}
    try:
        signatures=((1,1,1,1),(1,-1,1,-1));all_contracts=[]
        # Full 8-bit and 2-bit private alphabets, not random samples.
        for b,D in ((1,4),(2,1)):
            n=1+b+4;ids=tuple('mathematical-vertex-'+str(i) for i in range(n))
            contract=codec.FiniteSignContract(b,D,signatures,ids,lambda_reg=Q(1),priority_seed=19)
            all_contracts.append(contract);initial_alphabet=set();quotient={}
            if contract.record_ids!=stable_priority(ids,19):raise AssertionError('Actual Phase3 hash priority not used')
            for flat in product((-1,1),repeat=2*D):
                blocks={rid:flat[i*D:(i+1)*D] for i,rid in enumerate(contract.candidate_ids)}
                x,exact_rows=independently_materialize(contract,blocks)
                codec_x,y=codec.materialize_family(contract,blocks)
                if not np.array_equal(x,codec_x):raise AssertionError('Stored vectors differ from independent construction')
                if any(Q.from_float(float(x[i,j]))!=exact_rows[i][j] for i in range(n) for j in range(contract.dimension)):
                    raise AssertionError('FP32 changed exact family coordinates')
                full=build_reference_graph(x,contract.record_ids,0.4,seed=19,block_size=2)
                edge_ids={rid:{contract.record_ids[int(k)] for k in full.blockers[i]} for i,rid in enumerate(contract.record_ids)}
                for i,rid in enumerate(contract.candidate_ids):
                    if edge_ids[rid]!=set(contract.common_ids+(contract.private_blocker_ids[i],)):
                        raise AssertionError('Frozen-score candidate blockers differ')
                initial=codec.initialize_retained(contract,blocks);initial_alphabet.add(initial);counts['families']+=1
                for size in range(contract.budget+1):
                    for f in combinations(contract.record_ids,size):
                        forgotten=set(f);h=contract.budget-size
                        eligible=tuple(i for i,r in enumerate(contract.candidate_ids) if r not in forgotten and len(edge_ids[r]-forgotten)<=h)
                        expected=independent_state_bytes(contract,forgotten,eligible,blocks)
                        state=codec.transition(contract,initial,f)
                        if state!=expected:raise AssertionError('Canonical byte format differs from independent packing')
                        fresh_blocks={r:v for r,v in blocks.items() if r not in forgotten}
                        fresh=codec.initialize_retained(contract,fresh_blocks,deleted_ids=f,remaining_horizon=h)
                        if fresh!=state:raise AssertionError('Sequential state differs from fresh retained initialization at remaining horizon')
                        target,retained_graph,selected=oracle(contract,x,y,forgotten);counts['fresh_graphs']+=1
                        if codec.head_exact(contract,state)!=target:raise AssertionError('Exact head differs from fresh graph/full rational ridge oracle')
                        view=codec.inspect_state(contract,state)
                        if view.payload_bits!=len(eligible)*D or view.remaining_horizon!=h:raise AssertionError('Payload or horizon accounting differs')
                        quotient_key=(tuple(sorted(forgotten)),tuple(blocks[contract.candidate_ids[i]] for i in eligible))
                        if quotient.setdefault(quotient_key,state)!=state:raise AssertionError('Dropped private blocks changed canonical state')
                        for order in permutations(f):
                            service=codec.CanonicalService(contract,initial);prefix=[]
                            for rid in order:
                                prefix.append(rid);service.delete([rid]);counts['sequential_transitions']+=1
                                kept={r:v for r,v in blocks.items() if r not in prefix}
                                fresh_prefix=codec.initialize_retained(contract,kept,deleted_ids=prefix,remaining_horizon=contract.budget-len(prefix))
                                if service.state!=fresh_prefix:raise AssertionError('Order-dependent intermediate bytes')
                            if service.state!=state or service.head()!=target:raise AssertionError('Batch/order-dependent endpoint')
                        counts['retained_sets']+=1;counts['exact_heads']+=1
            check('full_initial_alphabet_'+str(2*D)+'_bits',len(initial_alphabet)==2**(2*D))
        check('every_fresh_graph_exact_head_and_canonical_history_passed',counts['retained_sets']==counts['exact_heads']==counts['fresh_graphs'])
        check('full_finite_alphabets_exhausted',counts['families']==260)
        contract=all_contracts[1];blocks={r:(1,) for r in contract.candidate_ids};initial=codec.initialize_retained(contract,blocks)
        metrics=codec.accounting(contract,initial)
        check('initial_private_payload_exactly_mD',metrics['private_sign_payload_bits']==contract.m*contract.D)
        check('padding_is_charged_not_hidden',metrics['payload_padding_bits']==6 and metrics['private_sign_payload_bytes']==1)
        check('all_serialized_state_bytes_accounted',metrics['serialized_state_bytes']==metrics['private_sign_payload_bytes']+metrics['deleted_index_metadata_bytes']+metrics['version_contract_binding_and_count_bytes'])
        after_common=codec.transition(contract,initial,contract.common_ids[:1])
        after_private=codec.transition(contract,after_common,[contract.private_blocker_ids[1]])
        check('partial_private_deletion_keeps_only_its_candidate',codec.inspect_state(contract,after_private).eligible_indices==(1,))
        check('partial_state_has_zero_current_head',not any(codec.head_exact(contract,after_private)))
        check('public_only_prefix_retains_every_candidate',codec.inspect_state(contract,after_common).eligible_indices==(0,1))
        for label,removed in [('anchor',[contract.anchor_id]),('candidate',[contract.candidate_ids[0]]),('two_private',contract.private_blocker_ids)]:
            state=codec.transition(contract,initial,removed)
            check(label+'_deletion_drops_all_private_bits',codec.inspect_state(contract,state).payload_bits==0)
        activated=codec.transition(contract,initial,contract.common_ids+(contract.private_blocker_ids[1],))
        head=codec.head_exact(contract,activated)
        check('exact_head_is_not_claimed_FP64',any(Q.from_float(float(v))!=v for v in head))
        check('output_materialization_charged_separately',codec.materialized_head_accounting(contract,activated)['canonical_fraction_pair_JSON_bytes']>0)
        float_contract=codec.FiniteSignContract(contract.b,contract.D,signatures,contract.record_ids,lambda_reg=0.1,priority_seed=19)
        check('stored_FP64_lambda_is_exact_ratio',float_contract.lambda_reg==Q.from_float(0.1) and float_contract.lambda_reg!=Q(1,10))
        service=codec.CanonicalService(contract,initial)
        for name,request in [('unknown',['outside-mathematical-universe']),('duplicate',[contract.anchor_id]*2),('text',contract.anchor_id),
                             ('overbudget',contract.record_ids[:contract.budget+1])]:
            before=service.state;refuses('atomic_'+name+'_refusal',lambda request=request:service.delete(request))
            check(name+'_refusal_preserves_exact_old_bytes',service.state is before)
        service.delete([contract.common_ids[0]]);before=service.state
        refuses('repeated_historical_deletion_refused',lambda:service.delete([contract.common_ids[0]]))
        check('historical_refusal_atomic',service.state is before)
        refuses('cumulative_overbudget_refused',lambda:service.delete([r for r in contract.record_ids if r!=contract.common_ids[0]][:contract.budget]))
        check('cumulative_refusal_atomic',service.state is before)
        check('empty_request_is_byte_identity',service.delete([]) is before)
        refuses('mutable_state_refused',lambda:codec.inspect_state(contract,bytearray(initial)))
        refuses('truncated_state_refused',lambda:codec.inspect_state(contract,initial[:-1]))
        refuses('appended_state_bytes_refused',lambda:codec.inspect_state(contract,initial+b'\x00'))
        refuses('nonzero_padding_refused',lambda:codec.inspect_state(contract,initial[:-1]+bytes([initial[-1]|128])))
        refuses('different_public_contract_refused',lambda:codec.inspect_state(float_contract,initial))
        bad=bytearray(activated);bad[48:51]=bytes([2,1,1])
        refuses('noncanonical_deletion_history_refused',lambda:codec.inspect_state(contract,bytes(bad)))
        bad=bytearray(initial);bad[40:48]=struct.pack('<Q',contract.budget+1)
        refuses('serialized_overbudget_count_refused',lambda:codec.inspect_state(contract,bytes(bad)))
        refuses('remaining_horizon_cannot_be_reset',lambda:codec.initialize_retained(contract,blocks,deleted_ids=contract.common_ids[:1],remaining_horizon=contract.budget))
        invalid=dict(blocks);invalid[contract.candidate_ids[0]]=(0,)
        refuses('fresh_initializer_validates_even_ineligible_survivors',lambda:codec.initialize_retained(contract,invalid,deleted_ids=[contract.anchor_id]))
        one_deleted=contract.candidate_ids[0];remaining={r:v for r,v in blocks.items() if r!=one_deleted}
        check('deleted_candidate_payload_not_required',codec.inspect_state(contract,codec.initialize_retained(contract,remaining,deleted_ids=[one_deleted])).payload_bits==0)
        refuses('deleted_private_payload_refused_by_fresh_interface',lambda:codec.initialize_retained(contract,blocks,deleted_ids=[one_deleted]))
        # Input mutation cannot change the service; update/decoding does not
        # invoke retained initializer or the full private materializer.
        service=codec.CanonicalService(contract,initial);blocks.clear()
        with patch.object(codec,'initialize_retained',side_effect=AssertionError('retained reread')),patch.object(codec,'materialize_family',side_effect=AssertionError('full cache access')):
            service.delete(contract.common_ids+(contract.private_blocker_ids[1],));actual=service.head()
        check('transition_and_head_need_no_retained_cache',actual==head)
        check('service_has_only_contract_and_authoritative_bytes',codec.CanonicalService.__slots__==('contract','_state'))
        refuses('non_power_four_hidden_dimension_refused',lambda:codec.FiniteSignContract(1,2,signatures,tuple('v'+str(i) for i in range(6))))
        refuses('coherent_duplicate_public_signatures_refused',lambda:codec.FiniteSignContract(1,1,(signatures[0],signatures[0]),tuple('v'+str(i) for i in range(6))))
        refuses('outside_proved_scorer_dimension_refused',lambda:codec.FiniteSignContract(1,4**10,signatures,tuple('v'+str(i) for i in range(6))))
        class Substr(str):pass
        refuses('identifier_subclass_consistently_refused',lambda:codec.FiniteSignContract(1,1,signatures,tuple(Substr('v'+str(i)) for i in range(6))))
        big=codec.FiniteSignContract(255,1,signatures,tuple('width-math-vertex-'+str(i) for i in range(260)))
        state=codec.initialize_retained(big,{r:(1,) for r in big.candidate_ids})
        state=codec.transition(big,state,[big.record_ids[-1]])
        check('multibyte_deleted_index_layout',big.index_bytes==2 and codec.inspect_state(big,state).deleted_indices==(259,))
        check('public_storage_is_explicit',metrics['public_contract_serialized_bytes']>0 and metrics['public_full_graph_CSR_int64_bytes_if_materialized']>0)
        check('no_total_memory_or_erasure_claim',metrics['total_byte_optimality_claim'] is False and metrics['physical_erasure_claim'] is False)
        check('all_bound_sources_unchanged',source_hashes()==start)
        report.update(status='passed',check_count=len(checks),checks=checks,counts=counts,
                      immutable_state_format='CCUFW01; little-endian metadata; LSB-first sign bits; zero tail padding',
                      examples_accounting=metrics,numpy_version=np.__version__)
    except Exception as error:
        report.update(check_count=len(checks),checks=checks,counts=counts,exception_type=type(error).__name__,reason=str(error))
    out.parent.mkdir(parents=True,exist_ok=True)
    history=out.parent/(out.stem+'_sources');history.mkdir(exist_ok=False)
    for name in ('finite_word_codec.py','check_finite_word_codec.py'):
        (history/name).write_bytes((ROOT/'phase10'/name).read_bytes())
    report['source_snapshot']=str(history.relative_to(ROOT))
    with out.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    value=run(p.parse_args().out)
    print(json.dumps({k:v for k,v in value.items() if k not in {'source_sha256','checks','examples_accounting'}},indent=2))
    sys.exit(value['status']!='passed')
