"""Independent bounded QA of actual codec bytes; mathematical software only."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,product
from collections import Counter
from copy import deepcopy
from unittest.mock import patch
import argparse,hashlib,json,struct,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from phase10 import finite_word_codec as codec
from phase3.reference_graph import build_reference_graph


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():
    names=('phase10/finite_word_codec.py','phase10/check_codec_review.py','phase10/SCORER_TRANSFER.md','phase10/FINITE_WORD_THEORY.md','phase10/CODEC.md','phase10/CODEC_REVIEW.md','phase3/panels.py','phase3/reference_graph.py')
    return {name:sha(ROOT/name) for name in names}


def make_contract(b=2,D=4,m=3,lam=F(1,3)):
    signatures=tuple(tuple(1 if (i&j).bit_count()%2==0 else -1 for j in range(4)) for i in range(m))
    ids=tuple('codec-review-algebra-'+str(i) for i in reversed(range(1+b+2*m)))
    return codec.FiniteSignContract(b,D,signatures,ids,lam,7)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);bindings=sources();captured={name:(ROOT/name).read_bytes() for name in bindings};checks=[];counts=Counter()
    report={'schema':'ccu-independent-codec-review-1','status':'failed','source_sha256':bindings,
        'evidence_role':'bounded_exact_algebra_and_byte_contract_checks_only','empirical_benchmark':False,'primary_study_started':False}
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except (TypeError,ValueError):checks.append(name);return
        raise AssertionError('Not rejected: '+name)
    try:
        for D in (1,4,16):
            c=make_contract(D=D)
            for pattern in range(4):
                blocks={rid:tuple(1 if ((pattern+i*3+j).bit_count()%2)==0 else -1 for j in range(D)) for i,rid in enumerate(c.candidate_ids)}
                initial=codec.initialize_retained(c,blocks);x,y=codec.materialize_family(c,blocks)
                # Independent target uses actual frozen graph and exact values
                # converted from the stored FP32 matrix, not head_exact's formula.
                graph=build_reference_graph(x,c.record_ids,codec.THRESHOLD,seed=c.priority_seed)
                exact=tuple(tuple(F.from_float(float(v)) for v in row) for row in x)
                for size in range(c.budget+1):
                    for Fids in combinations(c.record_ids,size):
                        forgotten=set(Fids);state=codec.transition(c,initial,Fids)
                        expected_eligible=tuple(i for i,rid in enumerate(c.candidate_ids) if rid not in forgotten and len((set(c.common_ids)|{c.private_blocker_ids[i]})-forgotten)<=c.budget-size)
                        view=codec.inspect_state(c,state)
                        assert view.deleted_indices==tuple(i for i,rid in enumerate(c.record_ids) if rid in forgotten)
                        assert view.eligible_indices==expected_eligible and view.remaining_horizon==c.budget-size
                        retained={rid:word for rid,word in blocks.items() if rid not in forgotten}
                        assert state==codec.initialize_retained(c,retained,deleted_ids=Fids,remaining_horizon=c.budget-size)
                        for order in (Fids,tuple(reversed(Fids))):
                            service=codec.CanonicalService(c,initial)
                            for rid in order:service.delete([rid])
                            assert service.state==state
                            # Save/reload an immutable prefix, then continue.
                            prefix=order[:len(order)//2];suffix=order[len(order)//2:]
                            service=codec.CanonicalService(c,codec.transition(c,initial,prefix));service.delete(suffix)
                            assert service.state==state
                        head=codec.head_exact(c,state)
                        selected=tuple(map(int,graph.selected_indices(Fids)));n=len(selected)
                        predictions={i:sum((a*b for a,b in zip(exact[i],head)),F(0)) for i in selected}
                        for coordinate in range(c.dimension):
                            residual=sum((exact[i][coordinate]*(predictions[i]-F(int(y[i]))) for i in selected),F(0))+c.lambda_reg*n*head[coordinate]
                            assert residual==0
                        if n==0:assert not any(head)
                        accounting=codec.accounting(c,state)
                        assert len(state)==codec.HEADER_BYTES+size*c.index_bytes+(len(expected_eligible)*D+7)//8
                        assert accounting['private_sign_payload_bits']==len(expected_eligible)*D
                        assert accounting['payload_padding_bits']==8*((len(expected_eligible)*D+7)//8)-len(expected_eligible)*D
                        assert accounting['public_full_graph_CSR_int64_bytes_if_materialized']==graph.priority_indices.nbytes+graph.indptr.nbytes+graph.indices.nbytes
                        counts['state_cases']+=1;counts['exact_normal_equation_coordinates']+=c.dimension
                counts['private_word_patterns']+=1
        check('all_retained_rebuild_and_history_bytes_agree',counts['state_cases']>0)
        check('all_actual_graph_target_normal_equations_exact',counts['exact_normal_equation_coordinates']>0)
        check('packing_across_bit_nibble_byte_boundaries',counts['private_word_patterns']==12)
        check('complete_logical_state_and_graph_byte_accounting',True)
        # Exhaust the smallest nontrivial private alphabet to check conditional
        # state images, independently of the higher-dimensional pattern checks.
        c=make_contract(b=1,D=1,m=2);images={f:set() for size in range(c.budget+1) for f in combinations(c.record_ids,size)}
        for signs in product((-1,1),repeat=2):
            words={rid:(signs[i],) for i,rid in enumerate(c.candidate_ids)};state=codec.initialize_retained(c,words)
            for f in images:images[f].add(codec.transition(c,state,f))
        for f,states in images.items():
            e=sum(set(f)<=set(c.common_ids+(p,)) for p in c.private_blocker_ids)
            assert len(states)==2**e
        check('conditional_state_alphabets_exact_in_small_exhaustive_family',True)
        c=make_contract();words={rid:[1,-1,1,-1] for rid in c.candidate_ids};state=codec.initialize_retained(c,words)
        private=codec.transition(c,state,[c.private_blocker_ids[1]])
        check('partial_disposal_keeps_only_still_reachable_candidate',codec.inspect_state(c,private).eligible_indices==(1,) and codec.inspect_state(c,private).payload_bits==c.D)
        commons=codec.transition(c,state,c.common_ids)
        check('only_common_deletions_keep_every_private_word',codec.inspect_state(c,commons).payload_bits==c.m*c.D)
        for role,request in [('anchor',[c.anchor_id]),('candidate',[c.candidate_ids[0]]),('two_private',c.private_blocker_ids[:2])]:
            dead=codec.transition(c,state,request)
            check(role+'_deletion_discards_all_future_private_bits',codec.inspect_state(c,dead).payload_bits==0 and not any(codec.head_exact(c,dead)))
        activated=codec.transition(c,private,c.common_ids)
        check('zero_remaining_horizon_keeps_current_nonzero_head_word',codec.inspect_state(c,activated).remaining_horizon==0 and codec.inspect_state(c,activated).payload_bits==c.D and any(codec.head_exact(c,activated)))
        changed=deepcopy(words)
        for rid in (c.candidate_ids[0],c.candidate_ids[2]):changed[rid]=[-v for v in changed[rid]]
        check('discarded_private_words_cannot_change_canonical_bytes',codec.transition(c,codec.initialize_retained(c,changed),[c.private_blocker_ids[1]])==private)
        service=codec.CanonicalService(c,state);before=service.state
        for rid in words:words[rid][:]=[-1]*c.D
        with patch.object(codec,'initialize_retained',side_effect=AssertionError('forbidden retained reread')),patch.object(codec,'materialize_family',side_effect=AssertionError('forbidden original cache')):
            service.delete(c.common_ids+(c.private_blocker_ids[1],));service.head()
        check('IDs_only_update_does_not_call_input_initializers_or_oracles',service.state==activated)
        check('original_input_mutation_has_no_effect_on_existing_state',before==state)
        check('service_retains_only_declared_public_contract_and_bytes',codec.CanonicalService.__slots__==('contract','_state') and type(service.state) is bytes)
        # Every rejected update leaves the same original bytes object.
        service=codec.CanonicalService(c,state)
        for name,request in [('unknown',['not-in-public-universe']),('duplicate',[c.anchor_id,c.anchor_id]),('nonstr',[0]),('text',c.anchor_id),('overbudget',c.record_ids[:c.budget+1])]:
            old=service.state;reject(name+'_request_rejected',lambda request=request:service.delete(request));assert service.state is old
        service.delete([c.anchor_id]);old=service.state
        reject('already_deleted_request_rejected',lambda:service.delete([c.anchor_id]));check('rejection_is_atomic',service.state is old)
        check('empty_request_is_identity',service.delete([]) is old)
        reject('fresh_horizon_cannot_reset',lambda:codec.initialize_retained(c,{rid:[1]*c.D for rid in c.candidate_ids},deleted_ids=[c.anchor_id],remaining_horizon=c.budget))
        reject('fresh_initializer_requires_all_surviving_candidate_inputs',lambda:codec.initialize_retained(c,{}))
        reject('wrong_sign_type_rejected',lambda:codec.initialize_retained(c,{rid:[True]*c.D for rid in c.candidate_ids}))
        reject('wrong_sign_length_rejected',lambda:codec.initialize_retained(c,{rid:[1] for rid in c.candidate_ids}))
        for name,bad in [('mutable',bytearray(state)),('truncated',state[:-1]),('extra',state+b'\x00'),('version',b'!'+state[1:]),('contract',state[:8]+b'0'*32+state[40:]),('oversized_count',state[:40]+struct.pack('<Q',c.budget+1)+state[48:])]:
            reject(name+'_state_rejected',lambda bad=bad:codec.inspect_state(c,bad))
        # D=4,m=3 gives four high padding bits; setting one must be rejected.
        reject('noncanonical_padding_rejected',lambda:codec.inspect_state(c,state[:-1]+bytes([state[-1]|128])))
        two=codec.transition(c,state,c.common_ids);offset=codec.HEADER_BYTES
        swapped=two[:offset]+two[offset+1:offset+2]+two[offset:offset+1]+two[offset+2:]
        reject('unsorted_deleted_metadata_rejected',lambda:codec.inspect_state(c,swapped))
        duplicate=two[:offset+1]+two[offset:offset+1]+two[offset+2:]
        reject('duplicate_deleted_metadata_rejected',lambda:codec.inspect_state(c,duplicate))
        reject('different_lambda_contract_rejected',lambda:codec.inspect_state(make_contract(lam=F(2,3)),state))
        class StringSubclass(str):pass
        reject('public_ID_subclass_rejected_consistently',lambda:codec.FiniteSignContract(c.b,c.D,c.signature_signs,tuple(StringSubclass(x) for x in c.record_ids),c.lambda_reg,c.priority_seed))
        for name,arguments in [('zero_b',{'b':0}),('boolean_b',{'b':True}),('nonpower_D',{'D':2}),('overdimension',{'D':4**10}),('nonpositive_lambda',{'lambda_reg':F(0)}),('boolean_lambda',{'lambda_reg':True}),('nan_lambda',{'lambda_reg':float('nan')}),('FP32_lambda',{'lambda_reg':np.float32(.5)})]:
            values={'b':c.b,'D':c.D,'signature_signs':c.signature_signs,'record_ids':c.record_ids,'lambda_reg':c.lambda_reg,'priority_seed':c.priority_seed};values.update(arguments)
            reject(name+'_contract_rejected',lambda values=values:codec.FiniteSignContract(**values))
        reject('excess_signature_coherence_rejected',lambda:codec.FiniteSignContract(c.b,c.D,(c.signature_signs[0],)*c.m,c.record_ids))
        exact_float=make_contract(lam=float(.1));check('FP64_lambda_interpreted_exactly',exact_float.lambda_reg==F.from_float(.1) and exact_float.lambda_reg!=F(1,10))
        output=codec.materialized_head_accounting(c,activated);head=codec.head_exact(c,activated)
        encoded=json.dumps([[hex(v.numerator),hex(v.denominator)] for v in head],separators=(',',':')).encode('ascii')
        check('exact_output_representation_bytes_recompute',output['canonical_fraction_pair_JSON_bytes']==len(encoded))
        large=make_contract(lam=F(10**5000));large_state=codec.initialize_retained(large,{rid:[1]*large.D for rid in large.candidate_ids});large_active=codec.transition(large,large_state,large.common_ids+(large.private_blocker_ids[0],))
        check('arbitrary_rational_lambda_survives_public_and_output_serialization',any(codec.head_exact(large,large_active)) and codec.materialized_head_accounting(large,large_active)['canonical_fraction_pair_JSON_bytes']>0 and int(large.public_dict()['lambda_numerator_hex'],16)==10**5000)
        check('accounting_does_not_claim_process_or_total_byte_optimality',not codec.accounting(c,state)['total_byte_optimality_claim'] and not codec.accounting(c,state)['physical_erasure_claim'] and not output['Python_object_and_allocator_bytes_included'])
        if sources()!=bindings:raise AssertionError('Reviewed source changed during check')
        report.update(status='passed',checks=checks,passed=len(checks),case_counts=dict(counts))
    except Exception as error:report.update(status='failed',checks=checks,passed=len(checks),case_counts=dict(counts),error=repr(error))
    for name in ('finite_word_codec.py','check_codec_review.py'):
        (out/name).write_bytes(captured['phase10/'+name])
    (out/'checks.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'cases':dict(counts),'error':report.get('error')}));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
