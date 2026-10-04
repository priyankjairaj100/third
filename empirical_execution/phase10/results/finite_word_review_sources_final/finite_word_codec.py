"""Exact sequential codec for the explicitly public finite-sign hard family.

This is a mathematical family service, not a general corpus compressor.
Only immutable state bytes persist privately. Public parameters, deletion
metadata, padding, initialization inputs, workspace and returned heads are
separate costs; mD describes only the initial private sign payload.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from collections.abc import Mapping

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from phase3.reference_graph import stable_priority

MAGIC=b'CCUFW01\x00'
HEADER_BYTES=len(MAGIC)+32+8
CANDIDATE_NORM_SQUARED=Q(16641,16384)
THRESHOLD=float.fromhex('0x1.999999999999ap-2')


def _positive_int(value,name):
    if type(value) is not int or value<1: raise ValueError(name+' must be a positive integer')
    return value


def _power_four(value,name):
    _positive_int(value,name)
    if value&(value-1) or (value.bit_length()-1)%2:
        raise ValueError(name+' must be a power of four')
    return 1<<((value.bit_length()-1)//2)


def _lambda(value):
    if isinstance(value,(bool,np.bool_)): raise ValueError('Boolean lambda is invalid')
    if isinstance(value,Q): result=value
    elif type(value) is int: result=Q(value)
    elif type(value) in (float,np.float64):
        if not math.isfinite(value): raise ValueError('Finite stored FP64 lambda required')
        result=Q.from_float(float(value))
    else: raise TypeError('Lambda must be rational, integer, or an exact stored FP64 value')
    if result<=0: raise ValueError('Lambda must be positive')
    return result


def _signs(values,length):
    values=tuple(values)
    if len(values)!=length or any(type(v) is not int or v not in (-1,1) for v in values):
        raise ValueError('Exactly the declared number of integer +/-1 signs required')
    return values


@dataclass(frozen=True)
class FiniteSignContract:
    """Private-independent public parameters; contains no candidate sign block."""
    b: int
    D: int
    signature_signs: tuple
    record_ids: tuple
    lambda_reg: Q=Q(1)
    priority_seed: int=0

    def __post_init__(self):
        _positive_int(self.b,'b');_power_four(self.D,'D')
        raw=tuple(tuple(row) for row in self.signature_signs)
        if len(raw)<2: raise ValueError('The stated hard family requires at least two candidates')
        k=len(raw[0]);rootk=_power_four(k,'k')
        if 2+k+self.D>2**20: raise ValueError('Total dimension exceeds the proved frozen-scorer regime 2^20')
        signatures=tuple(_signs(row,k) for row in raw)
        for i,row in enumerate(signatures):
            for other in signatures[:i]:
                if 6*abs(sum(x*y for x,y in zip(row,other)))>k:
                    raise ValueError('Public signature coherence must be at most 1/6')
        ids=tuple(self.record_ids)
        if any(type(r) is not str for r in ids): raise ValueError('Public identifiers must be exact strings')
        if len(ids)!=1+self.b+2*len(signatures): raise ValueError('Public universe has wrong size')
        ordered=stable_priority(ids,self.priority_seed)
        # Check finite-word coordinates themselves, not merely their notation.
        for value in (Q(1,2),Q(7,8*rootk),Q(1,128*_power_four(self.D,'D'))):
            stored=np.float32(float(value))
            if not np.isfinite(stored) or Q.from_float(float(stored))!=value:
                raise ValueError('Construction coordinate is not exactly representable in FP32')
        object.__setattr__(self,'signature_signs',signatures)
        object.__setattr__(self,'record_ids',ordered)
        object.__setattr__(self,'lambda_reg',_lambda(self.lambda_reg))

    @property
    def m(self): return len(self.signature_signs)
    @property
    def k(self): return len(self.signature_signs[0])
    @property
    def dimension(self): return 2+self.k+self.D
    @property
    def budget(self): return self.b+1
    @property
    def anchor_id(self): return self.record_ids[0]
    @property
    def common_ids(self): return self.record_ids[1:1+self.b]
    @property
    def private_blocker_ids(self): return self.record_ids[1+self.b:1+self.b+self.m]
    @property
    def candidate_ids(self): return self.record_ids[1+self.b+self.m:]
    @property
    def index_bytes(self): return max(1,((len(self.record_ids)-1).bit_length()+7)//8)

    def public_dict(self):
        return {'schema':'ccu-finite-sign-public-contract-1','b':self.b,'D':self.D,
                'signature_signs':self.signature_signs,'record_ids':self.record_ids,
                'lambda_numerator_hex':hex(self.lambda_reg.numerator),'lambda_denominator_hex':hex(self.lambda_reg.denominator),
                'priority_seed':self.priority_seed,'threshold_hex':THRESHOLD.hex(),
                'roles':'priority-sorted anchor, commons, private blockers, candidates',
                'stored_features':'identical FP32 curator and learner values; learner is not normalized'}

    def public_bytes(self):
        return json.dumps(self.public_dict(),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')

    @property
    def digest(self): return hashlib.sha256(self.public_bytes()).digest()


def blocker_ids(contract,index):
    if type(index) is not int or not 0<=index<contract.m: raise ValueError('Candidate index outside contract')
    return frozenset(contract.common_ids+(contract.private_blocker_ids[index],))


def _deleted(contract,identifiers):
    if isinstance(identifiers,(str,bytes)): raise TypeError('Requests contain complete identifiers, not a text string')
    values=tuple(identifiers)
    if any(type(v) is not str for v in values) or len(set(values))!=len(values):
        raise ValueError('Unique textual deletion identifiers required')
    known=set(contract.record_ids)
    if not set(values)<=known: raise ValueError('Unknown deletion identifier')
    if len(values)>contract.budget: raise ValueError('Cumulative deletion budget exceeded')
    selected=set(values)
    return tuple(i for i,r in enumerate(contract.record_ids) if r in selected)


def eligible_indices(contract,deleted_indices):
    """Apply the general future-eligibility rule, including survival."""
    positions=tuple(deleted_indices)
    if positions!=tuple(sorted(set(positions))) or any(type(i) is not int or not 0<=i<len(contract.record_ids) for i in positions):
        raise ValueError('Canonical deletion indices required')
    if len(positions)>contract.budget: raise ValueError('Cumulative deletion budget exceeded')
    forgotten={contract.record_ids[i] for i in positions};remaining=contract.budget-len(positions)
    return tuple(i for i,r in enumerate(contract.candidate_ids)
                 if r not in forgotten and len(blocker_ids(contract,i)-forgotten)<=remaining)


@dataclass(frozen=True)
class StateView:
    deleted_indices: tuple
    eligible_indices: tuple
    payload: memoryview
    payload_bits: int
    remaining_horizon: int


def inspect_state(contract,state):
    """Validate canonical bytes before exposing a transient readonly view."""
    if type(state) is not bytes: raise TypeError('Authoritative state must be immutable bytes')
    if len(state)<HEADER_BYTES or state[:len(MAGIC)]!=MAGIC or state[len(MAGIC):len(MAGIC)+32]!=contract.digest:
        raise ValueError('Wrong state version, truncated header, or different public contract')
    count=struct.unpack_from('<Q',state,len(MAGIC)+32)[0]
    if count>contract.budget: raise ValueError('State deletion count exceeds initial budget')
    end=HEADER_BYTES+count*contract.index_bytes
    if end>len(state): raise ValueError('Truncated deletion metadata')
    positions=tuple(int.from_bytes(state[HEADER_BYTES+j*contract.index_bytes:HEADER_BYTES+(j+1)*contract.index_bytes],'little') for j in range(count))
    eligible=eligible_indices(contract,positions);bits=len(eligible)*contract.D
    if len(state)!=end+(bits+7)//8: raise ValueError('Missing or extra private payload bytes')
    payload=memoryview(state)[end:]
    if bits%8 and int(payload[-1])>>(bits%8): raise ValueError('Nonzero canonical padding bits')
    return StateView(positions,eligible,payload,bits,contract.budget-count)


def _serialize(contract,positions,payload):
    metadata=b''.join(i.to_bytes(contract.index_bytes,'little') for i in positions)
    result=MAGIC+contract.digest+struct.pack('<Q',len(positions))+metadata+bytes(payload)
    inspect_state(contract,result)
    return result


def initialize_retained(contract,retained_sign_blocks,*,deleted_ids=(),remaining_horizon=None):
    """Fresh retained initializer; input contains every surviving candidate only.

    Initialization may read retained private input. The returned state stores
    only future-eligible sign blocks and retains no reference to this mapping.
    This initializer is never called by the sequential transition.
    """
    positions=_deleted(contract,deleted_ids);h=contract.budget-len(positions)
    if remaining_horizon is not None and (type(remaining_horizon) is not int or remaining_horizon!=h):
        raise ValueError('Fresh initialization must use the actual remaining horizon')
    if not isinstance(retained_sign_blocks,Mapping): raise TypeError('Retained sign blocks must be a mapping')
    forgotten={contract.record_ids[i] for i in positions}
    surviving={r for r in contract.candidate_ids if r not in forgotten}
    if set(retained_sign_blocks)!=surviving: raise ValueError('Fresh input must contain exactly surviving candidate blocks')
    eligible=eligible_indices(contract,positions);places={i:j for j,i in enumerate(eligible)}
    payload=bytearray((len(eligible)*contract.D+7)//8)
    for i,rid in enumerate(contract.candidate_ids):
        if rid not in surviving: continue
        signs=_signs(retained_sign_blocks[rid],contract.D)
        if i not in places: continue
        for j,sign in enumerate(signs):
            bit=places[i]*contract.D+j
            if sign==1: payload[bit//8]|=1<<(bit%8)
    return _serialize(contract,positions,payload)


def transition(contract,state,request_ids):
    """IDs-only update. Original state bytes remain unchanged on every refusal."""
    old=inspect_state(contract,state);requested=_deleted(contract,request_ids)
    if set(requested)&set(old.deleted_indices): raise ValueError('A requested identifier was already deleted')
    positions=tuple(sorted(old.deleted_indices+requested))
    if len(positions)>contract.budget: raise ValueError('Cumulative deletion budget exceeded')
    if not requested: return state
    eligible=eligible_indices(contract,positions)
    old_places={i:j for j,i in enumerate(old.eligible_indices)}
    if not set(eligible)<=set(old_places): raise AssertionError('Future eligibility must be monotone under the remaining horizon')
    payload=bytearray((len(eligible)*contract.D+7)//8)
    for place,index in enumerate(eligible):
        before=old_places[index]*contract.D;after=place*contract.D
        for j in range(contract.D):
            bit=(int(old.payload[(before+j)//8])>>((before+j)%8))&1
            if bit: payload[(after+j)//8]|=1<<((after+j)%8)
    return _serialize(contract,positions,payload)


def _candidate_exact(contract,index,signs):
    signs=_signs(signs,contract.D);rootk=_power_four(contract.k,'k');rootd=_power_four(contract.D,'D')
    return (Q(0),Q(1,2))+tuple(Q(7*v,8*rootk) for v in contract.signature_signs[index])+tuple(Q(v,128*rootd) for v in signs)


def head_exact(contract,state):
    """Return a transient exact rational head; generally not a floating array."""
    view=inspect_state(contract,state);forgotten={contract.record_ids[i] for i in view.deleted_indices}
    active=[i for i in view.eligible_indices if blocker_ids(contract,i)<=forgotten]
    if not active: return tuple(Q(0) for _ in range(contract.dimension))
    if len(active)!=1 or view.remaining_horizon!=0: raise AssertionError('Hard-family activation invariant failed')
    index=active[0];offset=view.eligible_indices.index(index)*contract.D
    signs=tuple(1 if (int(view.payload[(offset+j)//8])>>((offset+j)%8))&1 else -1 for j in range(contract.D))
    scale=CANDIDATE_NORM_SQUARED+2*contract.lambda_reg
    return tuple(v/scale for v in _candidate_exact(contract,index,signs))


class CanonicalService:
    """No persistent decoded blocks, original cache, seed, or materialized head."""
    __slots__=('contract','_state')

    def __init__(self,contract,state):
        inspect_state(contract,state);self.contract=contract;self._state=state

    @property
    def state(self): return self._state

    def delete(self,request_ids):
        updated=transition(self.contract,self._state,request_ids)
        self._state=updated
        return updated

    def head(self): return head_exact(self.contract,self._state)


def materialize_family(contract,sign_blocks):
    """Mathematical input/oracle helper; never used by transition or head decode.

    This intentionally materializes a full original private cache. It is an
    initialization/oracle cost, not extra free service storage.
    """
    if set(sign_blocks)!=set(contract.candidate_ids): raise ValueError('One sign block per candidate required')
    d=contract.dimension;zero=Q(0);rows=[]
    rows.append((Q(1),)+(zero,)*(d-1))
    common=(Q(1,2),Q(7,8))+(zero,)*(d-2)
    rows.extend([common]*contract.b);rootk=_power_four(contract.k,'k')
    for signs in contract.signature_signs:
        rows.append((Q(1,2),zero)+tuple(Q(7*v,8*rootk) for v in signs)+(zero,)*contract.D)
    for i,rid in enumerate(contract.candidate_ids): rows.append(_candidate_exact(contract,i,sign_blocks[rid]))
    x=np.asarray([[float(v) for v in row] for row in rows],dtype=np.float32)
    if any(Q.from_float(float(x[i,j]))!=rows[i][j] for i in range(len(rows)) for j in range(d)):
        raise ValueError('Materialization changed an exact FP32 coordinate')
    y=np.asarray([0]*(1+contract.b+contract.m)+[1]*contract.m,dtype=np.float64)
    return x,y


def accounting(contract,state):
    view=inspect_state(contract,state);n=len(contract.record_ids);d=contract.dimension
    edges=(contract.b+contract.m)+contract.b*(contract.b-1)//2+contract.m*(contract.b+1)
    return {'serialized_state_bytes':len(state),'private_sign_payload_bits':view.payload_bits,
            'private_sign_payload_bytes':len(view.payload),'payload_padding_bits':8*len(view.payload)-view.payload_bits,
            'version_contract_binding_and_count_bytes':HEADER_BYTES,'deleted_index_metadata_bytes':len(view.deleted_indices)*contract.index_bytes,
            'deleted_records':len(view.deleted_indices),'remaining_horizon':view.remaining_horizon,
            'eligible_candidates':len(view.eligible_indices),'public_contract_serialized_bytes':len(contract.public_bytes()),
            'public_full_graph_CSR_int64_bytes_if_materialized':8*((n+1)+edges+n),
            'public_ID_UTF8_bytes':sum(len(r.encode('utf-8')) for r in contract.record_ids),
            'public_fixed_FP32_rows_bytes_if_materialized':(1+contract.b+contract.m)*d*4,
            'original_full_FP32_feature_cache_bytes_if_materialized':n*d*4,
            'original_binary_targets_FP64_bytes_if_materialized':n*8,
            'returned_exact_head_coordinates':d,'exact_head_numerator_denominator_bytes':'value dependent; Python object and integer overhead excluded from serialized-state counts',
            'transition_workspace':'old immutable state plus new bytearray and bytes, deletion/eligibility indices, public hashing and transient sign/bit operations; not a peak-RSS bound',
            'private_original_inputs_retained_by_service':False,'head_retained_by_service':False,
            'total_byte_optimality_claim':False,'physical_erasure_claim':False}


def materialized_head_accounting(contract,state):
    """Explicitly materialize and charge one exact output representation.

    Its JSON bytes are an accounting format, not a minimum-size head encoding.
    This function itself allocates the head and serialization workspace.
    """
    head=head_exact(contract,state)
    encoded=json.dumps([[hex(v.numerator),hex(v.denominator)] for v in head],separators=(',',':')).encode('ascii')
    return {'exact_head_coordinates':len(head),'canonical_fraction_pair_JSON_bytes':len(encoded),
            'fraction_integer_encoding':'canonical signed hexadecimal numerator and positive hexadecimal denominator',
            'numerator_magnitude_bits':sum(abs(v.numerator).bit_length() for v in head),
            'denominator_magnitude_bits':sum(v.denominator.bit_length() for v in head),
            'Python_object_and_allocator_bytes_included':False,'output_retained_by_service':False}
