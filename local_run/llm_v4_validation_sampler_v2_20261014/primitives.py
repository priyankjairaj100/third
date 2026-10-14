"""Uniform fresh validation sampling; no labels, files, entropy, or model access."""
from fractions import Fraction
import hashlib
import math
import random

def require(ok,why):
    if not ok:raise ValueError(why)

def generator(protocol,plan_sha,frame_sha,seed,stage):
    require(stage in ('validation','validation_swap'),'validation_seed_role')
    require(len(seed)==64 and all(c in '0123456789abcdef' for c in seed),'seed256')
    return random.Random(int(protocol.domain_digest(protocol.PREFIX+'sampling',
        [plan_sha,frame_sha,stage,seed,None]),16))

def collect(stream,rows,tau,rng,capacity=200,progress=None):
    require(math.isfinite(tau) and type(capacity) is int and capacity>0,'finite_cut_positive_capacity')
    reservoir=[];population=total=0;digest=hashlib.sha256()
    for i,j,score in stream:
        require(0<=i<j<rows and math.isfinite(score),'finite_ordered_pair')
        digest.update(f'{i}\t{j}\t{score.hex()}\n'.encode('ascii'));total+=1
        if score>tau:
            population+=1;item=(i,j,score)
            if population<=capacity:reservoir.append(item)
            else:
                index=rng.randrange(population)
                if index<capacity:reservoir[index]=item
        if progress is not None and total%10000000==0:progress(total)
    require(total==rows*(rows-1)//2,'complete_pair_population')
    require(len(reservoir)==min(capacity,population) and
        len({(x[0],x[1]) for x in reservoir})==len(reservoir),'complete_unique_reservoir')
    return {'population':population,'pair_count':total,'stream_sha256':digest.hexdigest(),
            'raw':sorted(reservoir,key=lambda x:(x[0],x[1]))}

def fraction(value):return {'numerator':value.numerator,'denominator':value.denominator}

def decorate(raw,population,rows,protocol,plan_sha,frame_sha,models):
    require(population>0 and len(raw)==min(200,population),'nonempty_complete_validation')
    n=len(raw);result=[]
    for i,j,score in raw:
        ids=[rows[i]['record_id'],rows[j]['record_id']]
        identity=protocol.identities(plan_sha,frame_sha,ids)
        canonical=[i,j] if identity['canonical_record_ids']==ids else [j,i]
        result.append({'frame_row_indices':[i,j],'original_full_row_indices':[
            rows[i]['original_full_row_index'],rows[j]['original_full_row_index']],
            'record_ids':ids,'source_unit_ids':[rows[i]['source_unit_id'],rows[j]['source_unit_id']],
            'same_source':rows[i]['source_unit_id']==rows[j]['source_unit_id'],**identity,
            'canonical_frame_row_indices':canonical,'score_hex':score.hex(),
            'validation_population':population,'validation_sample_size':n,
            'inclusion_probability':fraction(Fraction(n,population)),
            'inclusion_weight':fraction(Fraction(population,n)),
            'model_seeds':{m['model_id']:protocol.model_seed(plan_sha,frame_sha,
                identity['opaque_pair_id'],m['model_id'],m['revision']) for m in models}})
    require(len({r['opaque_pair_id'] for r in result})==n,'distinct_validation_identities')
    return result
