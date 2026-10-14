"""V4 reservoir selection primitives; no filesystem, entropy or model side effects."""
from bisect import bisect_left
from fractions import Fraction
import hashlib
import math
import random
import protocol as p

def stratum(score, cuts):
    p.require(math.isfinite(score), 'finite_score')
    return bisect_left(cuts, score)

def insert(reservoir, value, seen, capacity, rng):
    p.require(seen > 0 and capacity > 0, 'positive_reservoir_parameters')
    if seen <= capacity:
        reservoir.append(value)
    else:
        position = rng.randrange(seen)
        if position < capacity:
            reservoir[position] = value

def generators(plan_sha, frame_sha, seed, strata):
    p.require(len(seed) == 64 and all(c in '0123456789abcdef' for c in seed), 'seed256')
    return [random.Random(int(p.domain_digest(p.PREFIX+'sampling',
        [plan_sha,frame_sha,'selection',seed,k]),16)) for k in range(strata)]

def select(stream, rows, cuts, capacity, rngs, progress=None):
    p.require(cuts == sorted(set(cuts)) and len(rngs) == len(cuts)+1, 'strata_contract')
    populations = [0]*len(rngs)
    reservoirs = [[] for _ in rngs]
    h = hashlib.sha256()
    total = 0
    for i,j,score in stream:
        p.require(0 <= i < j < rows, 'pair_indices')
        k = stratum(score,cuts)
        populations[k] += 1
        insert(reservoirs[k], (i,j,score,k), populations[k], capacity, rngs[k])
        h.update(f'{i}\t{j}\t{score.hex()}\n'.encode('ascii'))
        total += 1
        if progress is not None and total % 10000000 == 0:
            progress(total)
    p.require(total == rows*(rows-1)//2, 'complete_pair_population')
    selected = sorted((v for r in reservoirs for v in r),key=lambda x:(x[0],x[1]))
    p.require(len({(x[0],x[1]) for x in selected}) == len(selected), 'duplicate_selected_pair')
    p.require([len(r) for r in reservoirs] == [min(capacity,n) for n in populations], 'sample_sizes')
    return {'populations':populations,'raw':selected,'stream_sha256':h.hexdigest(),'pair_count':total}

def weights(populations, capacity):
    return [None if n == 0 else Fraction(n,min(capacity,n)) for n in populations]
