#!/usr/bin/env python3
"""Independent arbitrary-polynomial audit of the indexed repair implementation.
The inputs need not arise from a curation graph. This exercises the generic
specialization/truncation engine against a direct cumulative oracle.
"""
from itertools import combinations
import json
from pathlib import Path
import random
from algorithm_round2 import IndexedState, assert_bounds


def direct(original, deleted, horizon, dimension):
    out = {}
    for key, value in original.items():
        reduced = tuple(u for u in key if u not in deleted)
        if len(reduced) > horizon:
            continue
        previous = out.get(reduced, (0,) * dimension)
        out[reduced] = tuple(a+b for a, b in zip(previous, value))
    return {key: value for key, value in out.items() if any(value)}


def main():
    rng = random.Random(20261004)
    trials = 2000
    intermediate = 0
    zero_maps = 0
    for _ in range(trials):
        n = rng.randrange(0, 11)
        h = rng.randrange(n+1)
        d = rng.randrange(1, 8)
        keys = [key for size in range(h+1) for key in combinations(range(n), size)]
        original = {}
        for key in keys:
            if rng.random() < .32:
                value = tuple(rng.randrange(-2, 3) for _ in range(d))
                if any(value):
                    original[key] = value
        state = IndexedState(n, h, d, range(n), original)
        meter = state.meter()
        deleted = set()
        remaining = h
        order = rng.sample(range(n), h)
        while order:
            batch_size = rng.randrange(1, min(len(order), 4)+1)
            batch, order = order[:batch_size], order[batch_size:]
            if batch and rng.random() < .5:
                batch = batch + rng.choices(batch, k=3)
            deleted.update(batch)
            state.delete(batch, meter)
            remaining = h-len(deleted)
            expected = direct(original, deleted, remaining, d)
            assert state.coefficients() == expected
            state.check_invariants()
            assert_bounds(state, meter)
            intermediate += 1
            zero_maps += not expected
        assert state.coefficients() == direct(original, deleted, remaining, d)
    result = {'seed': 20261004, 'arbitrary_polynomial_trials': trials,
              'intermediate_batch_comparisons': intermediate,
              'empty_result_maps': zero_maps, 'status': 'passed'}
    Path('round3_audit_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
