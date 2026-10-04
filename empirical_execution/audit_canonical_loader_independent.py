"""Regression of independently discovered bool/int canonical-encoding alias.

Run with python3 -O as well: validation must not rely on Python assertions.
Algebraic fixtures here are software checks, not synthetic empirical datasets.
"""
import copy
import json
from pathlib import Path
from ccu.exact_canonical import ExactCanonicalSummary


def run():
    state = ExactCanonicalSummary.build([[1,0],[0,1]],[[1],[1]],[[],[]],2,backend='python')
    source = json.loads(state.canonical_bytes())
    variants = []
    x = copy.deepcopy(source); x['alive']=[0,True]; variants.append(x)
    x = copy.deepcopy(source); x['horizon']=True; variants.append(x)
    x = copy.deepcopy(source); x['coefficients'][1][0]=[False]; variants.append(x)
    x = copy.deepcopy(source); x['coefficients'][0][1]['pivots'][0]=False; variants.append(x)
    x = copy.deepcopy(source); x['coefficients'][0][1]['count']=True; variants.append(x)
    x = copy.deepcopy(source); x['coefficients'][0][1]['nonpivot'].append([]); variants.append(x)
    rejected = 0
    for x in variants:
        data = (json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode()
        try:
            ExactCanonicalSummary.from_bytes(data)
        except ValueError:
            rejected += 1
        else:
            raise RuntimeError('Invalid state accepted')
    if ExactCanonicalSummary.from_bytes(state.canonical_bytes()).canonical_bytes()!=state.canonical_bytes():
        raise RuntimeError('Valid roundtrip mismatch')
    return {'invalid_state_rejections':rejected,'valid_roundtrip':True,
            'python_assertions_enabled':__debug__,'failed_checks':0}


if __name__=='__main__':
    result = run()
    folder = Path(__file__).parent/'canonical_results'
    folder.mkdir(exist_ok=True)
    (folder/'independent_loader_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
