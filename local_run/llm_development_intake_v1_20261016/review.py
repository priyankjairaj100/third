"""Separate implementation of the whole-source reservation arithmetic review."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(salt, unit):
    return int(hashlib.sha256((salt+'\0'+unit).encode()).hexdigest(),16)


def check(folder, plan):
    result=json.loads((folder/'result.json').read_bytes())
    for name, expected in result['outputs'].items():
        assert Path(name).name==name
        raw=(folder/name).read_bytes()
        assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==expected
    rows=json.loads((folder/'allocation.private.json').read_bytes())
    prefixes=json.loads((folder/'development_prefixes.private.json').read_bytes())
    assert len(rows)==result['rows']
    ids=set(); units={}; byrole=defaultdict(list)
    for i,row in enumerate(rows):
        assert row['original_full_row_index']==i and row['record_id'] not in ids
        ids.add(row['record_id']); unit=row['source_unit_id']
        d=digest('ccu-v1-source-split',unit)
        original='train' if d*10<7*2**256 else ('calibration' if d*20<17*2**256 else 'test')
        assert original==row['partition']
        reserved=digest(plan['development_reserve_salt'],unit)<(2**256+9)//10
        role='calibration' if original=='calibration' else ('development_' if reserved else 'evaluation_')+original
        assert role==row['role'] and (unit not in units or units[unit]==role)
        units[unit]=role; byrole[role].append(row)
        assert row['genuine_source_request_eligible']==row['source_kind'].startswith('native_')
        assert not {'text','Tags','labels','original_fields'}.intersection(row)
    for role in ('development_train','evaluation_train','calibration','development_test','evaluation_test'):
        byrole[role]
    counts={role:{'rows':len(items),'source_units':len({r['source_unit_id'] for r in items}),
                   'unknown_singleton_rows':sum(r['source_kind']=='unknown_singleton' for r in items)}
            for role,items in byrole.items()}
    assert counts==result['allocation']
    groups=defaultdict(list)
    for row in byrole['development_train']:
        groups[row['source_unit_id']].append(row['record_id'])
    ordered=sorted(groups,key=lambda s:(digest(plan['development_order_salt'],s),s))
    for target in plan['development_targets']:
        expected=[]; chosen=[]
        for unit in ordered:
            if len(expected)>=target:break
            chosen.append(unit);expected.extend(sorted(groups[unit]))
        actual=prefixes[str(target)]
        assert actual=={'target':target,'actual':len(expected),'shortfall':max(0,target-len(expected)),
                       'overshoot':max(0,len(expected)-target),'record_ids':expected,'source_ids':chosen}
    assert all(result[k] is False for k in ('development_threshold_accepted','semantic_guard_executed',
        'post_guard_panels_frozen','effect_experiments_started','human_validity_claimed',
        'original_primary_input_acceptance','confirmatory_study_ready'))
    return {'schema':'third-development-reservation-arithmetic-review-v1','passed':True,
            'rows':len(rows),'source_units':len(units),'allocation':counts,
            'whole_source_prefixes_checked':len(prefixes),'original_partitions_recomputed':True,
            'reserve_roles_independently_recomputed':True,'output_payload_hashes_checked':True,
            'source_native_metadata_replay_repeated':False,'full_cache_rehash_repeated':False,
            'scientific_effects_verified':False,'reviewer_kind':'separate deterministic implementation',
            'intake_result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
            'review_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__=='__main__':
    folder=Path(sys.argv[1]); plan=json.loads((Path(__file__).parent/'PLAN.json').read_bytes())
    report=check(folder,plan)
    with (folder/'independent_review.json').open('x') as out:
        json.dump(report,out,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
    print(json.dumps({'independent_reservation_review_passed':True,'rows':report['rows']}),flush=True)
