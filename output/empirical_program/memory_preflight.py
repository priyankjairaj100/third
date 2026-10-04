#!/usr/bin/env python3
"""Analytic forecasts only; inputs must come from a measured natural graph."""
import argparse, json

def nonnegative(value):
    n=int(value)
    if n<0: raise argparse.ArgumentTypeError('must be nonnegative')
    return n

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimension',type=nonnegative,required=True)
    p.add_argument('--outputs',type=nonnegative,required=True)
    p.add_argument('--keys',type=nonnegative,required=True)
    p.add_argument('--eligible',type=nonnegative,required=True)
    p.add_argument('--rank',type=nonnegative)
    p.add_argument('--metadata-bytes',type=nonnegative)
    p.add_argument('--workspace-bytes',type=nonnegative)
    p.add_argument('--baseline-metadata-bytes',type=nonnegative)
    p.add_argument('--baseline-workspace-bytes',type=nonnegative)
    p.add_argument('--cap-gib',type=float,default=128)
    a=p.parse_args()
    if a.dimension<1 or a.outputs<1 or a.cap_gib<=0: p.error('dimensions, outputs and cap must be positive')
    d,c=a.dimension,a.outputs
    s=d*(d+1)//2+d*c+1
    coeff=8*s
    proposed=coeff*a.keys
    baseline_rows=4*(d+c)*a.eligible
    baseline_moments=8*s
    base_floor=baseline_rows+baseline_moments
    def total(floor,m,w):
        return None if m is None or w is None else floor+m+w
    pt=total(proposed,a.metadata_bytes,a.workspace_bytes)
    bt=total(base_floor,a.baseline_metadata_bytes,a.baseline_workspace_bytes)
    out={
        'kind':'analytic_forecast_not_observed_experiment',
        'coordinates_per_coefficient':s,'FP64_bytes_per_coefficient':coeff,
        'proposed_statistic_payload_bytes':proposed,
        'rank_coordinate_dense_payload_bytes':None if a.rank is None else a.rank*coeff,
        'eligible_FP32_feature_label_bytes':baseline_rows,
        'baseline_current_FP64_moment_bytes':baseline_moments,
        'proposed_payload_plus_supplied_metadata_workspace':pt,
        'baseline_payload_plus_supplied_metadata_workspace':bt,
        'common_cap_bytes':int(a.cap_gib*1024**3),
        'forecast_native_infeasible_from_payload_alone':proposed>a.cap_gib*1024**3,
        'forecast_over_cap_with_supplied_overheads':None if pt is None else pt>a.cap_gib*1024**3,
        'fast_map_M_le_2r':None if a.rank is None else a.keys<=2*a.rank,
        'note':'Missing metadata/workspace prevents total-memory feasibility conclusions. FP32 labels assumed for row byte calculation; adjust to actual source precision. Shared assets, allocator and serialization copies must be included once in supplied totals. No universal lower bound for structured ridge encodings is claimed.'
    }
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
