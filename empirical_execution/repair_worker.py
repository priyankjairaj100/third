#!/usr/bin/env python3
"""Fresh-process summary replay with Python audit-hook I/O guard.

This is a software instrumentation check, not an OS security boundary, side-
channel guarantee, proof of deletion from allocator memory, or privacy proof.
The worker receives only an aggregate state and an IDs-only request file.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
from ccu.summary import IndexedRidgeSummary
from ccu.core import solve_ridge

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',required=True)
    p.add_argument('--requests',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    state=IndexedRidgeSummary.load(a.state)
    requests=json.loads(Path(a.requests).read_text())
    audit={'active':False,'attempts':[]}
    def hook(event,args):
        if audit['active'] and (event=='open' or event.startswith('socket.') or event in {'subprocess.Popen','os.system'}):
            audit['attempts'].append({'event':event})
            raise RuntimeError('I/O attempted during guarded repair')
    sys.addaudithook(hook)
    outputs=[]
    for batch in requests['batches']:
        audit['active']=True
        try:
            meter=state.delete(batch)
            sol=state.decode(requests['lambda'])
            gram,cross,n=state.statistics()
            outputs.append({'count':n,'remaining_horizon':state.horizon,
              'weights':sol.weights.tolist(),'weight_sha256':hashlib.sha256(sol.weights.tobytes()).hexdigest(),
              'normal_equation_residual_fro':sol.normal_equation_residual_fro,'work':meter})
        finally:audit['active']=False
    Path(a.output).write_text(json.dumps({'status':'completed','scope':__doc__,
       'io_attempts_during_repair':audit['attempts'],'outputs':outputs},indent=2)+'\n')

if __name__=='__main__':main()
