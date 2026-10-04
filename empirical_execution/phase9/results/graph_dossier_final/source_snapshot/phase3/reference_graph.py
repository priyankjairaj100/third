"""Exhaustive primary curator with a pair-local FP64 reference score.

Unlike BLAS batch multiplication, each scalar normalization/dot uses the same
coordinate order regardless of corpus size, tile shape, or deletion history.
This fixes a finite-precision target, not real-arithmetic cosine certification.
"""
import hashlib,math
import numpy as np
from ccu.core import BlockerGraph
from phase3.panels import graph_normalize,reference_cosines

def stable_priority(record_ids,seed=0):
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('nonnegative integer priority seed required')
    if any(not isinstance(r,str) or not r for r in record_ids) or len(set(record_ids))!=len(record_ids):
        raise ValueError('unique nonempty record IDs required')
    return tuple(sorted(record_ids,key=lambda r:(hashlib.sha256(f'priority-v1|{seed}|{r}'.encode()).digest(),r)))

def build_reference_graph(features,record_ids,threshold,source_ids=None,*,seed=0,block_size=256):
    x=np.asanyarray(features);ids=tuple(record_ids)
    if x.ndim!=2 or x.dtype!=np.dtype('float32') or x.shape[0]!=len(ids) or x.shape[1]<1:
        raise ValueError('aligned finite FP32 feature matrix required')
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not -1<=threshold<=1:
        raise ValueError('finite threshold in [-1,1] required')
    if isinstance(block_size,bool) or not isinstance(block_size,int) or block_size<1:
        raise ValueError('positive block size required')
    names=ids if source_ids is None else tuple(source_ids)
    if len(names)!=len(ids) or any(not isinstance(s,str) or not s for s in names):
        raise ValueError('one nonempty source ID per record required')
    ordered_ids=stable_priority(ids,seed);lookup={r:i for i,r in enumerate(ids)}
    order=np.asarray([lookup[r] for r in ordered_ids],dtype=np.int64)
    for lo in range(0,len(ids),block_size):graph_normalize(x[lo:lo+block_size])
    blockers=[[] for _ in ids]
    for lo in range(0,len(ids),block_size):
        late=order[lo:lo+block_size];left=graph_normalize(x[late])
        for eo in range(0,lo+1,block_size):
            early=order[eo:eo+block_size];right=left if eo==lo else graph_normalize(x[early])
            scores=reference_cosines(left,right)
            ii,jj=np.nonzero(scores>threshold)
            keep=eo+jj<lo+ii
            for a,b in zip(ii[keep],jj[keep]):blockers[int(late[a])].append(int(early[b]))
    ptr=np.zeros(len(ids)+1,dtype=np.int64)
    for i,b in enumerate(blockers):b.sort();ptr[i+1]=ptr[i]+len(b)
    index=np.asarray([i for b in blockers for i in b],dtype=np.int64)
    for a in [ptr,index,order]:a.flags.writeable=False
    return BlockerGraph(ids,names,order,ptr,index,float(threshold))

def id_edges(graph):
    return {(graph.record_ids[i],graph.record_ids[int(j)]) for i,b in enumerate(graph.blockers) for j in b}
