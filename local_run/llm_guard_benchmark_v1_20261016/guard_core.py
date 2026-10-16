"""All-grid guard preparation using unchanged ordered FP64 cosine operations."""
import math
import numpy as np


def scan(cache, train, heldout, heldout_ids, thresholds, panels, block_size=256):
    if cache.dtype != np.dtype('float32') or cache.ndim != 2:
        raise ValueError('native_FP32_cache')
    if type(block_size) is not int or block_size < 1:
        raise ValueError('positive_block')
    if not thresholds or thresholds != sorted(set(thresholds)) or any(not math.isfinite(t) or not -1<=t<=1 for t in thresholds):
        raise ValueError('finite_sorted_thresholds')
    if len(heldout) != len(heldout_ids) or len(set(heldout_ids)) != len(heldout_ids):
        raise ValueError('unique_heldout_identity')
    for indices in (train, heldout):
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0<=i<len(cache) for i in indices):
            raise ValueError('unique_valid_indices')
    if set(train) & set(heldout):
        raise ValueError('train_heldout_overlap')
    order = sorted(range(len(heldout)), key=lambda i: heldout_ids[i])
    right_indices = [heldout[i] for i in order]
    witnesses = np.full((len(train),len(thresholds)),-1,dtype=np.int64)
    witness_scores = np.zeros((len(train),len(thresholds)),dtype=np.float64)
    maxima = np.full(len(train),-np.inf,dtype=np.float64)
    pairs=0
    for start in range(0,len(train),block_size):
        end=min(start+block_size,len(train)); left=panels.graph_normalize(cache[train[start:end]])
        for other in range(0,len(right_indices),block_size):
            ri=right_indices[other:other+block_size]
            scores=panels.reference_cosines(left,panels.graph_normalize(cache[ri])); pairs+=scores.size
            maxima[start:end]=np.maximum(maxima[start:end],scores.max(axis=1))
            for k,tau in enumerate(thresholds):
                positive=scores>tau
                has=positive.any(axis=1)
                local=np.flatnonzero(has & (witnesses[start:end,k]<0))
                if len(local):
                    first=positive[local].argmax(axis=1)
                    witnesses[start+local,k]=np.asarray(ri)[first]
                    witness_scores[start+local,k]=scores[local,first]
    return {'witness_indices':witnesses,'witness_scores':witness_scores,'maxima':maxima,'pairs':pairs}
