"""Strict local natural-record ingestion and an explicitly lexical pilot encoder.

The lexical encoder supports implementation checks only. It must never be
reported as the preregistered E5 semantic representation or confirmation data.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,unicodedata,re
import numpy as np

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_natural_jsonl(path,require_labels=False,require_sources=False):
    rows=[];seen=set()
    with Path(path).open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            r=json.loads(line)
            for key in ('record_id','text','provenance'):
                if key not in r:raise ValueError(f'{path}:{line_no}: missing {key}')
            if not isinstance(r['record_id'],str) or not r['record_id'] or r['record_id'] in seen:
                raise ValueError(f'{path}:{line_no}: invalid/duplicate ID')
            seen.add(r['record_id'])
            if not isinstance(r['text'],str) or not r['text'].strip():raise ValueError('empty/nonstring text')
            if require_labels and (r.get('label') is None or not np.all(np.isfinite(r['label']))):
                raise ValueError('original finite labels required; inventing labels is forbidden')
            if require_sources and not r.get('source_id'):raise ValueError('original source IDs required')
            if not r['provenance']:raise ValueError('record provenance required')
            rows.append(r)
    if not rows:raise ValueError('empty natural-data panel')
    return rows

def normalize_text(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFC',text)).strip()

def lexical_engineering_features(rows,dimension):
    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer=HashingVectorizer(n_features=int(dimension),analyzer='word',
        ngram_range=(1,2),alternate_sign=False,norm='l2',lowercase=False,
        token_pattern=r'(?u)\b\w+\b',dtype=np.float32)
    matrix=vectorizer.transform([normalize_text(r['text']) for r in rows]).toarray()
    if np.any(np.linalg.norm(matrix,axis=1)==0):
        raise ValueError('zero lexical vector: log exclusion/choose amended pilot before proceeding')
    matrix=np.ascontiguousarray(matrix,dtype=np.float32)
    meta={'name':'sklearn.HashingVectorizer','n_features':int(dimension),
          'analyzer':'word','ngrams':[1,2],'alternate_sign':False,'lowercase':False,
          'norm':'l2','dtype':'float32','corpus_fitted_state':False,
          'E5_or_semantic_model':False,'purpose':'nonconfirmatory_software_pilot_only',
          'limitations':['hash collisions','lexical similarity not audited semantic duplication',
                        'preview sampling and connector rendering','no held-out NLP utility claim'],
          'array_sha256':hashlib.sha256(matrix.tobytes()).hexdigest()}
    return matrix,meta

def load_canonical_features(path,record_manifest_path,rows):
    """Ingest a locally available pinned representation without row-order guessing."""
    ids=json.loads(Path(record_manifest_path).read_text())
    if ids!=[r['record_id'] for r in rows]:raise ValueError('feature row-ID manifest differs from records')
    z=np.load(path,allow_pickle=False)
    if z.dtype!=np.float32 or z.ndim!=2 or len(z)!=len(rows) or not np.isfinite(z).all():
        raise ValueError('expected finite canonical FP32 [N,d] feature array')
    return z
