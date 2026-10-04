#!/usr/bin/env python3
"""Targeted accidental-disclosure scan of publishable Phase 5 artifacts.

No matched secrets, News bodies, or fingerprint bytes are printed or reported.
Archive members are scanned in memory without extracting archive paths.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
SCOPE=ROOT/'empirical_execution/phase5'
OUTPUT=SCOPE/'results/publication_content_audit.json'
PATTERNS={
 'private_key_header':rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
 'aws_access_key':rb'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b',
 'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})\b',
 'openai_style_secret':rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b',
 'google_api_key':rb'\bAIza[0-9A-Za-z_-]{35}\b',
 'jwt_like_token':rb'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b',
 'url_embedded_credentials':rb'https?://[^/\s:@]+:[^/\s@]+@',
 'signed_url_parameter':rb'(?:X-Amz-(?:Credential|Signature|Security-Token)|X-Goog-Signature|[?&]sig)=[^&\s\"\'<>]{16,}',
 'quoted_secret_assignment':rb'(?:api[_-]?key|secret|password|access[_-]?token|auth[_-]?token)\s*[:=]\s*[\"\']([^\"\'\r\n]{8,})[\"\']',
}
COMPILED={name:re.compile(pattern,re.IGNORECASE if name in {'url_embedded_credentials','signed_url_parameter','quoted_secret_assignment'} else 0) for name,pattern in PATTERNS.items()}


def candidates():
    proc=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','--','empirical_execution/phase5'],cwd=ROOT,capture_output=True,text=True,check=True)
    return sorted({ROOT/p for p in proc.stdout.splitlines() if (ROOT/p).is_file() and ROOT/p!=OUTPUT})


def fingerprints():
    path=ROOT/'empirical_execution/data/cc_news_engineering_preview.jsonl'
    if not path.exists():raise FileNotFoundError('Excluded local News source is required for publication scan')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    values=set();short=0
    for row in rows:
        text=row.get('text')
        if not isinstance(text,str) or not text:raise ValueError('Expected excluded News text missing')
        encodings=[text.encode(),json.dumps(text,ensure_ascii=False)[1:-1].encode(),json.dumps(text,ensure_ascii=True)[1:-1].encode()]
        if len(encodings[0])<120:short+=1
        for raw in encodings:
            if len(raw)<120:continue
            for start in {0,(len(raw)-120)//2,len(raw)-120}:
                values.add(raw[start:start+120])
    return sorted(values),{'source_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'news_records':len(rows),'news_records_shorter_than_120_bytes':short,
      'fingerprint_bytes':120,'distinct_fingerprint_count':len(values),
      'fingerprint_locations':'first, middle and final 120 bytes for raw UTF-8 and JSON-escaped variants; deduplicated',
      'fingerprints_or_body_text_disclosed':False}


def main():
    fps,news=fingerprints();findings=[];counts=collections.Counter();inventory=[];archives=[]
    def scan(label,raw,depth=0):
        if depth>8:raise ValueError('unexpected nested archive depth')
        counts['byte_streams_checked']+=1;counts['expanded_bytes_checked']+=len(raw)
        for name,pattern in COMPILED.items():
            number=sum(1 for _ in pattern.finditer(raw))
            if number:findings.append({'path':label,'kind':name,'match_count':number,'matched_content_disclosed':False})
        matches=sum(f in raw for f in fps)
        if matches:findings.append({'path':label,'kind':'excluded_news_body_fingerprint','match_count':matches,'matched_content_disclosed':False})
        lower=label.lower()
        if raw.startswith(b'PK\x03\x04'):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                archives.append({'path':label,'format':'zip_or_npz','members':len(archive.infolist())})
                for item in archive.infolist():
                    if not item.is_dir():counts['archive_members_checked']+=1;scan(label+'!'+item.filename,archive.read(item),depth+1)
        elif raw.startswith(b'\x1f\x8b'):
            unpacked=gzip.decompress(raw)
            if lower.endswith(('.tar.gz','.tgz')):
                with tarfile.open(fileobj=io.BytesIO(unpacked),mode='r:') as archive:
                    members=archive.getmembers();archives.append({'path':label,'format':'tar.gz','members':len(members)})
                    for item in members:
                        if item.isfile():
                            handle=archive.extractfile(item)
                            if handle is None:raise ValueError('unreadable tar member')
                            counts['archive_members_checked']+=1;scan(label+'!'+item.name,handle.read(),depth+1)
                        elif not item.isdir():raise ValueError('unexpected nonregular archive member')
            else:
                archives.append({'path':label,'format':'gzip','members':1});counts['archive_members_checked']+=1
                scan(label+'!gunzip',unpacked,depth+1)
    files=candidates()
    for path in files:
        raw=path.read_bytes();label=str(path.relative_to(ROOT))
        inventory.append({'path':label,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        scan(label,raw)
    report={'schema':'ccu-publication-content-audit-2','status':'passed' if not findings else 'needs_review',
      'scope':'All Git-publishable Phase5 files and recursively decompressed gzip/tar.gz/zip/NPZ members; targeted accidental-disclosure scan, not a guarantee',
      'files_checked':len(files),**dict(counts),'news_fingerprint_check':news,'pattern_names':list(PATTERNS),
      'findings':findings,'false_positives':[],
      'approved_content':'Existing Civil100 text/original labels and official SemDeDup source with included license are permitted by current project release policy',
      'excluded_file_policy':'Git-ignored News body files and archived per-job source directories are not separately candidate public files; actual jobs.tar.gz members are scanned',
      'raw_content_or_credentials_printed':False,'input_authenticity_or_scientific_readiness_claim':False,
      'inventory_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
      'files':inventory,'archives':archives,'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'refresh_required_if_any_candidate_file_changes':True}
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','files_checked','byte_streams_checked','archive_members_checked','expanded_bytes_checked','news_fingerprint_check','findings']},indent=2))
    if findings:raise SystemExit(2)

if __name__=='__main__':main()
