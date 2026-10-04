#!/usr/bin/env python3
"""Validate locally supplied source-rich Civil JSONL or CSV; never guess source IDs.

This does not fetch data, label missing examples, or turn previews into an
immutable original corpus. It preserves original IDs/labels/source metadata
and emits a checksum-pinned acquisition manifest for the next execution stage.
"""
from pathlib import Path
import argparse,csv,json,hashlib
from ccu.data import sha256_file,normalize_text

REQUIRED={'id','text','toxicity','article_id','publication_id','parent_id','created_date'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-url',required=True)
    p.add_argument('--source-version',required=True);p.add_argument('--source-license',required=True)
    p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();digest=sha256_file(a.input)
    if digest!=a.expected_sha256:raise ValueError('input checksum mismatch')
    if a.output.resolve()==a.input.resolve():raise ValueError('output cannot overwrite original input')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix(a.output.suffix+'.partial');seen=set();counts={'records':0,'missing_article_sources':0}
    def iterator():
        with a.input.open(encoding='utf-8',newline='') as f:
            if a.input.suffix.lower()=='.csv':yield from csv.DictReader(f)
            else:
                for line in f:
                    if line.strip():yield json.loads(line)
    try:
        with temporary.open('w',encoding='utf-8') as out:
            for row in iterator():
                if not REQUIRED<=set(row):raise ValueError('missing required original fields: '+str(sorted(REQUIRED-set(row))))
                ident=str(row['id'])
                if not ident or ident in seen:raise ValueError('empty or duplicate original id')
                seen.add(ident);text=normalize_text(row['text']);label=float(row['toxicity'])
                if not text or not 0<=label<=1:raise ValueError('empty text or invalid toxicity label')
                publication=str(row['publication_id']);article=str(row['article_id'])
                missing=publication in {'','None','nan','-1'} or article in {'','None','nan','-1'}
                source=None if missing else json.dumps([publication,article],separators=(',',':'))
                counts['missing_article_sources']+=missing
                r={'record_id':'civil:'+ident,'text':text,'label':label,'source_id':source,
                   'source_kind':'missing_singleton' if missing else 'publication_qualified_article',
                   'fields':{key:row[key] for key in REQUIRED if key!='text'},
                   'provenance':{'source_url':a.source_url,'source_version':a.source_version,
                      'source_license':a.source_license,'input_sha256':digest,'original_record_id':ident,
                      'original_metadata_fields_validated':True,'byte_original_input_preserved':True,
                      'normalized_text':'NFC_whitespace_only'}}
                out.write(json.dumps(r,ensure_ascii=False)+'\n');counts['records']+=1
        if not counts['records']:raise ValueError('empty dataset')
        temporary.replace(a.output)
    except Exception:
        temporary.unlink(missing_ok=True);raise
    manifest={**counts,'input_file':str(a.input),'input_sha256':digest,'output_file':str(a.output),
              'output_sha256':sha256_file(a.output),'source_url':a.source_url,'source_version':a.source_version,
              'source_license':a.source_license,'status':'source_metadata_ingested_not_yet_split_or_calibrated'}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
