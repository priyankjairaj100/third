"""Software schema/calendar tests only; no WCEP or News empirical corpus invented."""
from pathlib import Path
import gzip
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from empirical_execution.phase5 import replication as r


def main():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def refused(name, call):
        try: call()
        except (ValueError, FileExistsError, __import__('sqlite3').IntegrityError): checks.append(name)
        else: raise AssertionError(name)
    (ROOT / 'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='phase5_replication_software_', dir=ROOT/'tmp') as temp:
        p = Path(temp)
        paths = []
        for n, collection in enumerate(('train','val','test')):
            event={'id':n,'date':'2018-01-01','summary':'SUMMARY MUST NEVER ENTER ENCODER',
                   'category':'CATEGORY IS NOT DUPLICATE GOLD','collection':collection,
                   'articles':[{'title':'Title','text':'Title\nAuthor text with negation not removed.',
                                'url':'https://example.invalid/repeated','origin':'CommonCrawl'}]*2}
            path=p/(collection+'.jsonl.gz');path.write_bytes(gzip.compress((json.dumps(event)+'\n').encode(),mtime=0));paths.append(path)
        audit=r.parse_wcep(paths,p/'parsed',snapshot_revision='software_schema_fixture_only',collection_scope='all_official_splits')
        rows=[x for _,x in r._events(p/'parsed/records.jsonl')]
        check('all six distinct repeated-string article instances preserved',len(rows)==6 and len({x['record_id'] for x in rows})==6)
        check('no summary or category in encoder input',all('SUMMARY' not in x['text'] and 'CATEGORY' not in x['text'] for x in rows))
        check('only exact repeated first title removed',all(x['text']=='Title Author text with negation not removed.' for x in rows))
        check('URL presence does not invent genuine domain/source',all(x['source_kind']=='unknown_singleton' for x in rows))
        check('no authenticity or readiness claim',not audit['confirmatory_study_ready'] and not audit['input_authenticity_verified'])
        panel=r.event_panel(p/'parsed',p/'panel',target=3)
        check('whole boundary event included and overshoot reported',panel['selected_articles']==4 and panel['overshoot']==1)
        smaller=r.event_panel(p/'parsed',p/'smaller',target=1)
        check('nested whole event panels',set(smaller['selected_events'])<=set(panel['selected_events']))
        allpanel=r.event_panel(p/'parsed',p/'allpanel',target=100)
        check('shortfall preserved',allpanel['shortfall']==94)
        check('frozen News threshold only',panel['threshold_policy']=='reuse_frozen_same_encoder_CC_News_threshold_no_WCEP_retuning')
        refused('overwrite refused',lambda:r.event_panel(p/'parsed',p/'panel',target=3))
        refused('boolean target refused',lambda:r.event_panel(p/'parsed',p/'bad',target=True))
        duplicate=p/'duplicate.jsonl';duplicate.write_text(json.dumps({'id':0,'date':'2018-01-01','collection':'train','articles':[]})+'\n')
        refused('duplicate native event IDs refused',lambda:r.parse_wcep([paths[0],duplicate],p/'dup',snapshot_revision='software',collection_scope='declared_subset'))
        check('failed parse leaves explicit incomplete marker',(p/'dup/FAILED.json').exists())
        altered=p/'parsed/records.jsonl';altered.write_text(altered.read_text()+'\n')
        refused('changed article bytes refused',lambda:r.event_panel(p/'parsed',p/'badbytes'))
        source=[{'record_id':f'software_date_{i}','partition':'train','original_fields':{'date':day}} for i,day in enumerate(['2018-01-02','2018-01-01','2018-01-01','2018-02-01'])]
        chrono=r.news_chronological_requests(source,horizon=3)
        check('whole date group ordering',chrono['checkpoints'][0]['batch_record_ids']==['software_date_1','software_date_2'])
        check('cumulative date prefix',chrono['checkpoints'][-1]['cumulative_records']==3)
        check('next group over budget disclosed',chrono['blocked_next_group']['records']==1)
        check('no pseudo random replicates',chrono['independent_random_replicates']==0)
        monthly=r.news_chronological_requests(source,horizon=2,resolution='month')
        check('overbudget first month not cut',not monthly['checkpoints'] and monthly['blocked_next_group']['records']==3)
        refused('heldout/calibration cannot receive date withdrawals',lambda:r.news_chronological_requests([{**source[0],'partition':'calibration'}]))
        refused('2017 calibration year refused',lambda:r.news_chronological_requests([{'record_id':'x','partition':'train','original_fields':{'date':'2017-01-01'}}]))
        refused('missing date not inferred',lambda:r.news_chronological_requests([{'record_id':'x','partition':'train','original_fields':{}}]))
        refused('duplicate record IDs refused',lambda:r.news_chronological_requests([source[0],source[0]]))
        check('empty population is empty stream',r.news_chronological_requests([],horizon=0)['checkpoints']==[])
    result={'status':'passed','check_count':len(checks),'checks':checks,'code_sha256':r.file_sha256(r.__file__),
            'actual_WCEP_or_News_experiment_run':False,'synthetic_empirical_data_created':False,
            'scope':'transient explicitly named schema/calendar software fixtures; not a research dataset'}
    target=Path(__file__).parent/'results/replication_checks.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
