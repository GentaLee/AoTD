"""Check every input before taking over GPUs for the full recipe."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import yaml
from collections import Counter

root=Path(os.environ['AOTD_ROOT'])
out=Path(os.environ['AOTD_RUN_DIR'])
split=os.environ.get('AOTD_EVAL_SPLIT','val')
csv_path=root/'data/nextqa'/f'{split}.csv'
video=Path(os.environ.get('AOTD_VIDEO_DIR',str(root/'data/nextqa/NExTVideo')))
report={'split':split,'video_dir':str(video),'errors':[],'training':{}}
def records(path):
    # Keep memory bounded while checking large QA/CoT JSON arrays.
    decoder=json.JSONDecoder()
    with path.open() as f:
        buf=f.read(65536).lstrip()
        if not buf.startswith('['):raise ValueError(f'Expected JSON record array: {path}')
        buf=buf[1:]
        while True:
            buf=buf.lstrip(' \t\r\n,')
            if buf.startswith(']'):return
            try:row,end=decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk=f.read(65536)
                if not chunk:raise
                buf+=chunk;continue
            buf=buf[end:]
            yield row
def error(s):
    report['errors'].append(s);print('[missing] '+s,flush=True)
rows=list(csv.DictReader(csv_path.open(encoding='utf-8-sig')))
expected={'val':4996,'test':8564}[split]
if len(rows)!=expected:error(f'{csv_path}: expected {expected} questions, got {len(rows)}')
missing=[v for v in sorted({x['video'] for x in rows}) if not (video/f'{v}.mp4').is_file()]
report.update(eval_questions=len(rows),eval_videos=len({x['video'] for x in rows}),missing_video_ids=missing)
if missing:error(f'{len(missing)} evaluation videos missing in {video}')
if os.environ.get('AOTD_FULL_MODE','all')=='all':
    data=Path(os.environ.get('AOTD_TRAIN_DATA',str(root/'data/original_six_source')))
    provenance_path=root/'data/original_sources/provenance.json'
    rebuilt=provenance_path.is_file() and json.loads(provenance_path.read_text()).get('real_agent_reconstruction',False)
    report['training_protocol']='real_agent_reconstruction' if rebuilt else 'author_processed_data'
    # Genuine reconstruction yield depends on actual execution/filter results;
    # retain the full QA requirement without manufacturing the paper's CoT count.
    for mode,minimum in [('instruct',150000),('aotd',150001 if rebuilt else 180000)]:
        y=data/f'{mode}.yaml'
        if not y.is_file():error(f'Original six-source training YAML required: {y}');continue
        try:
            config=yaml.safe_load(y.read_text());total=0;bad=0;missing_train=0;datasets=Counter()
            sources=[]
            for entry in config['datasets']:
                p=Path(entry['json_path'])
                if not p.is_absolute():p=y.parent/p
                entry['json_path']=str(p.resolve())
                if not p.is_file():raise FileNotFoundError(p)
                count=0
                media_seen={}
                for row in records(p):
                    count+=1
                    if row.get('dataset'):datasets[row['dataset']]+=1
                    if not row.get('conversation') or not row.get('video_path'):bad+=1;continue
                    name=row['video_path']
                    if name not in media_seen:media_seen[name]=Path(name).is_absolute() and Path(name).is_file()
                    if not media_seen[name]:missing_train+=1
                total+=count
                with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
                sources.append(dict(path=str(p),records=count,sha256=h))
            report['training'][mode]=dict(records=total,invalid_records=bad,missing_media=missing_train,sources=sources,dataset_counts=dict(datasets))
            if rebuilt and set(datasets)!={'AGQA','ANetQA','STAR','NExT-QA','CLEVRER','EgoQA'}:
                error(f'{mode}: rebuilt full training must include all six source datasets')
            if total<minimum:error(f'{mode}: {total} records; expected original full training pool (at least {minimum}); refusing reduced/smoke data')
            if bad or missing_train:error(f'{mode}: invalid={bad}, missing training media={missing_train}')
            (out/f'{mode}.yaml').write_text(yaml.safe_dump(config))
        except Exception as e:error(f'{mode}: {e}')
    if rebuilt and all(mode in report['training'] for mode in ['instruct','aotd']):
        if report['training']['aotd']['records']<=report['training']['instruct']['records']:
            error('AoTD reconstruction has no additional accepted CoT records')
report['ready']=not report['errors']
(out/'input_audit.json').write_text(json.dumps(report,indent=2))
print(f"[full check] evaluation={len(rows)} questions, ready={report['ready']}; report={out/'input_audit.json'}",flush=True)
sys.exit(0 if report['ready'] else 2)
