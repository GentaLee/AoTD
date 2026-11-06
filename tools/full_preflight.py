"""Audit the complete selected training pool and evaluation media before GPU use."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter
import yaml
from construction_io import records
from training_protocol import load_protocol, coverage_errors


def main():
    root=Path(os.environ['AOTD_ROOT'])
    out=Path(os.environ['AOTD_RUN_DIR']);out.mkdir(parents=True,exist_ok=True)
    split=os.environ.get('AOTD_EVAL_SPLIT','val')
    video=Path(os.environ.get('AOTD_VIDEO_DIR',str(root/'data/nextqa/NExTVideo')))
    protocol=load_protocol(root/'config/construction/sources.json')
    allowed={source['dataset'] for source in protocol['sources']}
    cot_sources={source['dataset'] for source in protocol['sources'] if source['cot']}
    report=dict(split=split,video_dir=str(video),errors=[],training={},protocol_id=protocol['protocol_id'],
                enabled_sources=sorted(allowed),excluded_sources=protocol.get('excluded_sources',[]))
    def error(message):
        report['errors'].append(message);print('[missing] '+message,flush=True)
    rows=list(csv.DictReader((root/'data/nextqa'/f'{split}.csv').open(encoding='utf-8-sig')))
    expected={'val':4996,'test':8564}[split]
    if len(rows)!=expected:error(f'Expected {expected} evaluation questions, got {len(rows)}')
    videos=sorted({row['video'] for row in rows})
    missing=[name for name in videos if not (video/f'{name}.mp4').is_file()]
    report.update(eval_questions=len(rows),eval_videos=len(videos),missing_video_ids=missing)
    if missing:error(f'{len(missing)} evaluation videos missing')
    configs={};qa_files=set();qa_ids={};reason_counts=Counter()
    if os.environ.get('AOTD_FULL_MODE','all')=='all':
        data=Path(os.environ.get('AOTD_TRAIN_DATA',str(root/'data/five_source_train')))
        report['training_protocol']='five_source_engineering_reconstruction'
        manifest={}
        try:
            manifest=json.loads((data/'manifest.json').read_text())
            if manifest.get('protocol_id')!=protocol['protocol_id'] or not manifest.get('full_pool'):
                raise ValueError('Training manifest is not a full pool for the selected protocol')
            inputs=manifest.get('input_sha256',{})
            if len(inputs)!=2:raise ValueError('Training manifest must identify QA and CoT input hashes')
            for name,expected_hash in inputs.items():
                with Path(name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual!=expected_hash:raise ValueError(f'Training source changed after preparation: {name}')
            if not manifest.get('training_file_sha256'):raise ValueError('Prepared training file hashes are missing')
        except Exception as exc:
            error(f'{exc}; existing data preserved. Use a fresh AOTD_TRAIN_DATA directory to rebuild the pool')
        for mode in ['instruct','aotd']:
            filename=data/f'{mode}.yaml'
            if not filename.is_file():error(f'Complete selected-source training YAML required: {filename}');continue
            try:
                config=yaml.safe_load(filename.read_text())
                total=0;bad=0;missing_train=0;datasets=Counter();sources=[];paths=set();digests=set();cot_ids=set()
                for entry in config['datasets']:
                    path=Path(entry['json_path'])
                    if not path.is_absolute():path=filename.parent/path
                    path=path.resolve();entry['json_path']=str(path)
                    if path in paths:raise ValueError(f'Duplicate training input: {path}')
                    paths.add(path)
                    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                    if digest in digests:raise ValueError('Duplicate training input content')
                    digests.add(digest)
                    if manifest.get('training_file_sha256',{}).get(path.name)!=digest:
                        error(f'Prepared training file changed or is untracked: {path}; use a fresh AOTD_TRAIN_DATA directory')
                    is_qa=mode=='instruct' or digest in qa_files
                    count=0;media_seen={}
                    for row in records(path):
                        count+=1;total+=1
                        name=row.get('dataset');sid=row.get('sample_id')
                        if name not in allowed:raise ValueError(f'Unexpected/excluded training dataset: {name}')
                        datasets[name]+=1
                        if not sid:raise ValueError('Training record missing sample_id')
                        if mode=='instruct':
                            if sid in qa_ids:raise ValueError(f'Duplicate QA sample: {sid}')
                            qa_ids[sid]=name
                        elif not is_qa:
                            if name not in cot_sources:raise ValueError(f'CoT is not enabled for {name}')
                            if qa_ids.get(sid)!=name:raise ValueError(f'CoT has no matching source QA: {sid}')
                            if sid in cot_ids:raise ValueError(f'Duplicate CoT sample: {sid}')
                            cot_ids.add(sid)
                            reason_counts[name]+=1
                        if not row.get('conversation') or not row.get('video_path'):bad+=1;continue
                        media_name=row['video_path']
                        if media_name not in media_seen:
                            media=Path(media_name);media_seen[media_name]=media.is_absolute() and media.is_file()
                        if not media_seen[media_name]:missing_train+=1
                    sources.append(dict(path=str(path),records=count,sha256=digest))
                    if mode=='instruct':qa_files.add(digest)
                if mode=='instruct':
                    for problem in coverage_errors(datasets,protocol):error(problem)
                elif not qa_files.issubset({source['sha256'] for source in sources}):
                    error('AoTD inputs must contain the same complete QA files as Instruct')
                if bad or missing_train:error(f'{mode}: invalid={bad}, missing training media={missing_train}')
                report['training'][mode]=dict(records=total,invalid_records=bad,missing_media=missing_train,
                                             sources=sources,dataset_counts=dict(datasets))
                configs[mode]=config
            except Exception as exc:error(f'{mode}: {exc}')
        if 'aotd' in report['training']:
            absent=cot_sources-set(reason_counts)
            if absent:error('Accepted CoT required for each enabled CoT source: '+', '.join(sorted(absent)))
        report['cot_dataset_counts']=dict(reason_counts)
        if 'instruct' in report['training'] and manifest.get('dataset_counts')!=report['training']['instruct']['dataset_counts']:
            error('Prepared manifest dataset counts do not match the audited QA pool')
        if 'instruct' in report['training'] and manifest.get('qa_records')!=report['training']['instruct']['records']:
            error('Prepared manifest QA total does not match the audited pool')
        if 'aotd' in report['training'] and manifest.get('cot_records')!=sum(reason_counts.values()):
            error('Prepared manifest CoT total does not match the audited pool')
    report['ready']=not report['errors']
    if report['ready']:
        for mode,config in configs.items():(out/f'{mode}.yaml').write_text(yaml.safe_dump(config))
    (out/'input_audit.json').write_text(json.dumps(report,indent=2))
    print(f'[full check] evaluation={len(rows)} questions, ready={report["ready"]}; report={out/"input_audit.json"}',flush=True)
    return 0 if report['ready'] else 2


if __name__=='__main__':sys.exit(main())
