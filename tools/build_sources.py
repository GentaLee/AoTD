"""Prepare/audit selected train sources and orchestrate resumable offline CoT stages."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from construction_io import atomic_json, records, sha256
from training_protocol import load_protocol

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'env/bin/python')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mode',choices=['prepare','check','run'],required=True)
    p.add_argument('--dataset',help='Explicit single-source preparation; full recipe requires every enabled source')
    p.add_argument('--stage',choices=['all','construct','transfer','filter','export'],default='all')
    p.add_argument('--retry-errors',action='store_true')
    a=p.parse_args()
    cfg=load_protocol()
    selected=[s for s in cfg['sources'] if not a.dataset or s['dataset']==a.dataset]
    if not selected: raise ValueError('Unknown source dataset')
    reports=[]; bad=False
    base=Path(os.environ.get('AOTD_CONSTRUCTION_DIR',str(ROOT/'data/construction')))
    base.mkdir(parents=True,exist_ok=True)
    lock=(base/'orchestrator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for source in selected:
        name=source['dataset'];annotation=ROOT/source['annotations'];media=ROOT/source['media']
        output=base/name;output.mkdir(parents=True,exist_ok=True)
        qa=output/'qa_normalized.jsonl'
        item=dict(dataset=name,annotations=str(annotation),origin=source['origin'],cot=source['cot'])
        if name=='NExT-QA':subprocess.run([PY,'-B',str(ROOT/'tools/link_nextqa_train.py')],check=True)
        native={'AGQA':('data/raw/AGQA/AGQA_balanced.zip',25000,'Charades'),
                'CLEVRER':('data/raw/CLEVRER/train.json',21000,'CLEVRER')}
        if not annotation.exists() and name in native:
            raw,count,media_name=native[name]
            if (ROOT/raw).is_file():
                subprocess.run([PY,'-B',str(ROOT/'tools/sample_native_qa.py'),'--dataset',name,
                                '--input',str(ROOT/raw),'--output',str(annotation),
                                '--video-dir',str(ROOT/'data/construction/media'/media_name),'--count',str(count)],check=True)
        if name=='CLEVRER' and annotation.exists():
            subprocess.run([PY,'-B',str(ROOT/'tools/link_clevrer_train.py')],check=True)
        if name=='EgoQA' and (ROOT/'data/raw/EgoQA/videochat2/train.json').is_file():
            native_annotations=ROOT/'data/raw/EgoQA/videochat2/train.json'
            current={}
            audit_path=annotation.with_suffix('.source_audit.json')
            if audit_path.is_file():current=json.loads(audit_path.read_text())
            exclusions=ROOT/source.get('exclusions_manifest','reproduction/metadata/egoqa_media_exclusions.json')
            exclusions_hash=sha256(exclusions) if exclusions.is_file() else None
            needs_import=(not annotation.is_file() or
                          current.get('annotation_sha256')!=sha256(native_annotations) or
                          current.get('exclusions_manifest_sha256')!=exclusions_hash or
                          current.get('video_dir')!=str(media.resolve()))
            if not needs_import:needs_import=current.get('output_sha256')!=sha256(annotation)
            if needs_import:
                command=[PY,'-B',str(ROOT/'tools/prepare_egoqa.py'),
                         '--input',str(native_annotations),'--video-dir',str(media),'--output',str(annotation)]
                if exclusions.is_file():command.extend(['--exclusions',str(exclusions)])
                subprocess.run(command,check=True)
        if not annotation.is_file():
            item.update(ready=False,reason='train_annotations_not_staged');bad=True
            print(f'[prepare missing] {name}: {annotation}',flush=True)
        else:
            subprocess.run([PY,'-B',str(ROOT/'tools/normalize_construction_sources.py'),
                            '--dataset',name,'--annotations',str(annotation),'--video-dir',str(media),
                            '--output',str(qa)],check=True)
            audit=json.loads(qa.with_suffix('.audit.json').read_text())
            item.update(records=audit['records'],videos=audit['videos'],missing_videos=audit['missing_videos'],ready=audit['media_ready'])
            if audit['records'] < source['minimum_qa_records']:
                item.update(ready=False,reason='incomplete_source_qa_pool')
                print(f'[prepare missing] {name}: {audit["records"]} QA, minimum {source["minimum_qa_records"]}',flush=True)
            bad |= not item['ready']
        reports.append(item)
    report=dict(sources=reports,all_selected_sources=len(selected)==len(cfg['sources']),ready=not bad,
                protocol=cfg['protocol'],protocol_id=cfg['protocol_id'],excluded_sources=cfg.get('excluded_sources',[]),phase='annotations_and_media')
    atomic_json(base/('readiness.json' if not a.dataset else f'{a.dataset}.readiness.json'),report)
    if a.mode=='prepare':
        print(f'[prepare] input audit saved; ready={not bad}; no GPU inference started',flush=True)
        return
    if bad:raise SystemExit(2)
    for source in selected:
        if not source['cot']: continue
        output=base/source['dataset'];qa=output/'qa_normalized.jsonl'
        check=subprocess.run([PY,'-B',str(ROOT/'tools/build_cot.py'),'--stage','check','--qa',str(qa),'--output-dir',str(output)])
        if check.returncode:bad=True
    if bad: raise SystemExit(2)
    if a.mode=='check':return
    keep=os.environ.get('AOTD_KEEPALIVE_SCRIPT',str(ROOT/'runtime/keep_alive/main.py'))
    run=Path(os.environ['AOTD_RUN_DIR'])
    for source in selected:
        if not source['cot']:continue
        name=source['dataset'];output=base/name;qa=output/'qa_normalized.jsonl'
        stages=['construct','transfer','filter','export'] if a.stage=='all' else [a.stage]
        for stage in stages:
            command=[PY,'-B',str(ROOT/'tools/build_cot.py'),'--stage',stage,'--qa',str(qa),'--output-dir',str(output)]
            if a.retry_errors:command.append('--retry-errors')
            if stage!='export':
                state=run/f'{name}_{stage}';state.mkdir(parents=True,exist_ok=True)
                mode='all' if stage=='construct' else 'eval-only'
                gpus=os.environ.get('AOTD_TRAIN_GPUS','0,1,2,3') if stage=='construct' else os.environ.get('AOTD_EVAL_GPU','0')
                command=[PY,'-B',str(ROOT/'tools/with_spare_keepalive.py'),'--mode',mode,
                         '--keepalive-script',keep,'--state-dir',str(state),'--',
                         'env',f'CUDA_VISIBLE_DEVICES={gpus}',*command]
            subprocess.run(command,check=True)
    # Export only after all sources are available and all full stages completed.
    if len(selected)==len(cfg['sources']) and a.stage=='all':
        destination=ROOT/'data/five_source_sources';destination.mkdir(parents=True,exist_ok=True)
        for filename in ['qa_normalized.jsonl','cot_filtered.jsonl']:
            temporary=destination/(filename+'.partial')
            seen=set();count=0
            with temporary.open('w') as stream:
                for source in selected:
                    if filename.startswith('cot') and not source['cot']:continue
                    for row in records(base/source['dataset']/filename):
                        if row['sample_id'] in seen:raise ValueError('Cross-source ID collision')
                        seen.add(row['sample_id']);count+=1
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            temporary.replace(destination/filename)
            print(f'[source export] {filename}: {count} records',flush=True)
            if count==0:raise ValueError(f'Empty full construction output: {filename}')
        atomic_json(destination/'provenance.json',dict(protocol=cfg['protocol'],sources=reports,
                    protocol_id=cfg['protocol_id'],excluded_sources=cfg.get('excluded_sources',[]),
                    original_author_data=False,real_agent_reconstruction=True))


if __name__=='__main__':main()
