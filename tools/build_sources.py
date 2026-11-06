"""Prepare/audit six-source inputs and orchestrate resumable offline CoT stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from construction_io import atomic_json, records

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'env/bin/python')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mode',choices=['prepare','check','run'],required=True)
    p.add_argument('--dataset',help='Explicit single-source preparation; full recipe requires all six')
    p.add_argument('--stage',choices=['all','construct','transfer','filter','export'],default='all')
    p.add_argument('--retry-errors',action='store_true')
    a=p.parse_args()
    cfg=json.loads((ROOT/'config/construction/sources.json').read_text())
    selected=[s for s in cfg['sources'] if not a.dataset or s['dataset']==a.dataset]
    if not selected: raise ValueError('Unknown source dataset')
    reports=[]; bad=False
    base=ROOT/'data/construction';base.mkdir(parents=True,exist_ok=True)
    for source in selected:
        name=source['dataset'];annotation=ROOT/source['annotations'];media=ROOT/source['media']
        output=base/name;output.mkdir(parents=True,exist_ok=True)
        qa=output/'qa_normalized.jsonl'
        item=dict(dataset=name,annotations=str(annotation),origin=source['origin'],cot=source['cot'])
        if name=='NExT-QA':subprocess.run([PY,'-B',str(ROOT/'tools/link_nextqa_train.py')],check=True)
        native={'AGQA':('data/raw/AGQA/AGQA_balanced.zip',25000,'Charades'),
                'ANetQA':('data/raw/ANetQA/qa_train.json',25000,'ANetQA'),
                'CLEVRER':('data/raw/CLEVRER/train.json',21000,'CLEVRER')}
        if not annotation.exists() and name in native:
            raw,count,media_name=native[name]
            if (ROOT/raw).is_file():
                subprocess.run([PY,'-B',str(ROOT/'tools/sample_native_qa.py'),'--dataset',name,
                                '--input',str(ROOT/raw),'--output',str(annotation),
                                '--video-dir',str(base/'media'/media_name),'--count',str(count)],check=True)
        if name=='CLEVRER' and annotation.exists():
            subprocess.run([PY,'-B',str(ROOT/'tools/link_clevrer_train.py')],check=True)
        if not annotation.is_file():
            item.update(ready=False,reason='train_annotations_not_staged');bad=True
            print(f'[prepare missing] {name}: {annotation}',flush=True)
        else:
            subprocess.run([PY,'-B',str(ROOT/'tools/normalize_construction_sources.py'),
                            '--dataset',name,'--annotations',str(annotation),'--video-dir',str(media),
                            '--output',str(qa)],check=True)
            audit=json.loads(qa.with_suffix('.audit.json').read_text())
            item.update(records=audit['records'],videos=audit['videos'],missing_videos=audit['missing_videos'],ready=audit['media_ready'])
            bad |= not item['ready']
        reports.append(item)
    report=dict(sources=reports,all_six_sources=len(selected)==6,ready=not bad,
                protocol=cfg['protocol'],phase='annotations_and_media')
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
    keep=os.environ.get('AOTD_KEEPALIVE_SCRIPT','/volume/pt-test/users/zhaoye/gpu-workspace/keep_alive/main.py')
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
    if len(selected)==6 and a.stage=='all':
        destination=ROOT/'data/original_sources';destination.mkdir(parents=True,exist_ok=True)
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
            print(f'[six-source export] {filename}: {count} records',flush=True)
            if count==0:raise ValueError(f'Empty full construction output: {filename}')
        atomic_json(destination/'provenance.json',dict(protocol=cfg['protocol'],sources=reports,
                    original_author_data=False,real_agent_reconstruction=True))


if __name__=='__main__':main()
