"""Convert normalized source QA and filtered CoT to training records.

Input QA fields: sample_id,dataset,video_id,video_path,question,options,answer.
Input CoT fields: sample_id,rationale. No synthetic or placeholder CoT is made.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
from construction_io import atomic_json, records
from training_protocol import load_protocol, coverage_errors


ARTIFACTS=('answering.json','reasoning.json','instruct.yaml','aotd.yaml','manifest.json')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def yaml_text(output, mode):
    paths=[output/'answering.json']
    if mode=='aotd':paths.append(output/'reasoning.json')
    return 'datasets:\n'+''.join('  - json_path: '+json.dumps(str(path.resolve()))+'\n' for path in paths)


def publish(output, staged, intent):
    """Resume a verified publication; the manifest is always the final commit."""
    # Validate the complete transaction before publishing any additional file.
    for name in ARTIFACTS:
        final=output/name;pending=staged/name
        source=final if final.exists() else pending
        if not source.is_file() or digest(source)!=intent['artifact_sha256'][name]:
            raise ValueError(f'Conflicting/incomplete training artifact preserved: {source}')
    for name in ARTIFACTS:
        final=output/name
        if not final.exists():
            (staged/name).replace(final)
            sync_directory(output)


def prepare(qa_path, cot_path, output, full_pool=False):
    output=Path(output).resolve();qa_path=Path(qa_path);cot_path=Path(cot_path)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return prepare_locked(qa_path,cot_path,output,full_pool)


def prepare_locked(qa_path, cot_path, output, full_pool):
    protocol=load_protocol()
    allowed={source['dataset'] for source in protocol['sources']}
    cot_sources={source['dataset'] for source in protocol['sources'] if source['cot']}
    for path in [qa_path,cot_path]:
        if not path.is_file():raise FileNotFoundError(f'Real training source required: {path}')
    hashes={str(path.resolve()):digest(path) for path in [qa_path,cot_path]}
    expected=dict(protocol_id=protocol['protocol_id'],full_pool=full_pool,input_sha256=hashes)
    staged=output/'.prepare-publication';staged.mkdir(exist_ok=True)
    intent_path=staged/'publication.json'
    if intent_path.is_file():
        intent=json.loads(intent_path.read_text())
        if intent.get('inputs')!=expected:
            raise ValueError('Training source/protocol changed during publication; preserve the pool and use a fresh output directory')
        publish(output,staged,intent)
        manifest=json.loads((output/'manifest.json').read_text())
        print(f'[prepare] verified/resumed {manifest["qa_records"]} QA + {manifest["cot_records"]} real filtered CoT; {output}')
        return manifest
    if (output/'manifest.json').is_file():
        manifest=json.loads((output/'manifest.json').read_text())
        if any(manifest.get(key)!=value for key,value in expected.items()):
            raise ValueError('Existing training source/protocol changed; preserve the pool and use a fresh output directory')
        for name in ['answering.json','reasoning.json']:
            path=output/name
            if not path.is_file() or digest(path)!=manifest.get('training_file_sha256',{}).get(name):
                raise ValueError(f'Existing training artifact changed or missing: {path}')
        for mode in ['instruct','aotd']:
            if (output/f'{mode}.yaml').read_text()!=yaml_text(output,mode):
                raise ValueError(f'Existing training YAML changed: {mode}')
        if full_pool:
            problems=coverage_errors(manifest.get('dataset_counts',{}),protocol)
            if problems:raise ValueError('; '.join(problems))
        print(f'[prepare] verified existing training pool; {output}')
        return manifest
    cot={}
    for row in records(cot_path):
        sid=row['sample_id'];text=row.get('rationale','').strip()
        if sid in cot or not text:raise ValueError(f'Duplicate/empty CoT: {sid}')
        cot[sid]=text
    qa=records(qa_path);seen=set();sources=Counter();reason_count=0;qa_count=0
    answer_path=output/'answering.json';reason_path=output/'reasoning.json'
    answer_tmp=staged/answer_path.name;reason_tmp=staged/reason_path.name
    with answer_tmp.open('w') as answering,reason_tmp.open('w') as reasoning:
        answering.write('[\n');reasoning.write('[\n')
        for i,row in enumerate(qa):
            sid=row['sample_id']
            if row['dataset'] not in allowed:raise ValueError(f'Unexpected/excluded source: {row["dataset"]}')
            if sid in cot and row['dataset'] not in cot_sources:raise ValueError(f'CoT not enabled for {row["dataset"]}')
            if sid in seen:raise ValueError(f'Duplicate QA ID: {sid}')
            seen.add(sid);sources[row['dataset']]+=1
            qa_count+=1
            media=Path(row['video_path'])
            if not media.is_absolute() or not media.is_file():raise FileNotFoundError(media)
            question=row['question'].strip();options=row.get('options',[])
            if not question:raise ValueError(f'Empty question: {sid}')
            if options:
                answer=int(row['answer'])
                if not 0<=answer<len(options)<=26:raise ValueError(f'Invalid answer/options: {sid}')
                target=chr(65+answer)
                text='Question: '+question+'\nOptions:\n'+'\n'.join(f'({chr(65+k)}) {v}' for k,v in enumerate(options))
                suffix="Answer with the option's letter from the given choices directly and only give the best option."
            else:
                target=str(row['answer']).strip();text='Question: '+question;suffix='Answer the question directly.'
                if not target:raise ValueError(sid)
            def record(prompt,response):
                return dict(sample_id=sid,dataset=row['dataset'],video_path=str(media),conversation=[
                    dict(role='user',content=[dict(type='text',text=text+'\n'+prompt),dict(type='video')]),
                    dict(role='assistant',content=[dict(type='text',text=response)])])
            if i:answering.write(',\n')
            answering.write(json.dumps(record(suffix,target)))
            if sid in cot:
                if reason_count:reasoning.write(',\n')
                reasoning.write(json.dumps(record('Explain the rationale to answer the question.',cot[sid])))
                reason_count+=1
        answering.write('\n]\n');reasoning.write('\n]\n')
        for stream in [answering,reasoning]:stream.flush();os.fsync(stream.fileno())
    if set(cot)-seen:raise ValueError('CoT IDs do not all match QA IDs')
    if full_pool:
        problems=coverage_errors(sources,protocol)
        if problems:raise ValueError('; '.join(problems))
        if not reason_count:raise ValueError('Full AoTD pool requires accepted CoT records')
    for path in [qa_path,cot_path]:
        if digest(path)!=hashes[str(path.resolve())]:raise ValueError('Training source changed during preparation; refusing publication')
    output_hashes={answer_path.name:digest(answer_tmp),reason_path.name:digest(reason_tmp)}
    for mode in ['instruct','aotd']:
        with (staged/f'{mode}.yaml').open('w') as stream:
            stream.write(yaml_text(output,mode));stream.flush();os.fsync(stream.fileno())
    manifest=dict(**expected,qa_records=qa_count,cot_records=reason_count,
                  dataset_counts=dict(sources),training_file_sha256=output_hashes)
    atomic_json(staged/'manifest.json',manifest)
    intent=dict(version=1,inputs=expected,artifact_sha256={name:digest(staged/name) for name in ARTIFACTS})
    atomic_json(intent_path,intent);sync_directory(staged)
    publish(output,staged,intent)
    print(f'[prepare] {qa_count} QA + {reason_count} real filtered CoT; {output}')
    return manifest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--qa',type=Path,required=True)
    p.add_argument('--cot',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--full-pool',action='store_true',help='Require complete coverage of every enabled source')
    a=p.parse_args()
    prepare(a.qa,a.cot,a.output_dir,a.full_pool)

if __name__=='__main__':main()
