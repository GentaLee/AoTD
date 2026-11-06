"""Convert normalized, real six-source QA and filtered CoT to official records.

Input QA fields: sample_id,dataset,video_id,video_path,question,options,answer.
Input CoT fields: sample_id,rationale. No synthetic or placeholder CoT is made.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from construction_io import records

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--qa',type=Path,required=True)
    p.add_argument('--cot',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    for path in [a.qa,a.cot]:
        if not path.is_file():raise FileNotFoundError(f'Real training source required: {path}')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    cot={}
    for row in records(a.cot):
        sid=row['sample_id'];text=row.get('rationale','').strip()
        if sid in cot or not text:raise ValueError(f'Duplicate/empty CoT: {sid}')
        cot[sid]=text
    qa=records(a.qa);seen=set();sources=Counter();reason_count=0;qa_count=0
    answer_path=a.output_dir/'answering.json';reason_path=a.output_dir/'reasoning.json'
    # Refuse to overwrite data that may be in use by another run.
    for f in [answer_path,reason_path]:
        if f.exists():raise FileExistsError(f)
    answer_tmp=answer_path.with_suffix('.json.partial');reason_tmp=reason_path.with_suffix('.json.partial')
    with answer_tmp.open('w') as answering,reason_tmp.open('w') as reasoning:
        answering.write('[\n');reasoning.write('[\n')
        for i,row in enumerate(qa):
            sid=row['sample_id']
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
    if set(cot)-seen:raise ValueError('CoT IDs do not all match QA IDs')
    answer_tmp.replace(answer_path);reason_tmp.replace(reason_path)
    for name,paths in [('instruct',[answer_path]),('aotd',[answer_path,reason_path])]:
        text='datasets:\n'+''.join('  - json_path: '+json.dumps(str(x.resolve()))+'\n' for x in paths)
        (a.output_dir/f'{name}.yaml').write_text(text)
    hashes={}
    for path in [a.qa,a.cot]:
        with path.open('rb') as f:hashes[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()
    (a.output_dir/'manifest.json').write_text(json.dumps(dict(qa_records=qa_count,cot_records=reason_count,dataset_counts=dict(sources),input_sha256=hashes),indent=2))
    print(f'[prepare] {qa_count} QA + {reason_count} real filtered CoT; {a.output_dir}')

if __name__=='__main__':main()
