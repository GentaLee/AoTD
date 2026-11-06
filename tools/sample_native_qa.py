"""Reproducible public-source reconstruction; original author sample IDs are unknown."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import zipfile
from construction_io import atomic_json, records, sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',choices=['AGQA','ANetQA','CLEVRER'],required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video-dir',type=Path,required=True)
    p.add_argument('--count',type=int,default=25000)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    import ijson
    def source():
        if a.dataset=='AGQA':
            with zipfile.ZipFile(a.input) as archive:
                with archive.open('AGQA_balanced/train_balanced.txt') as stream:
                    for key,row in ijson.kvitems(stream,''):
                        yield key,row
        else:
            if a.dataset=='ANetQA':
                for row in records(a.input):yield row['qa_id'],row
            else:
                for video in records(a.input):
                    if not 0<=int(video['scene_index'])<10000:
                        raise ValueError('CLEVRER source contains a non-train scene')
                    for question in video['questions']:
                        choices=question.get('choices',[])
                        # Native CLEVRER multiple-choice is scored per candidate;
                        # several choices can be correct. Preserve each original
                        # binary label instead of arbitrarily choosing one answer.
                        for choice in choices:
                            key=f"{video['scene_index']}_{question['question_id']}_{choice['choice_id']}"
                            yield key,dict(video_id=Path(video['video_filename']).stem,
                                question=question['question']+'\nCandidate statement: '+choice['choice']+
                                         '\nIs this candidate a correct answer to the question?',
                                answer=int(choice['answer']=='correct'),options=['No','Yes'])
    heap=[];visited=0
    for key,row in source():
        priority=int.from_bytes(hashlib.sha256(f'{a.seed}:{key}'.encode()).digest(),'big')
        item=(-priority,key,dict(sample_id=f'{a.dataset}:{key}',dataset=a.dataset,split='train',
                                 source_question_id=key,video_id=row['video_id'],
                                 video_path=str(a.video_dir/f"{row['video_id']}.mp4"),
                                 question=row['question'],options=row.get('options',[]),answer=row['answer']))
        if len(heap)<a.count:heapq.heappush(heap,item)
        elif item>heap[0]:heapq.heapreplace(heap,item)
        visited+=1
        if visited%250000==0:print(f'[sample {a.dataset}] scanned={visited} selected={len(heap)}',flush=True)
    if len(heap)!=a.count:raise ValueError('Not enough real source QA')
    selected=sorted(heap,key=lambda x:x[1]);a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_name(a.output.name+'.partial')
    with temporary.open('w') as stream:
        for _,key,row in selected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    temporary.replace(a.output)
    atomic_json(a.output.with_suffix('.sampling.json'),dict(dataset=a.dataset,train_candidates=visited,
                selected=len(selected),seed=a.seed,sampling='lowest SHA256(seed:original_question_id)',
                original_author_sampling=False,annotation_sha256=sha256(a.input),output_sha256=sha256(a.output),
                eligibility='native train MC candidate labels, binary Yes/No; author MC serialization unknown' if a.dataset=='CLEVRER' else 'source train annotations'))
    print(f'[sample {a.dataset}] {len(selected)} real train QA; original author IDs not claimed',flush=True)


if __name__=='__main__':main()
