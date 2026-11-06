"""Normalize public train annotations; preserve all records and report absent media."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from construction_io import atomic_json, records, sha256


def normalize(dataset, annotation, video_dir):
    if dataset == 'NExT-QA':
        seen = set()
        with annotation.open(encoding='utf-8-sig') as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                video = row['video']
                original_id = f"{video}_{row['qid']}"
                sid = original_id if original_id not in seen else f'{original_id}__row{index}'
                seen.add(original_id)
                yield dict(sample_id=sid, source_question_id=original_id, annotation_row=index,
                           dataset=dataset, video_id=video,
                           video_path=str(video_dir / f'{video}.mp4'), question=row['question'],
                           options=[row[f'a{i}'] for i in range(5)], answer=int(row['answer']), split='train')
    elif dataset == 'STAR':
        for row in records(annotation):
            choices = row['choices']
            labels = [str(choice['choice_id']) for choice in choices]
            texts = [choice['choice'] for choice in choices]
            answer = texts.index(row['answer']) if row['answer'] in texts else labels.index(str(row['answer']))
            # Official STAR clips are question-specific windows, not whole Charades videos.
            sid = row['question_id']
            yield dict(sample_id='STAR:' + sid, dataset=dataset, video_id=row['video_id'],
                       video_path=str(video_dir / f'{sid}.mp4'), question=row['question'],
                       options=[choice['choice'] for choice in choices], answer=answer,
                       start_seconds=float(row['start']), end_seconds=float(row['end']), split='train')
    else:
        # Other original sources enter through the same explicit, audited train schema.
        # Their original sampling IDs were not published; do not guess schema or samples.
        for row in records(annotation):
            if row.get('dataset') != dataset or row.get('split') != 'train':
                raise ValueError(f'{dataset}: explicit dataset and train split required')
            yield row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', required=True, choices=['NExT-QA', 'STAR', 'AGQA', 'ANetQA', 'CLEVRER', 'EgoQA'])
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.video_dir.is_absolute():
        raise ValueError('Use target-machine absolute media paths')
    if args.dataset in {'NExT-QA', 'STAR'} and 'train' not in args.annotations.stem.lower():
        raise ValueError('Training annotations must explicitly identify the train split')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.partial')
    seen = set(); media = {}; missing = Counter(); count = 0
    with temporary.open('w') as stream:
        for row in normalize(args.dataset, args.annotations, args.video_dir):
            sid = row['sample_id']
            if sid in seen or not row['question'].strip():
                raise ValueError(f'Duplicate ID or empty question: {sid}')
            seen.add(sid)
            path = Path(row['video_path'])
            if not path.is_absolute():
                raise ValueError(f'Non-absolute media path: {sid}')
            options = row.get('options', [])
            if options and not 0 <= int(row['answer']) < len(options) <= 26:
                raise ValueError(f'Invalid choices: {sid}')
            if not options and not str(row['answer']).strip():
                raise ValueError(f'Empty answer: {sid}')
            if str(path) not in media:
                media[str(path)] = path.is_file()
            if not media[str(path)]:
                missing[str(path)] += 1
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
            if count % 5000 == 0:
                print(f'[normalize {args.dataset}] {count} QA', flush=True)
    if not count:
        raise ValueError('No train annotations')
    temporary.replace(args.output)
    audit = dict(dataset=args.dataset, split='train', records=count, videos=len(media),
                 missing_videos=len(missing), missing_qa=sum(missing.values()),
                 missing_media=dict(missing), annotation_sha256=sha256(args.annotations),
                 output_sha256=sha256(args.output), media_ready=not missing,
                 sampling='all input train records; no fabricated author sampling IDs')
    atomic_json(args.output.with_suffix('.audit.json'), audit)
    print(f'[normalize] {count} QA, {len(media)} videos, {len(missing)} missing; {args.output}', flush=True)


if __name__ == '__main__':
    main()
