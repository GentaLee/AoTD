"""Repeat two real train QA samples for an isolated two-step batch-48 check."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root / 'data' / 'train_smoke'
data.mkdir(exist_ok=True, parents=True)
with (root / 'data/nextqa/train.csv').open(newline='', encoding='utf-8-sig') as stream:
    rows = list(csv.DictReader(stream))[:2]
records = []
for row in rows:
    path = root / 'data/nextqa/NExTVideo' / f"{row['video']}.mp4"
    assert path.is_file(), path
    question = ('Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\n'
                "Answer with the option's letter from the given choices directly and only give the best option.").format(**row)
    records.append(dict(sample_id=f"{row['video']}_{row['qid']}", video_path=str(path), conversation=[
        dict(role='user', content=[dict(type='text', text=question), dict(type='video')]),
        dict(role='assistant', content=[dict(type='text', text=chr(65+int(row['answer'])))])]))
unique_ids=[x['sample_id'] for x in records]
# Deliberate repetition only for validating gradient accumulation, never full data.
records=[dict(records[index%len(records)],sample_id=f"smoke_repeat_{index}") for index in range(96)]
(data / 'answering.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
(data / 'instruct.yaml').write_text('datasets:\n  - json_path: ' + json.dumps(str(data/'answering.json')) + '\n')
(data / 'manifest.json').write_text(json.dumps(dict(scope='QA-only repeated-fixture validation; not full training',
    split='train', source_csv_sha256=hashlib.sha256((root/'data/nextqa/train.csv').read_bytes()).hexdigest(),
    unique_source_sample_ids=unique_ids,records=len(records)), indent=2))
print(f'[data] prepared {len(records)} TRAIN QA records: {data}')
