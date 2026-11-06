"""Validate real media and exact processor/collator without allocating CUDA."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import torch
from transformers import AutoProcessor

root = Path(os.environ['AOTD_ROOT'])
spec = importlib.util.spec_from_file_location('aotd_train', root/'distillation/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
spec = importlib.util.spec_from_file_location('aotd_eval', root/'tools/eval_nextqa.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
assert evaluation.parse_letter('A') == 0
assert evaluation.parse_letter('(E).') == 4
assert evaluation.parse_letter('Answer: B') == 1
assert evaluation.parse_letter('A or B') is None
assert evaluation.parse_letter('') is None
report = []
for name in ['AoTD-7B-author', 'LLaVA-7B-base']:
    processor = AutoProcessor.from_pretrained(root/'models'/name, local_files_only=True, use_fast=False)
    with (root/'data/nextqa/val.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))[:2]
    for row in rows:
        path = root/'data/nextqa/NExTVideo'/f"{row['video']}.mp4"
        video = evaluation.get_video(path)
        conversation = [dict(role='user', content=[dict(type='text', text=evaluation.question_prompt(row)), dict(type='video')])]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        batch = processor(text=text, videos=video, padding=True, return_tensors='pt')
        count = (batch['input_ids'] == 32000).sum().item()
        assert count == 4608, count
        record = dict(model=name, sample_id=f"{row['video']}_{row['qid']}", frames=list(video.shape),
                      input_tokens=batch['input_ids'].shape[-1], video_tokens=count,
                      pixels=list(batch['pixel_values_videos'].shape))
        report.append(record)
        print('[cpu check]', record, flush=True)
    if name == 'LLaVA-7B-base':
        data = json.loads((root/'data/train_smoke/answering.json').read_text())
        collated = train.LlavaNextVideoDataCollatorWithPadding(processor)(data[:1])
        assert (collated['input_ids'] == 32000).sum().item() == 4608
        assert collated['labels'].shape == collated['input_ids'].shape
        assert torch.all(collated['labels'][collated['input_ids'] == processor.tokenizer.pad_token_id] == -100)
        assert (collated['labels'] != -100).sum().item() > 0
(root/'logs/cpu_processor_validation.json').write_text(json.dumps(dict(status='PASS',cuda_allocated=False,records=report),indent=2))
print('[cpu check] PASS: decode + processor 4608 video tokens + original training collator', flush=True)
