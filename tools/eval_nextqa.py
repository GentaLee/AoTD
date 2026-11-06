"""AoTD fixed-commit NExT-QA protocol, with deterministic auditable output."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import (LlavaNextVideoForConditionalGeneration,
                          LlavaNextVideoProcessor, StoppingCriteria, StoppingCriteriaList)


def sha256(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def parse_letter(text):
    # strict_letter_v1: only an unambiguous standalone answer, never a guess.
    match = re.fullmatch(r'\s*(?:(?:Answer|Option)\s*:\s*)?(?:\(([A-E])\)|([A-E]))[\s.!]*', text)
    return ord(match.group(1) or match.group(2)) - ord('A') if match else None


def get_video(path):
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if len(vr) == 0:
        raise ValueError(f'empty video: {path}')
    # Identical frame indices to the fixed AoTD evaluation source.
    indices = np.linspace(0, len(vr), 32, endpoint=False).astype(int)
    return vr.get_batch(indices).asnumpy()


def question_prompt(row):
    return ('Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\n'
            "Answer with the option's letter from the given choices directly and only give the best option.").format(**row)


class TokenProgress(StoppingCriteria):
    def __init__(self, input_length):
        self.input_length = input_length
        self.start = time.monotonic()

    def __call__(self, input_ids, scores, **kwargs):
        n = input_ids.shape[1] - self.input_length
        if n == 1 or n % 32 == 0:
            print(f'[generate] tokens={n}/1024 elapsed={time.monotonic()-self.start:.1f}s', flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--attn_implementation', default='sdpa', choices=['sdpa', 'eager', 'flash_attention_2'])
    args = parser.parse_args()
    if Path(args.data_dir).stem != args.split:
        raise ValueError('CSV filename must match --split')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    output = Path(args.save_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output.with_suffix('.jsonl')
    if output.exists() or jsonl_path.exists():
        raise FileExistsError(f'refusing to overwrite {output} / {jsonl_path}')
    with open(args.data_dir, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    selected = rows if args.limit is None else rows[:args.limit]
    if not selected:
        raise ValueError('empty dataset')
    for row in selected:
        assert 0 <= int(row['answer']) <= 4
        path = Path(args.video_dir) / f"{row['video']}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.manual_seed(42)
    assert torch.cuda.is_available(), 'CUDA is required'
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    print(f'[eval] loading {args.model_path}; {len(selected)}/{len(rows)} samples, 32 frames, BF16, {args.attn_implementation}', flush=True)
    processor = LlavaNextVideoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation=args.attn_implementation, local_files_only=True).eval()
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    load_seconds = time.monotonic() - start
    meta = dict(split=args.split, dataset_size=len(rows), limit=args.limit,
                full_dataset=len(selected) == len(rows), model_path=args.model_path,
                csv_sha256=sha256(args.data_dir), script_sha256=sha256(__file__),
                parsing='strict_letter_v1', frames=32, do_sample=False, max_new_tokens=1024,
                attention=args.attn_implementation, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0))
    records = []
    with jsonl_path.open('x', encoding='utf-8', buffering=1) as stream:
        for i, row in enumerate(tqdm(selected, desc=f'AoTD NExT-QA {args.split}', unit='question'), 1):
            sample_start = time.monotonic()
            path = Path(args.video_dir) / f"{row['video']}.mp4"
            print(f"[eval {i}/{len(selected)}] decode video={row['video']} qid={row['qid']}", flush=True)
            video = get_video(path)
            conversation = [dict(role='user', content=[dict(type='text', text=question_prompt(row)), dict(type='video')])]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, videos=video, padding=True, return_tensors='pt').to(model.device, torch.bfloat16)
            length = inputs.input_ids.shape[-1]
            print(f'[eval {i}/{len(selected)}] input_tokens={length} video_shape={video.shape}', flush=True)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True,
                    stopping_criteria=StoppingCriteriaList([TokenProgress(length)]))
            torch.cuda.synchronize()
            raw = processor.batch_decode(generated[:, length:], skip_special_tokens=True)[0]
            pred = parse_letter(raw)
            record = dict(sample_id=f"{row['video']}_{row['qid']}", video_id=row['video'], qid=row['qid'],
                          split=args.split, category=row['type'], question=row['question'],
                          options=[row[f'a{k}'] for k in range(5)], gt=int(row['answer']), pred=pred,
                          invalid=pred is None, correct=pred == int(row['answer']), raw_output=raw,
                          video_sha256=sha256(path), input_tokens=length,
                          generated_tokens=generated.shape[-1]-length,
                          seconds=time.monotonic()-sample_start)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
            print(f"[eval {i}/{len(selected)}] raw={raw!r} pred={pred} gt={record['gt']} correct={record['correct']} time={record['seconds']:.2f}s", flush=True)
    total = len(records)
    summary = dict(total=total, correct=sum(x['correct'] for x in records),
                   accuracy=sum(x['correct'] for x in records)/total,
                   invalid=sum(x['invalid'] for x in records),
                   invalid_rate=sum(x['invalid'] for x in records)/total)
    for group in 'CTD':
        group_records = [x for x in records if x['category'].startswith(group)]
        summary[group] = dict(total=len(group_records), correct=sum(x['correct'] for x in group_records),
                             accuracy=sum(x['correct'] for x in group_records)/len(group_records) if group_records else None)
    metrics = dict(model_load_seconds=load_seconds, total_seconds=time.monotonic()-start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    with output.open('x', encoding='utf-8') as stream:
        json.dump(dict(meta=meta, summary=summary, metrics=metrics, res_list=records), stream, ensure_ascii=False, indent=2)
    print(f'[eval] COMPLETE {json.dumps(summary)} output={output}', flush=True)


if __name__ == '__main__':
    main()
