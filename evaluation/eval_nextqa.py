"""AoTD fixed-commit NExT-QA protocol, with deterministic auditable output."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from decord import VideoReader, cpu
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from evaluation_checkpoint import fingerprint, prepare, atomic_json, summarize

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
    """Time-based liveness for a long generation; never print every token."""
    def __init__(self, input_length, interval=None, shared_clock=None, label=''):
        self.input_length = input_length
        self.start = time.monotonic()
        configured = float(os.environ.get('AOTD_PROGRESS_INTERVAL_SECONDS', '30')) if interval is None else interval
        if not math.isfinite(configured):
            raise ValueError('AOTD_PROGRESS_INTERVAL_SECONDS must be finite')
        self.interval = max(10.0, min(60.0, configured))
        self.clock = shared_clock if shared_clock is not None else [self.start]
        self.label = label

    def __call__(self, input_ids, scores, **kwargs):
        now = time.monotonic()
        if now - self.clock[0] >= self.interval:
            n = input_ids.shape[1] - self.input_length
            print(f'[eval generation] {self.label} tokens={n}/1024 '
                  f'elapsed={now-self.start:.1f}s', flush=True)
            self.clock[0] = now
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
    interval = float(os.environ.get('AOTD_PROGRESS_INTERVAL_SECONDS', '30'))
    if not math.isfinite(interval):
        raise ValueError('AOTD_PROGRESS_INTERVAL_SECONDS must be finite')
    interval = max(10.0, min(60.0, interval))
    verbose = os.environ.get('AOTD_VERBOSE', '0') == '1'
    if Path(args.data_dir).stem != args.split:
        raise ValueError('CSV filename must match --split')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    output = Path(args.save_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output.with_suffix('.jsonl')
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
    media_hashes = {}
    def verify_record(record, row):
        sid=f"{row['video']}_{row['qid']}"
        if (record.get('sample_id')!=sid or record.get('gt')!=int(row['answer']) or
            record.get('question')!=row['question'] or record.get('options')!=[row[f'a{k}'] for k in range(5)]):
            raise ValueError('Evaluation journal is not the exact source-question prefix')
        if (record.get('video_id')!=row['video'] or record.get('qid')!=row['qid'] or
            record.get('split')!=args.split or record.get('category')!=row['type']):
            raise ValueError('Evaluation journal source identity/category changed')
        path=Path(args.video_dir)/f"{row['video']}.mp4"
        if str(path) not in media_hashes:media_hashes[str(path)]=sha256(path)
        if record.get('video_sha256')!=media_hashes[str(path)]:
            raise ValueError('Previously evaluated video changed')
        prediction=parse_letter(record.get('raw_output',''))
        if (record.get('pred')!=prediction or record.get('correct')!=(prediction==int(row['answer'])) or
            record.get('invalid')!=(prediction is None)):
            raise ValueError('Evaluation prediction fields are inconsistent')
    print('[eval] checking durable prefix and model/input fingerprint',flush=True)
    identity=fingerprint(args.model_path,args.data_dir,args.video_dir,__file__,args.split,args.limit,args.attn_implementation)
    records, completed=prepare(output,identity,selected,verify_record)
    if completed is not None:
        print(f'[eval] COMPLETE already verified: {len(records)} questions; no GPU load',flush=True)
        return
    resumed=len(records)
    if resumed:print(f'[eval] resuming after {resumed}/{len(selected)} durable questions',flush=True)
    if resumed==len(selected):
        meta=dict(split=args.split,dataset_size=len(rows),limit=args.limit,full_dataset=len(selected)==len(rows),
                  model_path=args.model_path,csv_sha256=sha256(args.data_dir),script_sha256=sha256(__file__),
                  parsing='strict_letter_v1',frames=32,do_sample=False,max_new_tokens=1024,
                  attention=args.attn_implementation,resumed_records=resumed,recovered_complete_prefix=True)
        atomic_json(output,dict(meta=meta,summary=summarize(records),metrics=dict(
            model_load_seconds=0,total_seconds=0,recorded_inference_seconds=sum(item['seconds'] for item in records)),res_list=records))
        print(f'[eval] COMPLETE recovered all {resumed} durable questions; no GPU load',flush=True)
        return
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
                gpu_count=torch.cuda.device_count(), progress_interval_seconds=interval,resumed_records=resumed,
                resume_fingerprint_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
    eval_started = time.monotonic()
    progress_clock = [eval_started]
    with jsonl_path.open('a', encoding='utf-8', buffering=1) as stream:
        for i, row in enumerate(selected[resumed:], resumed+1):
            sample_start = time.monotonic()
            path = Path(args.video_dir) / f"{row['video']}.mp4"
            if verbose:
                print(f"[eval {i}/{len(selected)}] decode video={row['video']} qid={row['qid']}", flush=True)
            video = get_video(path)
            conversation = [dict(role='user', content=[dict(type='text', text=question_prompt(row)), dict(type='video')])]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, videos=video, padding=True, return_tensors='pt').to(model.device, torch.bfloat16)
            length = inputs.input_ids.shape[-1]
            if verbose:
                print(f'[eval {i}/{len(selected)}] input_tokens={length} video_shape={video.shape}', flush=True)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True,
                    stopping_criteria=StoppingCriteriaList([TokenProgress(length, interval, progress_clock, f'question={i}/{len(selected)}')]))
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
            stream.flush();os.fsync(stream.fileno())
            records.append(record)
            if verbose:
                print(f"[eval {i}/{len(selected)}] raw={raw!r} pred={pred} gt={record['gt']} correct={record['correct']} time={record['seconds']:.2f}s", flush=True)
            now = time.monotonic()
            if now-progress_clock[0] >= interval or i == len(selected):
                elapsed = now-eval_started
                eta = elapsed/max(i-resumed,1)*(len(selected)-i)
                print(f'[eval progress] questions={i}/{len(selected)} percent={100*i/len(selected):.1f}% '
                      f'elapsed={elapsed:.1f}s estimated_remaining={eta:.1f}s', flush=True)
                progress_clock[0] = now
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
    atomic_json(output,dict(meta=meta, summary=summary, metrics=metrics, res_list=records))
    print(f'[eval] COMPLETE {json.dumps(summary)} output={output}', flush=True)


if __name__ == '__main__':
    main()
