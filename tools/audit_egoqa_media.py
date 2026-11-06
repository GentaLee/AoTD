"""Audit every annotated EgoQA clip with the training decoder and frame indices."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def check(item):
    name, folder = item
    path = Path(folder) / name
    if not path.is_file():
        return dict(video=name, status='missing', reason='file_not_found')
    try:
        import numpy as np
        from decord import VideoReader, cpu
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        length = len(reader)
        if length <= 0:
            raise ValueError('No video frames')
        indices = np.linspace(0, length, 32, endpoint=False).astype(int)
        batch = reader.get_batch(indices)
        if batch.shape[0] != 32 or len(batch.shape) != 4 or batch.shape[-1] != 3:
            raise ValueError('Unexpected decoded frame shape')
        return dict(video=name, status='ok', frames=length)
    except Exception as exc:
        # The detailed deployment-specific decoder traceback stays in private logs.
        return dict(video=name, status='invalid', reason=type(exc).__name__, bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Use 1-8 bounded CPU workers')
    rows = json.loads(args.annotations.read_text())
    names = sorted({row['video'] for row in rows})
    start = last = time.monotonic()
    interval = max(10, min(60, float(os.environ.get('AOTD_PROGRESS_INTERVAL_SECONDS', '30'))))
    accepted = []
    excluded = []
    print(f'[media audit] checking all {len(names)} annotated clips with 32-frame decoding', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(check, ((name, str(args.video_dir)) for name in names), chunksize=4), 1):
            (accepted if result['status'] == 'ok' else excluded).append(result)
            now = time.monotonic()
            if now - last >= interval:
                print(f'[media audit] {index}/{len(names)} ({100*index/max(len(names),1):.1f}%) excluded={len(excluded)} elapsed={now-start:.1f}s', flush=True)
                last = now
    report = dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                  annotated_records=len(rows), annotated_videos=len(names), valid_videos=len(accepted),
                  missing_videos=sum(item['status'] == 'missing' for item in excluded),
                  invalid_videos=sum(item['status'] == 'invalid' for item in excluded),
                  exclusions=excluded, decoder='decord', frames_per_clip=32,
                  frame_indices='linspace(0, len(video), 32, endpoint=False)',
                  valid_video_names_sha256=hashlib.sha256('\n'.join(item['video'] for item in accepted).encode()).hexdigest(),
                  duration_seconds=round(time.monotonic()-start, 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(args.output)
    print(f'[media audit] complete: valid={len(accepted)} missing={report["missing_videos"]} invalid={report["invalid_videos"]}', flush=True)


if __name__ == '__main__':
    main()
