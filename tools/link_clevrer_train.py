"""Reuse original CLEVRER train RGB files by scene ID, never third-party labels."""
from pathlib import Path
from construction_io import atomic_json,records

root=Path(__file__).resolve().parents[1]
qa=root/'data/raw/CLEVRER/train_normalized.jsonl'
view=root/'data/construction/media/CLEVRER';view.mkdir(parents=True,exist_ok=True)
original=Path('/volume/pt-test/users/zhaoye/Video-R1-data/CLEVRER/train_videos')
ids={row['video_id'] for row in records(qa)}
missing=[]
for vid in sorted(ids):
    source=original/(vid+'.mp4');link=view/(vid+'.mp4')
    if not link.exists() and not link.is_symlink() and source.is_file():link.symlink_to(source)
    if not link.is_file():missing.append(vid)
atomic_json(root/'data/construction/clevrer_missing_train_ids.json',missing)
print(f'[CLEVRER media] {len(ids)-len(missing)}/{len(ids)} selected train videos; missing={len(missing)}',flush=True)
