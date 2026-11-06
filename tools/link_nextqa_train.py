"""Reuse only matching original NExT-QA RGB videos; never import third-party QA/CoT."""
import csv
from pathlib import Path
from construction_io import atomic_json

root=Path(__file__).resolve().parents[1]
view=root/'data/construction/media/NExT-QA'
view.mkdir(parents=True,exist_ok=True)
ids={row['video'] for row in csv.DictReader((root/'data/nextqa/train.csv').open())}
lookup={}
for folder in [root/'data/nextqa/NExTVideo',Path('/volume/pt-test/users/zhaoye/Video-R1-data/NeXT-QA/NextQA/NExTVideo')]:
    if folder.exists():
        for path in folder.rglob('*.mp4'):
            if path.stem in ids: lookup.setdefault(path.stem,path)
for vid,path in lookup.items():
    link=view/f'{vid}.mp4'
    if not link.exists() and not link.is_symlink(): link.symlink_to(path)
missing=sorted(vid for vid in ids if not (view/f'{vid}.mp4').is_file())
atomic_json(root/'data/construction/nextqa_missing_train_ids.json',missing)
print(f'[train media] {len(ids)-len(missing)}/{len(ids)} video IDs available; missing={missing}',flush=True)
