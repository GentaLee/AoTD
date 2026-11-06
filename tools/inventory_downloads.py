"""Record staged download artifacts without committing binaries or credentials."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--downloads',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
expected_ego=dict(bytes=3680916665,sha256='4341cfed79cfbe96f93ebd6fc2dff6219abea8a6df8d9e2b9c0800624a885831',
                  source='https://huggingface.co/datasets/ynhe/videochat2_data/resolve/eccfd0dc86fd0a03b5ca52bf65d0e9b11d7e9247/egoqa_split_videos.zip')
rows=[]
for path in sorted(args.downloads.rglob('*')):
    relative=path.relative_to(args.downloads)
    if any(part in {'.cache','teacher-access','__pycache__','token','credentials','auth.json','hosts.yml','.netrc'} for part in relative.parts):continue
    if path.suffix in {'.pem','.key'}:continue
    if not path.is_file() or path.name in {'inventory.json','README.md'}:continue
    size=path.stat().st_size
    row=dict(path=str(relative),bytes=size)
    if path.name=='egoqa_split_videos.zip':
        row['expected']=expected_ego
        if size!=expected_ego['bytes']:
            row['status']='incomplete_or_in_progress';rows.append(row);continue
    before=path.stat()
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):row['status']='changed_during_scan'
    else:
        row.update(sha256=digest,status='recorded')
        if path.name=='egoqa_split_videos.zip':row['status']='publisher_hash_verified' if digest==expected_ego['sha256'] else 'publisher_hash_mismatch'
    rows.append(row)
report=dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            files=rows,total_recorded_bytes=sum(row['bytes'] for row in rows),
            excluded=['credentials','HF token','HF access-check cache','model weights and media outside downloads'],
            binary_artifacts_committed=False)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(f'[downloads inventory] {len(rows)} files, {report["total_recorded_bytes"]} bytes; {args.output}',flush=True)
