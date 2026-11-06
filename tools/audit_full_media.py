"""Target-node full media checksum and first/last-frame decoding audit."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
from decord import VideoReader,cpu

root=Path(os.environ['AOTD_ROOT'])/'data/nextqa'
manifest=json.loads((root/'full_media_manifest.json').read_text())
assert manifest['complete'] and not manifest['errors']
files={x['video_id']:x for x in manifest['files']}
for split in manifest['splits']:
    rows=list(csv.DictReader((root/f'{split}.csv').open()))
    needed={x['video'] for x in rows}
    def audit(vid):
        info=files[vid];p=root/info['path']
        assert p.stat().st_size==info['bytes'],p
        with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256'],p
        vr=VideoReader(str(p),ctx=cpu(0),num_threads=1)
        length=len(vr);assert length>0,p
        frames=vr.get_batch([0,length-1]).asnumpy()
        assert len(frames.shape)==4 and frames.shape[0]==2 and frames.shape[-1]==3,p
        return dict(video_id=vid,frames=length,decoded_shape=list(frames.shape),sha256=info['sha256'])
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(audit,vid):vid for vid in sorted(needed)}
        for i,job in enumerate(as_completed(jobs),1):
            try:records.append(job.result())
            except Exception as exc:errors.append(dict(video_id=jobs[job],error=str(exc)))
            if i%25==0 or i==len(needed):print(f'[media audit {split}] {i}/{len(needed)} errors={len(errors)}',flush=True)
    report=dict(split=split,questions=len(rows),expected_videos=len(needed),verified_videos=len(records),errors=errors,records=records,status='PASS' if not errors else 'FAIL')
    (root/f'full_media_audit_{split}.json').write_text(json.dumps(report,indent=2))
    assert not errors,errors
    print(f'[media audit] PASS: {split} {len(rows)} questions / {len(records)} videos',flush=True)
