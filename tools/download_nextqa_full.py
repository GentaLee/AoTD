"""network-enabled source host-only preparation: download every video in selected official splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
REPO='VLM2Vec/nextqa-rawvideo'
REV='b454fdb10e211887097207db8dd74fcfa7b486ab'

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--split',nargs='+',choices=['train','val','test'],default=['val'])
    parser.add_argument('--ids-file', type=Path, help='JSON array of video IDs, e.g. a target-node missing-media audit')
    args=parser.parse_args()
    data=ROOT/'data/nextqa';videos=data/'NExTVideo';videos.mkdir(exist_ok=True)
    ids=set()
    for split in args.split:
        with (data/f'{split}.csv').open() as f:ids.update(x['video'] for x in csv.DictReader(f))
    if args.ids_file:
        wanted=set(json.loads(args.ids_file.read_text()))
        if not wanted <= ids: raise ValueError('Requested IDs are outside the selected official split')
        ids=wanted
    manifest_name='full_media_manifest.json' if args.split==['val'] and not args.ids_file else 'media_manifest_'+'_'.join(args.split)+('_missing' if args.ids_file else '')+'.json'
    info=HfApi().dataset_info(REPO,revision=REV,files_metadata=True)
    assert info.sha==REV
    metadata={x.rfilename:(REPO,REV,x) for x in info.siblings}
    missing=[x for x in ids if f'{x}.mp4' not in metadata]
    if missing:
        backup_repo='KerryMe/NExTVideo';backup_rev='fcbc6dc5371d9426bf71da76bf65e5c707cc90ea'
        backup=HfApi().dataset_info(backup_repo,revision=backup_rev,files_metadata=True)
        assert backup.sha==backup_rev
        for item in backup.siblings:
            if Path(item.rfilename).stem in missing:
                metadata[Path(item.rfilename).name]=(backup_repo,backup_rev,item)
    missing=[x for x in ids if f'{x}.mp4' not in metadata]
    if missing:raise RuntimeError(f'Both mirrors are missing video IDs: {missing}')
    def one(vid):
        repo,rev,item=metadata[f'{vid}.mp4'];path=videos/f'{vid}.mp4'
        expected=item.lfs.sha256 if item.lfs else None
        url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{item.rfilename}'
        for attempt in range(4):
            try:
                if not path.is_file() or path.stat().st_size!=item.size:
                    tmp=path.with_suffix('.mp4.partial')
                    with urlopen(url,timeout=120) as response,tmp.open('wb') as stream:
                        for block in iter(lambda:response.read(1024*1024),b''):stream.write(block)
                    tmp.replace(path)
                with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                if expected and digest!=expected:raise ValueError(f'Publisher SHA256 mismatch: {vid}')
                assert path.stat().st_size==item.size
                return dict(video_id=vid,path=str(path.relative_to(data)),sha256=digest,bytes=item.size,url=url)
            except Exception:
                if attempt==3:raise
                time.sleep(2*(attempt+1))
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(one,vid):vid for vid in sorted(ids)}
        for i,f in enumerate(as_completed(jobs),1):
            try:
                result=f.result();results.append(result)
                print(f'[full media {i}/{len(ids)}] {result["video_id"]} verified {result["bytes"]} bytes',flush=True)
            except Exception as exc:
                errors.append(dict(video_id=jobs[f],error=str(exc)));print(f'[ERROR] {jobs[f]} {exc}',flush=True)
            (data/manifest_name).write_text(json.dumps(dict(repo=REPO,revision=REV,splits=args.split,expected=len(ids),files=results,errors=errors,complete=len(results)==len(ids)),indent=2))
    if errors:raise RuntimeError(f'{len(errors)} failed media downloads')
    print(f'[full media] COMPLETE: {len(results)} verified videos',flush=True)

if __name__=='__main__':main()
