"""network-enabled source host network staging of official Charades 480p, resumable bounded chunks."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import fcntl
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
URL='https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip'
directory=ROOT/'data/raw/Charades/parts';directory.mkdir(parents=True,exist_ok=True)
lock=(directory.parent/'download.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
r=requests.head(URL,timeout=30);r.raise_for_status();size=int(r.headers['Content-Length'])
piece=1024**3
jobs=[(index,start,min(start+piece,size)-1) for index,start in enumerate(range(0,size,piece))]

def one(job):
    index,start,end=job
    path=directory/f'part_{index:03d}'
    wanted=end-start+1
    if not path.exists() or path.stat().st_size!=wanted:
        temporary=path.with_suffix('.partial')
        offset=temporary.stat().st_size if temporary.exists() else 0
        if offset>wanted:raise ValueError('Oversized partial chunk')
        for attempt in range(4):
            try:
                offset=temporary.stat().st_size if temporary.exists() else 0
                if offset==wanted:break
                with requests.get(URL,headers={'Range':f'bytes={start+offset}-{end}'},stream=True,timeout=(20,120)) as response:
                    response.raise_for_status()
                    if response.status_code!=206:raise ValueError('Server ignored byte range')
                    if not response.headers.get('Content-Range','').startswith(f'bytes {start+offset}-{end}/'):
                        raise ValueError('Wrong response range')
                    with temporary.open('ab') as stream:
                        for block in response.iter_content(1024**2):stream.write(block)
                if temporary.stat().st_size!=wanted:raise ValueError('Incomplete chunk')
                break
            except Exception:
                if attempt==3:raise
                time.sleep(2*(attempt+1))
        temporary.replace(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    print(f'[Charades verified] chunk {index+1}/{len(jobs)} bytes={wanted}',flush=True)
    return dict(part=path.name,start=start,end=end,bytes=wanted,sha256=digest)

files=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(one,job) for job in jobs]):
        files.append(future.result())
        manifest=dict(url=URL,bytes=size,etag=r.headers.get('ETag'),files=sorted(files,key=lambda x:x['start']),complete=len(files)==len(jobs))
        path=directory.parent/'manifest.json';temporary=path.with_suffix('.partial')
        temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(path)
        print(f'[Charades progress] {sum(x["bytes"] for x in files)/size:.1%}',flush=True)
