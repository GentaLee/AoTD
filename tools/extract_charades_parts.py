"""GPU-node CPU job: verify and read the official ZIP across bounded staged chunks."""
import bisect
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
from construction_io import atomic_json


class PartsFile(io.RawIOBase):
    def __init__(self,paths):
        self.paths=paths;self.starts=[0];self.position=0
        for path in paths:self.starts.append(self.starts[-1]+path.stat().st_size)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.starts[-1]+offset
        if self.position<0:raise ValueError('Negative seek')
        return self.position
    def read(self,size=-1):
        left=self.starts[-1]-self.position if size<0 else min(size,self.starts[-1]-self.position)
        blocks=[]
        while left>0:
            index=bisect.bisect_right(self.starts,self.position)-1
            wanted=min(left,self.starts[index+1]-self.position)
            with self.paths[index].open('rb') as stream:
                stream.seek(self.position-self.starts[index]);block=stream.read(wanted)
            if len(block)!=wanted:raise ValueError('Short read from staged chunk')
            blocks.append(block);self.position+=wanted;left-=wanted
        return b''.join(blocks)


def main():
    root=Path(__file__).resolve().parents[1]
    raw=root/'data/raw/Charades'
    manifest=json.loads((raw/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('All ZIP chunks must be staged before extraction')
    paths=[]
    for row in manifest['files']:
        path=raw/'parts'/row['part']
        if path.stat().st_size!=row['bytes']:raise ValueError(f'Chunk size mismatch: {path}')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=row['sha256']:raise ValueError(f'Transfer SHA256 mismatch: {path}')
        paths.append(path)
    destination=raw/'Charades_v1_480';destination.mkdir(exist_ok=True)
    verified=[]
    with zipfile.ZipFile(PartsFile(paths)) as archive:
        videos=[entry for entry in archive.infolist() if entry.filename.endswith('.mp4')]
        for index,entry in enumerate(videos,1):
            path=destination/Path(entry.filename).name
            if not path.is_file() or path.stat().st_size!=entry.file_size:
                temporary=path.with_suffix('.mp4.partial')
                with archive.open(entry) as source,temporary.open('wb') as output:
                    shutil.copyfileobj(source,output,1024**2)
                temporary.replace(path)
            verified.append(dict(video_id=path.stem,bytes=entry.file_size,zip_crc32=entry.CRC))
            if index%500==0:print(f'[Charades extraction] {index}/{len(videos)} videos',flush=True)
    # Repoint only our generated media view to the complete official archive.
    view=root/'data/construction/media/Charades';view.mkdir(parents=True,exist_ok=True)
    for path in destination.glob('*.mp4'):
        link=view/path.name
        if not link.exists() or link.is_symlink():
            temporary=link.with_suffix('.link.partial')
            if temporary.is_symlink():temporary.unlink()
            temporary.symlink_to(path);temporary.replace(link)
    atomic_json(raw/'extraction_audit.json',dict(complete=True,videos=verified,source=manifest['url'],
                chunk_sha256_verified=True,zip_crc_checked_for_new_extractions=True))
    print(f'[Charades] complete official RGB archive: {len(verified)} videos',flush=True)


if __name__=='__main__':main()
