"""Materialize real STAR train windows; share identical clips across question IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import subprocess
from construction_io import atomic_json,records


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('Use at most eight bounded ffmpeg workers')
    root=Path(__file__).resolve().parents[1]
    raw=root/'data/construction/media/Charades';raw.mkdir(parents=True,exist_ok=True)
    # Existing files are reused as RGB media only; no Video-R1 QA/CoT is read.
    existing=Path('/volume/pt-test/users/zhaoye/Video-R1-data/STAR')
    if existing.exists():
        for path in existing.glob('*.mp4'):
            link=raw/path.name
            if not link.exists() and not link.is_symlink():link.symlink_to(path)
    qa=root/'data/construction/STAR/qa_normalized.jsonl'
    view=root/'data/construction/media/STAR';view.mkdir(parents=True,exist_ok=True)
    clips=root/'data/construction/clips/STAR';clips.mkdir(parents=True,exist_ok=True)
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    windows={};missing=set();questions=0
    for row in records(qa):
        source=raw/(row['video_id']+'.mp4')
        questions+=1
        if not source.is_file():missing.add(row['video_id']);continue
        start=row['start_seconds'];end=row['end_seconds']
        if not 0<=start<end:raise ValueError('Invalid STAR window')
        key=f"{source.resolve()}:{source.stat().st_size}:{start:.6f}:{end:.6f}"
        target=clips/(hashlib.sha256(key.encode()).hexdigest()+'.mp4')
        windows.setdefault(key,dict(source=source,target=target,start=start,end=end,links=[]))['links'].append(Path(row['video_path']))
    def one(job):
        path=job['target']
        if not path.is_file():
            temporary=path.with_suffix('.partial.mp4')
            completed=subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-ss',str(job['start']),'-i',str(job['source']),
                '-t',str(job['end']-job['start']),'-an','-c:v','libx264','-preset','ultrafast','-crf','23','-threads','1',str(temporary)],
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=120)
            if completed.returncode:raise RuntimeError(completed.stderr[-1500:])
            if temporary.stat().st_size<1024:raise ValueError('Empty output video')
            temporary.replace(path)
        for link in job['links']:
            if not link.exists() or link.is_symlink():
                temporary_link=link.with_suffix('.link.partial')
                if temporary_link.is_symlink():temporary_link.unlink()
                temporary_link.symlink_to(path);temporary_link.replace(link)
        return len(job['links'])
    done=0;ready=0;failed=[]
    # Bound pending jobs instead of queueing one future for every frame/window.
    iterator=iter(windows.values())
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=set()
        while True:
            while len(pending)<a.workers*2:
                job=next(iterator,None)
                if job is None:break
                pending.add(pool.submit(one,job))
            if not pending:break
            finished,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                try:ready+=future.result()
                except Exception as e:failed.append(str(e))
                done+=1
            if done%100<a.workers or not pending:
                print(f'[STAR clip progress] {done}/{len(windows)} windows; {ready}/{questions} QA media ready',flush=True)
    report=dict(train_questions=questions,available_windows=len(windows),ready_questions=ready,
                missing_source_videos=sorted(missing),errors=failed,complete=not missing and not failed,
                method='decode/re-encode exact annotated start/end windows; 480p source; shared clips for identical windows')
    atomic_json(root/'data/construction/STAR/clips_audit.json',report)
    print(f'[STAR clips] {ready}/{questions} QA ready; {len(missing)} raw videos missing; errors={len(failed)}',flush=True)
    if failed:raise SystemExit(3)


if __name__=='__main__':main()
