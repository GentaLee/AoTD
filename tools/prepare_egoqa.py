"""Import all native VideoChat2 EgoQA train questions without changing answers."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from construction_io import atomic_json, records, sha256


def convert(input_path, video_dir, output, exclusions=None):
    input_path=Path(input_path)
    annotation_hash=sha256(input_path)
    exclusion_data={};excluded={};exclusions_hash=None
    if exclusions is not None:
        exclusions=Path(exclusions)
        exclusion_bytes=exclusions.read_bytes()
        exclusion_data=json.loads(exclusion_bytes)
        exclusions_hash=hashlib.sha256(exclusion_bytes).hexdigest()
        if exclusion_data.get('annotations_sha256')!=annotation_hash:
            raise ValueError('Audited exclusions do not match the source annotation SHA256')
        for item in exclusion_data['exclusions']:
            name=item['video']
            if name in excluded or not item.get('status') or not item.get('reason'):
                raise ValueError('Duplicate or incomplete audited exclusion entry')
            excluded[name]=item
        if exclusion_data.get('excluded_videos',len(excluded))!=len(excluded):
            raise ValueError('Audited exclusion count mismatch')
        if not isinstance(exclusion_data.get('annotated_records'),int) or not isinstance(exclusion_data.get('valid_videos'),int):
            raise ValueError('Audited exclusions require annotated_records and valid_videos')
        original_hash=exclusion_data.get('original_archive_sha256','')
        if len(original_hash)!=64 or any(c not in '0123456789abcdef' for c in original_hash):
            raise ValueError('Audited exclusions require the original archive SHA256')
    video_dir = Path(video_dir).resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(video_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    seen = set()
    videos = set()
    all_videos=set();excluded_qa=0;annotated_records=0
    count = 0
    with temporary.open('w') as stream:
        for index, row in enumerate(records(input_path)):
            relative = PurePosixPath(row['video'])
            if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.mp4':
                raise ValueError(f'Unsafe native video path at row {index}')
            all_videos.add(str(relative));annotated_records+=len(row['QA'])
            if str(relative) in excluded:
                excluded_qa+=len(row['QA'])
                continue
            media = video_dir.joinpath(*relative.parts)
            if not media.is_file():
                raise FileNotFoundError(media)
            videos.add(str(relative))
            for question_index, question in enumerate(row['QA']):
                prompt = question['q'].strip()
                answer = question['a']
                sid = f'EgoQA:{relative.with_suffix("")}:{question_index}'
                if sid in seen or not prompt or not answer.strip():
                    raise ValueError(f'Duplicate ID or empty QA: {sid}')
                seen.add(sid)
                result = dict(sample_id=sid, dataset='EgoQA', split='train',
                              source_annotation_row=index, source_video=str(relative),
                              source_question_index=question_index,
                              video_id=str(relative.with_suffix('')), video_path=str(media),
                              question=prompt, options=[], answer=answer)
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                count += 1
    if not count:
        raise ValueError('Empty EgoQA train annotations')
    if exclusions is not None:
        if set(excluded)-all_videos:raise ValueError('Audited exclusions reference videos absent from source annotations')
        if annotated_records!=exclusion_data['annotated_records']:
            raise ValueError('Source QA count does not match the audited annotation count')
        if len(videos)!=exclusion_data['valid_videos'] or len(excluded)!=len(all_videos)-len(videos):
            raise ValueError('Retained video count does not match the audited exclusions')
        if count!=annotated_records-excluded_qa:
            raise ValueError('Retained QA count does not match the audited exclusions')
    if sha256(input_path)!=annotation_hash:raise ValueError('Source annotations changed during import')
    if exclusions is not None and sha256(exclusions)!=exclusions_hash:
        raise ValueError('Audited exclusions changed during import')
    temporary.replace(output)
    atomic_json(output.with_suffix('.source_audit.json'), dict(
        dataset='EgoQA', split='train', records=count, videos=len(videos),
        video_dir=str(video_dir),
        missing_videos=0, media_ready=True,
        annotation_sha256=annotation_hash, output_sha256=sha256(output),
        annotated_records=annotated_records,excluded_videos=len(excluded),excluded_qa_records=excluded_qa,
        exclusions_manifest_sha256=exclusions_hash,
        original_archive_sha256=exclusion_data.get('original_archive_sha256'),
        sampling='all native train questions except explicitly audited media exclusions; original answer strings preserved'))
    print(f'[EgoQA] {count} QA, {len(videos)} videos, excluded={len(excluded)}, missing=0', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    default_exclusions=Path(__file__).resolve().parents[1]/'reproduction/metadata/egoqa_media_exclusions.json'
    parser.add_argument('--exclusions',type=Path,default=default_exclusions if default_exclusions.is_file() else None,
                        help='Explicit audited media exclusions, pinned to the native annotation SHA256')
    args = parser.parse_args()
    convert(args.input, args.video_dir, args.output,args.exclusions)


if __name__ == '__main__':
    main()
