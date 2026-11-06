#!/usr/bin/env python3
"""Download only the real media needed for AoTD train/val smoke, on network-enabled source host.

All files are pinned to immutable revisions.  This is a preparation tool;
the offline worker runtime does not invoke it or need network access.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen


ANNOTATIONS = "2432e9724f88ed9f40010e2989f104570a91de4e"
VIDEOS = [
    ("4010069381", "VLM2Vec/nextqa-rawvideo", "b454fdb10e211887097207db8dd74fcfa7b486ab", "4010069381.mp4", "3cd0e54587695a2c11c674c4f7f94cfed29ce8f70fbce2fff54bab27997758aa"),
    ("4882821564", "VLM2Vec/nextqa-rawvideo", "b454fdb10e211887097207db8dd74fcfa7b486ab", "4882821564.mp4", "49dd433b47f9e3c88c272332d5ae739fec1d9b4d96f5b93b8e648d2b45428b41"),
    ("3238737531", "KerryMe/NExTVideo", "fcbc6dc5371d9426bf71da76bf65e5c707cc90ea", "NExTVideo/1164/3238737531.mp4", "0e943af2882a83d44eb4cdaa88094ffaae2a8056b1773753b9ceb7dea58373fb"),
    ("8968804598", "KerryMe/NExTVideo", "fcbc6dc5371d9426bf71da76bf65e5c707cc90ea", "NExTVideo/1006/8968804598.mp4", "48195e77d3d9360dd42ad816a610078fdd2e4fe3d8442b1e55da32890aa7f497"),
]


def fetch(url, path, expected=None):
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        partial.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")
    print(f"ready {path.name}: {path.stat().st_size:,} bytes SHA256={digest}", flush=True)
    return {"path": str(path), "url": url, "sha256": digest, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/nextqa")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    probe=root
    while not probe.exists():probe=probe.parent
    filesystem=subprocess.check_output(['findmnt','-T',str(probe),'-n','-o','FSTYPE'],text=True).strip()
    if filesystem in {'overlay','tmpfs','ramfs',''}:
        parser.error('output must be on persistent mounted storage')
    root.mkdir(parents=True, exist_ok=True)
    (root / "NExTVideo").mkdir(exist_ok=True)
    files = []
    for name in ("train.csv", "val.csv", "test.csv", "map_vid_vidorID.json"):
        url = f"https://raw.githubusercontent.com/doc-doc/NExT-QA/{ANNOTATIONS}/dataset/nextqa/{name}"
        files.append(fetch(url, root / name))
    for video_id, repo, revision, source_path, digest in VIDEOS:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{source_path}"
        files.append(fetch(url, root / "NExTVideo" / f"{video_id}.mp4", digest))
    (root / "download_manifest.json").write_text(json.dumps({"annotation_revision": ANNOTATIONS, "files": files}, indent=2) + "\n")


if __name__ == "__main__":
    main()
