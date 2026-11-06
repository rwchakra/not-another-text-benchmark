"""
Rebuild the causal memory distortion benchmark (videos and contact frames) from
your own copy of Charades.

The dataset does not redistribute Charades videos or frames (the Charades license
does not allow it). It ships the cut points (benchmark.json, from the annotation
tool) and the position of every contact frame (causal_frame_positions.json). This
script reads the original Charades videos and writes

    <output-dir>/videos/<id>_{with_implication,without_implication,scrambled}_{complete,incomplete}.mp4
    <output-dir>/frames/<id>_contact.jpg

the same files and logic as causal_benchmark_builder.py and
regenerate_complete_videos.py used for the release. The 20 source videos are
original-resolution Charades videos (not the 480p version).

Usage (Charades_v1 = folder with the original Charades <id>.mp4 files, downloaded
from https://prior.allenai.org/projects/charades under the Charades license):

    python rebuild_causal_benchmark.py \
        --benchmark-json  benchmark.json \
        --positions-json  causal_frame_positions.json \
        --source-dir      Charades_v1 \
        --output-dir      causal

Then pass --output-dir (with causal_ground_truth.json copied into it) to
run_qwen.py --dataset causal --dataset_path.

--compare-dir <released causal dir> compares the rebuilt files with the released
ones (authors only, to check the rebuild against the original release).
"""
import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-json", required=True)
    p.add_argument("--positions-json", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--compare-dir", help="Released causal dir (with videos/ and frames/) to compare against")
    return p.parse_args()


def read_from(cap, start, count):
    """Seek to start and read up to count frames sequentially (as the builder did)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def reversed_segments(frames, n_segments):
    size = len(frames) // n_segments
    parts = [frames[i * size:(len(frames) if i == n_segments - 1 else (i + 1) * size)] for i in range(n_segments)]
    return [f for part in reversed(parts) for f in part]


def write_video(path, frames, fps, width, height):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()


def decode_all(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def mean_abs_diff(a, b):
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main():
    args = parse_args()
    with open(args.benchmark_json) as f:
        bench = json.load(f)
    with open(args.positions_json) as f:
        positions = json.load(f)
    n_segments = bench.get("methodology", {}).get("scrambled_segments", 4)
    videos_dir = os.path.join(args.output_dir, "videos")
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    missing, report = [], []
    for item in bench["test_items"]:
        vid = item["video_id"]
        src = os.path.join(args.source_dir, f"{vid}.mp4")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Missing source video: {src}")
            missing.append(vid)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        seg = item["segments"]
        v0, v1 = seg["video_range"]["start"], seg["video_range"]["end"]
        c0, c1 = seg["contact_range"]["start"], seg["contact_range"]["end"]
        i0, i1 = seg["irrelevant"]["start"], seg["irrelevant"]["end"]
        irrelevant = read_from(cap, i0, i1 - i0 + 1)
        rng = read_from(cap, v0, v1 - v0 + 1)  # rng[k] is source frame v0 + k
        cap.release()
        idx = list(range(v0, v0 + len(rng)))
        no_contact = [f for f, i in zip(rng, idx) if not (c0 <= i <= c1)]

        outputs = {
            "with_implication_complete": rng,
            "with_implication_incomplete": no_contact,
            "without_implication_complete": [f for f, i in zip(rng, idx) if i <= c1] + irrelevant,
            "without_implication_incomplete": [f for f, i in zip(rng, idx) if i < c0] + irrelevant,
            "scrambled_complete": reversed_segments(rng, n_segments),
            "scrambled_incomplete": reversed_segments(no_contact, n_segments),
        }
        for name, frames in outputs.items():
            cond, kind = name.rsplit("_", 1)
            expected = item["videos"][cond][kind]["frames_written"]
            if len(frames) != expected:
                print(f"Warning {vid} {name}: {len(frames)} frames, release had {expected}")
            write_video(os.path.join(videos_dir, f"{vid}_{name}.mp4"), frames, fps, width, height)

        segments = {"video_range": rng, "irrelevant": irrelevant}
        for fname, pos in positions.get(vid, {}).items():
            frame = segments[pos["segment"]][pos["offset"]]
            cv2.imwrite(os.path.join(frames_dir, fname), frame)

        if args.compare_dir:
            for name in outputs:
                ours = decode_all(os.path.join(videos_dir, f"{vid}_{name}.mp4"))
                ref = decode_all(os.path.join(args.compare_dir, "videos", f"{vid}_{name}.mp4"))
                diffs = [mean_abs_diff(a, b) for a, b in zip(ours, ref)]
                report.append(("video", f"{vid}_{name}", len(ours), len(ref), max(diffs) if diffs else float("inf")))
            for fname in positions.get(vid, {}):
                a = cv2.imread(os.path.join(frames_dir, fname))
                b = cv2.imread(os.path.join(args.compare_dir, "frames", fname))
                report.append(("frame", fname, 1, 1, mean_abs_diff(a, b) if b is not None else float("inf")))
        print(f"Rebuilt {vid}")

    print(f"Done. Missing source videos: {missing or 'none'}")
    if report:
        bad = [r for r in report if r[2] != r[3] or r[4] > 2.0]
        print(f"Compared {len(report)} files; {len(bad)} differ (frame count mismatch or max mean |diff| > 2.0)")
        for kind, name, n_ours, n_ref, d in sorted(report, key=lambda r: -r[4])[:15]:
            print(f"  {kind:5s} {name:50s} frames {n_ours}/{n_ref}  max mean|diff| {d:.3f}")


if __name__ == "__main__":
    main()
