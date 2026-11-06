"""
Rebuild the option frames for the temporal frame retrieval task from your own copy
of Charades.

The dataset does not redistribute Charades videos or frames (the Charades license
does not allow it). Instead, every option in charades_mcq_before_after.json lists
"frame_indices": the source frame index of each option frame, in the order the
model sees them. This script decodes each Charades video sequentially and writes

    <output-dir>/<video_id>/<mcq_id>/<A|B|C|D>/frame_000.jpg, frame_001.jpg, ...

which is the layout run_qwen.py expects under Charades_v1/frames.

Usage (Charades_v1 = the folder with the Charades <id>.mp4 files, downloaded from
https://prior.allenai.org/projects/charades under the Charades license):

    python rebuild_charades_option_frames.py \
        --json-path  charades_mcq_before_after.json \
        --video-dir  Charades_v1 \
        --output-dir Charades_v1/frames

--compare-dir <released frames dir> compares every rebuilt frame with the
corresponding released frame and reports the pixel difference (authors only, for
checking the indices against the original release).
"""
import argparse
import json
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-path", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true", help="Rewrite option folders that already contain frames")
    p.add_argument("--compare-dir", help="Released frames directory to compare the rebuilt frames against")
    p.add_argument("--limit", type=int, help="Stop after this many videos (for quick test runs)")
    return p.parse_args()


def read_frames(video_path, wanted):
    """Decode the video sequentially and return {index: frame} for the wanted indices."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    out, idx, last = {}, 0, max(wanted)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            out[idx] = frame
        idx += 1
    cap.release()
    return out


def listed_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def main():
    args = parse_args()
    with open(args.json_path) as f:
        dataset = json.load(f)

    stats = {"videos": 0, "options_written": 0, "options_skipped": 0, "missing_video": 0,
             "missing_frames": 0, "no_indices": 0}
    diffs = []  # mean absolute pixel difference per compared frame
    worst = []

    for vid, mcqs in dataset.items():
        if not mcqs:
            continue
        if args.limit is not None and stats["videos"] >= args.limit:
            break
        wanted = set()
        for mcq in mcqs:
            for option in mcq["options"].values():
                wanted.update(option.get("frame_indices", []))
        if not wanted:
            stats["no_indices"] += 1
            continue
        frames = read_frames(os.path.join(args.video_dir, f"{vid}.mp4"), wanted)
        if frames is None:
            print(f"Warning: cannot open {vid}.mp4 in {args.video_dir}")
            stats["missing_video"] += 1
            continue
        stats["videos"] += 1

        for mcq in mcqs:
            for key, option in mcq["options"].items():
                indices = option.get("frame_indices", [])
                out_dir = os.path.join(args.output_dir, vid, mcq["id"], key)
                if listed_images(out_dir) and not args.overwrite:
                    stats["options_skipped"] += 1
                else:
                    os.makedirs(out_dir, exist_ok=True)
                    for name in listed_images(out_dir):
                        os.remove(os.path.join(out_dir, name))
                    for j, idx in enumerate(indices):
                        if idx not in frames:
                            print(f"Warning: frame {idx} missing from {vid}.mp4")
                            stats["missing_frames"] += 1
                            continue
                        cv2.imwrite(os.path.join(out_dir, f"frame_{j:03d}.jpg"), frames[idx])
                    stats["options_written"] += 1

                if args.compare_dir:
                    ref_dir = os.path.join(args.compare_dir, vid, mcq["id"], key)
                    refs, ours = listed_images(ref_dir), listed_images(out_dir)
                    if len(refs) != len(ours):
                        print(f"Count mismatch {vid}/{mcq['id']}/{key}: released {len(refs)}, rebuilt {len(ours)}")
                    for r, o in zip(refs, ours):
                        a = cv2.imread(os.path.join(ref_dir, r)).astype(np.int16)
                        b = cv2.imread(os.path.join(out_dir, o)).astype(np.int16)
                        d = float(np.abs(a - b).mean()) if a.shape == b.shape else float("inf")
                        diffs.append(d)
                        worst.append((d, f"{vid}/{mcq['id']}/{key}/{r}"))

        if stats["videos"] % 20 == 0:
            print(f"{stats['videos']} videos done... {stats}")

    print(f"Done: {stats}")
    if diffs:
        arr = np.array(diffs)
        print(f"Compared {len(arr)} frames: mean |diff| {arr.mean():.3f}, median {np.median(arr):.3f}, "
              f"max {arr.max():.3f}, exact (0) {int((arr == 0).sum())}, > 2.0: {int((arr > 2.0).sum())}")
        for d, name in sorted(worst, reverse=True)[:10]:
            print(f"  {d:8.3f}  {name}")


if __name__ == "__main__":
    main()
