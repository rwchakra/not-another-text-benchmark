"""
Recover the exact Charades frame index behind every released option frame.

Why: the Charades license does not allow redistributing Charades data, so the
dataset can no longer ship the option frames
(Charades_v1/frames/<id>/<mcq_id>/{A,B,C,D}/frame_XXX.jpg). Users rebuild them from
their own Charades download instead (rebuild_charades_option_frames.py). The
released frames cannot be regenerated from the spans alone, because the scrambled
options used an unrecorded shuffle and 530 folders contain decoder defects (see
make_charades_option_frames.py). This script records, for every released frame,
which frame of the source video it is, so the rebuild reproduces the released
frames exactly.

How: each source video is decoded once, sequentially, into small grayscale
thumbnails. Each released frame is compared (mean squared error) with every
thumbnail. When a frame in the option's expected sample (the indices
make_charades_option_frames.py would sample from the span) is about as close as the
best match anywhere in the video, that expected index is used; otherwise the best
match anywhere in the video is used (this is what catches the stale frames).

Output: a copy of the MCQ JSON in which every option gets "frame_indices", the
source frame index of each released frame in presentation order (sorted file
name order, which is the order run_qwen.py feeds them to the model), and a CSV
report with the match error of every frame.

Run it once, on a copy of the v1 release that still contains the frames:

    python recover_charades_frame_indices.py \
        --json-path  <v1>/charades_mcq_before_after.json \
        --video-dir  <v1>/Charades_v1 \
        --frames-dir <v1>/Charades_v1/frames \
        --out-json   charades_mcq_before_after_v2.json \
        --report     frame_index_report.csv
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

THUMB_SIZE = (80, 60)  # (width, height)
# An expected-sample index is preferred over the global best match when its error
# is within this factor (plus a small absolute slack) of the global best.
PREFER_FACTOR = 1.10
PREFER_SLACK = 0.1
# Matches whose error is above this (on a 0-255 scale) are flagged in the report.
SUSPICIOUS_MSE = 60.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-path", required=True, help="v1 charades_mcq_before_after.json")
    p.add_argument("--video-dir", required=True, help="Directory with the Charades <id>.mp4 files")
    p.add_argument("--frames-dir", required=True, help="v1 Charades_v1/frames directory")
    p.add_argument("--out-json", required=True, help="Where to write the JSON with frame_indices")
    p.add_argument("--report", required=True, help="Where to write the per-frame CSV report")
    p.add_argument("--frames-per-span", type=int, default=8)
    p.add_argument("--limit", type=int, help="Stop after this many videos (for quick test runs)")
    return p.parse_args()


def thumb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def decode_thumbs(video_path):
    """All frames of the video as thumbnails, decoded sequentially, plus the
    container's fps and frame count (which make_charades_option_frames.py used)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thumbs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        thumbs.append(thumb(frame))
    cap.release()
    if not thumbs:
        return None, fps, total
    return np.stack(thumbs), fps, total


def expected_indices(fps, total_frames, span, n_frames):
    """Same sampling as make_charades_option_frames.sample_indices."""
    if fps <= 0 or total_frames <= 0:
        return []
    start = max(0.0, span[0])
    end = min(total_frames / fps, span[1])
    if end <= start:
        return []
    return [int(t * fps) for t in np.linspace(start, end, n_frames)]


def released_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def match(frame_thumb, thumbs, candidates):
    errs = ((thumbs - frame_thumb) ** 2).mean(axis=(1, 2))
    best = int(errs.argmin())
    best_err = float(errs[best])
    valid = [c for c in candidates if 0 <= c < len(thumbs)]
    if valid:
        cand = min(valid, key=lambda c: errs[c])
        if errs[cand] <= best_err * PREFER_FACTOR + PREFER_SLACK:
            return cand, float(errs[cand]), best, best_err, True
    return best, best_err, best, best_err, False


def main():
    args = parse_args()
    with open(args.json_path) as f:
        dataset = json.load(f)

    rows = []
    stats = {"videos": 0, "options": 0, "frames": 0, "expected": 0, "fallback": 0,
             "suspicious": 0, "missing_video": 0, "missing_frames": 0}

    for n, (vid, mcqs) in enumerate(dataset.items()):
        if not mcqs:
            continue
        if args.limit is not None and stats["videos"] >= args.limit:
            break
        thumbs, fps, total = decode_thumbs(os.path.join(args.video_dir, f"{vid}.mp4"))
        if thumbs is None:
            print(f"Warning: cannot decode {vid}.mp4")
            stats["missing_video"] += 1
            continue
        stats["videos"] += 1

        for mcq in mcqs:
            mcq_id = mcq["id"]
            for key, option in mcq["options"].items():
                folder = os.path.join(args.frames_dir, vid, mcq_id, key)
                files = released_files(folder)
                if not files:
                    print(f"Warning: no released frames in {folder}")
                    stats["missing_frames"] += 1
                    continue
                candidates = expected_indices(fps, total, option["span"], args.frames_per_span)
                indices = []
                for name in files:
                    img = cv2.imread(os.path.join(folder, name))
                    idx, err, gbest, gerr, from_expected = match(thumb(img), thumbs, candidates)
                    indices.append(idx)
                    stats["frames"] += 1
                    stats["expected" if from_expected else "fallback"] += 1
                    suspicious = err > SUSPICIOUS_MSE
                    stats["suspicious"] += suspicious
                    rows.append([vid, mcq_id, key, name, idx, f"{err:.2f}", gbest, f"{gerr:.2f}",
                                 int(from_expected), int(suspicious)])
                option["frame_indices"] = indices
                stats["options"] += 1

        if stats["videos"] % 20 == 0:
            print(f"{stats['videos']} videos done... {stats}")

    with open(args.out_json, "w") as f:
        json.dump(dataset, f, indent=2)
    with open(args.report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "mcq_id", "option", "file", "frame_index", "mse",
                    "global_best_index", "global_best_mse", "from_expected_sample", "suspicious"])
        w.writerows(rows)

    print(f"Done: {stats}")
    print(f"Wrote {args.out_json} and {args.report}")


if __name__ == "__main__":
    main()
