"""
Recover where every released causal-task contact frame comes from, so the frames
can be rebuilt from Charades instead of being redistributed.

Only the contact frames are recovered: they are the only probe frames used in the
evaluation (run_qwen.py and the paper). The v1 release also contained filler and
lure frames, which were not used; the lures had an unrecorded photometric edit and
cannot be rebuilt exactly.

Each contact frame frames/<id>_contact.jpg is matched (mean squared error on small
grayscale thumbnails) against
  - the released <id>_with_implication_complete.mp4, which is exactly the source
    frames video_range.start .. video_range.end, and
  - the tail of <id>_without_implication_complete.mp4, which is exactly the source
    frames irrelevant.start .. irrelevant.end.
The best match is stored as {"segment": "video_range" | "irrelevant", "offset": k},
meaning the k-th frame read after seeking to that segment's start, which is how
rebuild_causal_benchmark.py (and the original builder) read the source video.

Run once, by the authors, on a copy of the v1 release:

    python recover_causal_frame_positions.py \
        --benchmark-json <v1>/causal_benchmark/benchmark/annotations/benchmark.json \
        --videos-dir     <v1>/causal_benchmark/benchmark/videos \
        --frames-dir     <v1>/causal_benchmark/benchmark/frames \
        --out-json       causal_frame_positions.json
"""
import argparse
import json
import os

import cv2
import numpy as np

THUMB_SIZE = (80, 60)
SUSPICIOUS_MSE = 60.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-json", required=True)
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def thumb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def decode_thumbs(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(thumb(frame))
    cap.release()
    return out


def main():
    args = parse_args()
    with open(args.benchmark_json) as f:
        bench = json.load(f)

    result, flagged, total = {}, [], 0
    for item in bench["test_items"]:
        vid = item["video_id"]
        seg = item["segments"]
        with_impl = decode_thumbs(os.path.join(args.videos_dir, f"{vid}_with_implication_complete.mp4"))
        without_impl = decode_thumbs(os.path.join(args.videos_dir, f"{vid}_without_implication_complete.mp4"))
        n_pre = seg["contact_range"]["end"] - seg["video_range"]["start"] + 1
        irrelevant = without_impl[n_pre:]

        expected_r = seg["video_range"]["end"] - seg["video_range"]["start"] + 1
        expected_i = seg["irrelevant"]["end"] - seg["irrelevant"]["start"] + 1
        if len(with_impl) != expected_r or len(irrelevant) != expected_i:
            print(f"Note {vid}: decoded {len(with_impl)}/{expected_r} range frames, "
                  f"{len(irrelevant)}/{expected_i} irrelevant frames")

        candidates = [("video_range", k, t) for k, t in enumerate(with_impl)] + \
                     [("irrelevant", k, t) for k, t in enumerate(irrelevant)]
        stack = np.stack([c[2] for c in candidates])

        names = [f"{vid}_contact.jpg"]
        result[vid] = {}
        for name in names:
            img = cv2.imread(os.path.join(args.frames_dir, name))
            errs = ((stack - thumb(img)) ** 2).mean(axis=(1, 2))
            best = int(errs.argmin())
            segment, offset, _ = candidates[best]
            err = float(errs[best])
            result[vid][name] = {"segment": segment, "offset": offset, "match_mse": round(err, 2)}
            total += 1
            if err > SUSPICIOUS_MSE:
                flagged.append((vid, name, err))

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Matched {total} frames from {len(result)} videos; {len(flagged)} above MSE {SUSPICIOUS_MSE}")
    for vid, name, err in flagged:
        print(f"  {name}: {err:.1f}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
