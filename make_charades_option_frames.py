"""
Build the per-option frame folders for the Charades before/after MCQs.

Use this for building frames for NEW MCQs. To reproduce the frames of the released
benchmark (the ones the paper's results were computed on), use
rebuild_charades_option_frames.py instead: it reads the exact source frame indices
stored in the released JSON, which this script cannot recover (see the notes on the
scrambled order and the frames.zip defects below).

Pipeline (Charades v1, test split only):
    inputs:  Charades_v1_test.csv, Charades_v1_classes.txt, Charades_v1/<id>.mp4
    step 1:  MCQ generation -> charades_mcq_before_after.json
             (this exact name; run_qwen.py and the eval scripts look for it).
             Every option is written with "frames": [].
    step 2:  this script -> Charades_v1/frames/<id>/<mcq_id>/{A,B,C,D}/frame_XXX.jpg

Each question has four options (JSON distractor_type in brackets):
    1. Correct: the true temporal neighbor frames.
    2. Scrambled (random): frames sampled from a random 1-5 s window of the video and
       randomly shuffled.
    3. Reversed (reversed): the same span of frames as the correct option, played backward.
    4. Closest Distractor (neighbor): the false temporal neighbor frames, i.e. the previous
       action to the anchor if the question is of type "after", and vice versa. When no
       suitable neighboring action exists on that side, the anchor action itself is used,
       which can never be the correct answer.

Frame ordering per option (checked against the frames.zip release):
    Correct, Closest         FRAMES_PER_SPAN frames sampled uniformly over "span", in time order
    Reversed                 reverse_order: true; the same sample, in reverse order
    Scrambled                scrambled: true; the same sample, permuted. The permutation is
                             seeded by --seed, mcq id and option key, so reruns are reproducible

Some stored spans run past the end of the video. They are clipped to the video's
duration (frame count / fps), so the last sampled index is one past the final frame
and cannot be read. That frame is skipped rather than padded, leaving a gap in the
numbering (7 frames; no frame_007.jpg, or no frame_000.jpg for a reversed option).

The frames.zip release has a defect in exactly these 530 folders: the original decoder
returned nothing in 103 of them (7 frames) and a stale frame from an unrelated point
in the same video in 427 (8 frames, the last one temporally wrong). Every other
frame in the release matches this script.

Option folders that already contain frames are skipped unless --overwrite is given,
so rerunning never touches the frames existing results were computed on.
The released frames.zip used an unrecorded shuffle, so scrambled orders regenerated
with --overwrite will not match it.
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON_PATH = os.path.join(SCRIPT_DIR, "charades_mcq_before_after.json")
DEFAULT_VIDEO_DIR = os.path.join(SCRIPT_DIR, "Charades_v1")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_VIDEO_DIR, "frames")
FRAMES_PER_SPAN = 8
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--json-path",
        default=DEFAULT_JSON_PATH,
        help="Path to charades_mcq_before_after.json",
    )
    parser.add_argument(
        "--video-dir",
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing Charades .mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where option frames will be saved",
    )
    parser.add_argument(
        "--frames-per-span",
        type=int,
        default=FRAMES_PER_SPAN,
        help="Number of frames to sample per option span",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for scrambled options")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract option folders that already contain frames",
    )
    parser.add_argument(
        "--update-json",
        action="store_true",
        help="Write the frame paths (relative to the JSON's directory) into each "
             "option's \"frames\" list, in place",
    )
    parser.add_argument("--limit", type=int, help="Stop after this many MCQs (for quick test runs)")
    return parser.parse_args()


def sample_indices(fps, total_frames, span, n_frames):
    """
    Frame indices sampled uniformly over span, in time order. The span is clipped
    to the video's duration and times are truncated to frame indices, as in the
    frames.zip release. A span clipped at the end of the video samples index
    total_frames, which cannot be read (see write_frames).
    """
    start = max(0.0, span[0])
    end = min(total_frames / fps, span[1])
    if end <= start:
        return []
    return [int(t * fps) for t in np.linspace(start, end, n_frames)]


def order_indices(indices, option, mcq_id, key, seed):
    """Apply the distractor's temporal manipulation to time-ordered indices."""
    if option.get("reverse_order"):
        return indices[::-1]
    if option.get("scrambled"):
        indices = list(indices)
        random.Random(f"{seed}:{mcq_id}:{key}").shuffle(indices)
        return indices
    return indices


def write_frames(cap, indices, out_dir):
    """
    Save frame_<position>.jpg for each index. Unreadable frames are skipped, not
    padded, which leaves a gap in the numbering. Returns paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for j, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(out_dir, f"frame_{j:03d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def existing_frames(out_dir):
    if not os.path.isdir(out_dir):
        return []
    return [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".jpg")]


def main():
    args = parse_args()

    with open(args.json_path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded dataset with {len(dataset)} videos.")
    print(f"JSON path: {args.json_path}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Frames per span: {args.frames_per_span}")

    json_dir = os.path.dirname(os.path.abspath(args.json_path))
    counts = {"written": 0, "skipped": 0, "failed": 0}
    n_mcqs = 0

    for vid, mcqs in dataset.items():
        if not mcqs:
            continue
        if args.limit is not None and n_mcqs >= args.limit:
            break

        video_path = os.path.join(args.video_dir, f"{vid}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: cannot open {video_path}")
            counts["failed"] += 4 * len(mcqs)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for mcq in mcqs:
            mcq_id = mcq["id"]
            for key, option in mcq["options"].items():
                out_dir = os.path.join(args.output_dir, vid, mcq_id, key)
                paths = existing_frames(out_dir)

                if paths and not args.overwrite:
                    counts["skipped"] += 1
                else:
                    for path in paths:
                        os.remove(path)
                    paths = []
                    if fps > 0 and total_frames > 0 and option["span"][1] > option["span"][0]:
                        indices = sample_indices(fps, total_frames, option["span"], args.frames_per_span)
                        paths = write_frames(cap, order_indices(indices, option, mcq_id, key, args.seed), out_dir)
                    if not paths:
                        print(f"Warning: no frames for {vid}/{mcq_id}/{key} span={option['span']}")
                        counts["failed"] += 1
                        continue
                    counts["written"] += 1

                if args.update_json:
                    option["frames"] = [os.path.relpath(p, json_dir) for p in paths]

            n_mcqs += 1
            if n_mcqs % 100 == 0:
                print(f"Processed {n_mcqs} MCQs... {counts}")

        cap.release()

    print(f"Done: {n_mcqs} MCQs, option folders {counts}")
    print(f"All frames saved under {args.output_dir}")

    if args.update_json:
        tmp_path = f"{args.json_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(dataset, f, indent=2)
        os.replace(tmp_path, args.json_path)
        print(f"Frame paths written into {args.json_path}")


if __name__ == "__main__":
    main()
