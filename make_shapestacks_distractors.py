import os
import random
import shutil
import json

OPTIONS = ["A", "B", "C", "D"]

def read_frames(dir_path):
    frames = sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith(".jpg")
    )
    return frames

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_sequence(src_dir, frames, dst_dir):
    ensure_dir(dst_dir)
    for i, fname in enumerate(frames):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, f"{i:05d}.jpg")
        shutil.copy(src, dst)

with open("sampled_dirs.txt", "r") as f:
     dirs = [line.strip().lstrip("./") for line in f if line.strip()]

# Data is already in shapestacks_final with input/ and label/ subdirs
base_path = 'path_to_data'

# --- NEW: global metadata container ---
all_metadata = {}

for base_dir in dirs:
    env_dir = os.path.join(base_path, base_dir)
    input_dir = os.path.join(env_dir, "input")
    label_dir = os.path.join(env_dir, "label")

    if not os.path.isdir(input_dir) or not os.path.isdir(label_dir):
        print(f"Skipping {base_dir}: missing input/label")
        continue

    input_frames = read_frames(input_dir)
    label_frames = read_frames(label_dir)

    if len(input_frames) == 0 or len(label_frames) < 3:
        print(f"Skipping {base_dir}: insufficient frames")
        continue

    # Prepare sequences
    correct_seq = label_frames
    reversed_input = list(reversed(input_frames))
    reversed_correct = list(reversed(label_frames))

    # Scrambled correct: keep first and last, shuffle middle
    middle = label_frames[1:-1]
    random.shuffle(middle)
    scrambled_correct = [label_frames[0]] + middle + [label_frames[-1]]

    sequences = {
        "correct": (label_dir, correct_seq),
        "rev_input": (input_dir, reversed_input),
        "rev_correct": (label_dir, reversed_correct),
        "scrambled": (label_dir, scrambled_correct),
    }

    # Randomly assign correct option
    correct_option = random.choice(OPTIONS)
    remaining_options = [o for o in OPTIONS if o != correct_option]

    distractor_keys = ["rev_input", "rev_correct", "scrambled"]
    random.shuffle(distractor_keys)

    assignment = {correct_option: "correct"}
    for opt, key in zip(remaining_options, distractor_keys):
        assignment[opt] = key

    # Save sequences directly in the env directory
    for opt, key in assignment.items():
        src_dir, frames = sequences[key]
        dst_dir = os.path.join(env_dir, opt)

        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)

        copy_sequence(src_dir, frames, dst_dir)

    # --- NEW: record metadata ---
    all_metadata[base_dir] = {
        "correct_option": correct_option,
        "options": assignment
    }

    print(f"Processed {base_dir} (correct = {correct_option})")

# --- NEW: write global JSON ---
with open("options_shapestacks.json", "w") as jf:
    json.dump(all_metadata, jf, indent=2)