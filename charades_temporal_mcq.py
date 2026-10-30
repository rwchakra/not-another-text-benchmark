import os
import csv
import json
import random
from collections import defaultdict

BASE_PATH = "path_to_charades"
CHARADES_CSV = BASE_PATH + "/Charades_v1_test.csv"
CLASS_FILE = BASE_PATH + "/Charades_v1_classes.txt"
OUT_JSON = "path_to_output.json"

MIN_DUR = 1.0
MAX_DUR = 5.0

# ---------------------------------------------------------------------
# Load class text mapping
# ---------------------------------------------------------------------
def load_class_mapping(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            cid, text = line.strip().split(" ", 1)
            mapping[cid] = text.strip()
    return mapping


# ---------------------------------------------------------------------
# Parse Charades CSV
# ---------------------------------------------------------------------
def parse_charades_csv(csv_path):
    data = {}
    max_end = 0.0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["id"].strip()
            actions_str = row.get("actions", "").strip()
            data[vid] = []

            if not actions_str:
                continue

            for seg in actions_str.split(";"):
                parts = seg.split()
                if len(parts) != 3:
                    continue
                c_id, start, end = parts
                start, end = float(start), float(end)
           
                dur = end - start
                if dur <= MIN_DUR or dur >= MAX_DUR:
                    continue
                data[vid].append({
                    "class_id": c_id,
                    "start": start,
                    "end": end
                })

        for vid in data:
            data[vid].sort(key=lambda x: x["start"])

    return data


def make_question(label_text, rel_type):
    label_text = label_text.strip().rstrip(".")
    if rel_type == "after":
        return f"What happens after the following action: {label_text}?"
    elif rel_type == "before":
        return f"What happens before the following action: {label_text}?"
    return None


# ---------------------------------------------------------------------
# Temporal utilities
# ---------------------------------------------------------------------
def find_next_action(actions, current):
    for act in actions:
        if act["start"] > current["end"]:
            return act
    return None


def find_previous_action(actions, current):
    for act in reversed(actions):
        if act["end"] < current["start"]:
            return act
    return None


def sample_random_span(video_length, min_len=1.0, max_len=5.0):
    span_len = random.uniform(min_len, max_len)
    start = random.uniform(0, max(0, video_length - span_len))
    end = min(video_length, start + span_len)
    return [start, end]


# ---------------------------------------------------------------------
# MCQ Creation
# ---------------------------------------------------------------------
def create_mcq(video_id, rel_type, context, correct, idx, id2label, actions, video_length, corpus_class_ids):
    qid = f"{video_id}_{idx:04d}"
    label_text = id2label.get(context["class_id"], context["class_id"])
    question = make_question(label_text, rel_type)

    # classes inside this video (exclude them for random-scramble)
    classes_in_video = {a["class_id"] for a in actions}

    letters = ["A", "B", "C", "D"]
    correct_pos = random.choice(letters)
    options = {}

    # -----------------------------
    # Correct option
    # -----------------------------
    correct_option = {
        "frames": [],
        "span": [correct["start"], correct["end"]],
        "class_id": correct["class_id"],
        "text_option": id2label[correct["class_id"]],
        "label": "correct"
    }

    # -----------------------------
    # Distractor 1 — Random Scramble
    # -----------------------------
    rand_span = sample_random_span(video_length)
    # pick random class ID NOT in this video (This is just for the text-only evaluation: the frames will be scrambled anyway for the frame option)
    valid_corpus = [cid for cid in corpus_class_ids if cid not in classes_in_video]
    rand_cid = random.choice(valid_corpus)
    valid_corpus = [cid for cid in corpus_class_ids if cid not in classes_in_video and cid != rand_cid]
    distractor_random = {
        "frames": [],
        "span": rand_span,
        "class_id": rand_cid,
        "text_option": id2label[rand_cid],
        "label": "incorrect",
        "distractor_type": "random",
        "scrambled": True
    }

    # -----------------------------
    # Distractor 2 — Reversed (uses same class id as correct), but random id in text_option for text evaluation
    # -----------------------------
    #Sample randomly beyond a 10 index range to reduce semantic similarity
    new_rand_cid = random.choice([cid for cid in valid_corpus if abs(int(cid.lstrip('c')) - int(str(correct["class_id"]).lstrip('c'))) > 10]) 
    distractor_reversed = {
        "frames": [],
        "span": [correct["start"], correct["end"]],  # canonical order; reversed order applied later in sampling
        "class_id": correct["class_id"],
        "text_option": id2label[new_rand_cid],
        "label": "incorrect",
        "reverse_order": True,
        "distractor_type": "reversed"
    }

    # -----------------------------
    # Distractor 3 — Neighbor span
    # -----------------------------
    neighbor = None
    if rel_type == "after":
        neighbor = find_previous_action(actions, context)
    else:
        neighbor = find_next_action(actions, context)

    if neighbor:
        distractor_neighbor = {
            "frames": [],
            "span": [neighbor["start"], neighbor["end"]],
            "class_id": neighbor["class_id"],
            "text_option": id2label[neighbor["class_id"]],
            "label": "incorrect",
            "distractor_type": "neighbor"
        }
    else:
        # fallback if, e.g. current context is start of video and previous action is none — return same option as current context 
        fallback_span = [context["start"], context["end"]]
        fallback_cid = context["class_id"]
        distractor_neighbor = {
            "frames": [],
            "span": fallback_span,
            "class_id": fallback_cid,
            "text_option": id2label[fallback_cid],
            "label": "incorrect",
            "distractor_type": "neighbor"
        }

    distractors = [distractor_random, distractor_reversed, distractor_neighbor]
    random.shuffle(distractors)

    d_i = 0
    for letter in letters:
        if letter == correct_pos:
            options[letter] = correct_option
        else:
            options[letter] = distractors[d_i]
            d_i += 1

    return {
        "id": qid,
        "video_id": video_id,
        "type": rel_type,
        "question": question,
        "options": options,
        "metadata": {
            "context_class_id": context["class_id"],
            "correct_class_id": correct["class_id"],
            "context_span": [context["start"], context["end"]],
            "correct_span": [correct["start"], correct["end"]],
            "correct_option": correct_pos
        }
    }


# ---------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------
def build_mcq_dataset(videos, id2label):
    dataset = {}
    corpus_class_ids = list(id2label.keys())
    total_mcq = 0
    with open(CHARADES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        video_length = {}
        for row in reader:
            vid = row["id"].strip()
            length = float(row["length"].strip())
            video_length[vid] = length
    for vid, actions in videos.items():
        entries = []
        idx_local = 0
        if not actions:
            dataset[vid] = []
            continue

        #video_length = max(a["end"] for a in actions)
        video_duration = video_length.get(vid, 0.0)



        for a in actions:
            after_act = find_next_action(actions, a)
            before_act = find_previous_action(actions, a)

            if after_act:
                q = create_mcq(vid, "after", a, after_act, idx_local, id2label, actions, video_duration, corpus_class_ids)
                entries.append(q)
                idx_local += 1

            if before_act:
                q = create_mcq(vid, "before", a, before_act, idx_local, id2label, actions, video_duration, corpus_class_ids)
                entries.append(q)
                idx_local += 1

        dataset[vid] = entries
        total_mcq += len(entries)

    print(f"Total videos: {len(videos)}")
    print(f"Total MCQs: {total_mcq}")
    return dataset


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading class labels...")
    id2label = load_class_mapping(CLASS_FILE)

    print("Parsing CSV...")
    videos = parse_charades_csv(CHARADES_CSV)

    print("Building MCQ JSON...")
    dataset = build_mcq_dataset(videos, id2label)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved: {OUT_JSON}")
    print("Done.")

if __name__ == "__main__":
    main()
