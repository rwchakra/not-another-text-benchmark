"""
Qwen2.5-VL video inference for the causal, charades and shapestacks benchmarks.

Merges causal/qwen_2.5_video.py, charades/qwen_2.5_video.py and
shapestacks/qwen2.5_shapestacks_video.py. Prompts, frame sampling, generation
settings and result formats match the originals so results stay comparable.

Usage:
    python run_qwen.py --dataset charades --dataset_path /data/charades --model_size 72B
    python run_qwen.py --dataset shapestacks --dataset_path /data/shapestacks --model_size 7B --limit 5

Expected layout under --dataset_path:
    causal:      videos/<id>_<condition>.mp4, frames/<id>_contact.jpg, causal_ground_truth.json
    charades:    charades_mcq_before_after.json, Charades_v1/<id>.mp4,
                 Charades_v1/frames/<id>/<mcq_id>/{A,B,C,D}/*.jpg
    shapestacks: json/options_shapestacks.json, shapestacks_final/<sample>/{input,A,B,C,D}/*.jpg

Model weights are fetched from the HuggingFace Hub into the standard cache
(set HF_HOME or pass --cache_dir to put them elsewhere).
"""
import argparse
import importlib.util
import json
import os
import random
import re
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DATASETS = ["causal", "charades", "shapestacks"]
MODEL_SIZES = ["3B", "7B", "32B", "72B"]
OPTION_KEYS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def even_indices(total, num):
    """Evenly spaced indices; all of them if there are fewer than num."""
    if total >= num:
        return np.linspace(0, total - 1, num, dtype=int)
    return range(total)


def extract_frames_from_video(video_path, num_frames):
    """Extract evenly spaced frames from a video file as PIL images."""
    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video has no frames: {video_path}")
        cap.release()
        return None

    frames = []
    for idx in even_indices(total_frames, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames if len(frames) > 0 else None


def get_frames_from_folder(folder, num_frames):
    """Evenly sampled frames from an image folder as PIL images, or None."""
    if not os.path.exists(folder):
        print(f"Warning: Folder not found: {folder}")
        return None

    files = sorted(f for f in os.listdir(folder) if f.endswith((".jpg", ".png", ".jpeg")))
    if len(files) == 0:
        print(f"Warning: No image files in {folder}")
        return None

    return [
        Image.open(os.path.join(folder, files[i])).convert("RGB")
        for i in even_indices(len(files), num_frames)
    ]


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    """Write via a temp file so a killed job never leaves a truncated results file."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def require_paths(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing dataset paths:\n  " + "\n  ".join(missing))


def prepare_inputs(model, processor, content, videos, images=None):
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=videos,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    return text, inputs


def greedy_generate(model, processor, inputs, max_new_tokens):
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def default_attn_implementation():
    """flash_attention_2 when flash-attn is installed and a GPU is present, else PyTorch SDPA."""
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def load_model(model_size, cache_dir=None, attn_implementation=None):
    model_name = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
    attn_implementation = attn_implementation or default_attn_implementation()
    print(f"Loading model {model_name} (attention: {attn_implementation})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    )
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model loaded successfully!")
    return model, processor


# ---------------------------------------------------------------------
# Causal Memory Distortion (Uses Charades)
# ---------------------------------------------------------------------
CAUSAL_NUM_FRAMES = 16
CAUSAL_COMPLETE_SAMPLE_SIZE = 10
CAUSAL_CONDITIONS = {
    "complete": ("yes", [
        ("complete_with_implication", "_with_implication_complete.mp4"),
        ("complete_without_implication", "_without_implication_complete.mp4"),
        ("complete_scrambled", "_scrambled_complete.mp4"),
    ]),
    "incomplete": ("no", [
        ("incomplete_with_implication", "_with_implication_incomplete.mp4"),
        ("incomplete_without_implication", "_without_implication_incomplete.mp4"),
        ("incomplete_scrambled", "_scrambled_incomplete.mp4"),
    ]),
}


def parse_yes_no(response_text):
    """Leading whole-word yes/no; anything else is invalid."""
    m = re.match(r"\s*(yes|no)\b", response_text.lower())
    return m.group(1) if m else "invalid"


def causal_split(all_video_ids, split_path, has_results, seed):
    """
    Pick which videos get the complete vs incomplete conditions. The split is
    saved next to the results so a resumed run reuses it instead of redrawing.
    """
    if os.path.exists(split_path):
        print(f"Reusing complete/incomplete split from {split_path}")
        return load_json(split_path, None)

    if has_results:
        raise RuntimeError(
            f"Results exist but split file {split_path} is missing; resuming would "
            "draw a new split and mix conditions. Use a fresh --output."
        )

    complete = random.Random(seed).sample(all_video_ids, CAUSAL_COMPLETE_SAMPLE_SIZE)
    split = {
        "seed": seed,
        "complete": complete,
        "incomplete": [v for v in all_video_ids if v not in complete],
    }
    save_json(split_path, split)
    return split


def run_causal(model, processor, dataset_path, output, limit, seed):
    video_dir = os.path.join(dataset_path, "videos")
    frames_dir = os.path.join(dataset_path, "frames")
    ground_truth_path = os.path.join(dataset_path, "causal_ground_truth.json")
    require_paths(video_dir, frames_dir, ground_truth_path)

    print(f"Loading ground truth from: {ground_truth_path}")
    ground_truth = load_json(ground_truth_path, None)
    all_video_ids = [item["video_id"] for item in ground_truth["videos"]]
    print(f"Total videos available: {len(all_video_ids)}")

    all_results = load_json(output, [])
    completed = set((r["video_id"], r["condition"]) for r in all_results)
    if all_results:
        print(f"Resuming from {len(all_results)} existing results")

    split = causal_split(
        all_video_ids, os.path.splitext(output)[0] + "_split.json", bool(all_results), seed
    )
    print(f"Complete assignments: {len(split['complete'])}")
    print(f"Incomplete assignments: {len(split['incomplete'])}")

    new_trials = 0
    for group, (answer, conditions) in CAUSAL_CONDITIONS.items():
        for condition_name, video_suffix in conditions:
            print("\n" + "=" * 60)
            print(f"Processing {condition_name.upper()}")
            print("=" * 60)

            for video_id in split[group]:
                if limit is not None and new_trials >= limit:
                    print(f"Reached --limit {limit}")
                    return
                if (video_id, condition_name) in completed:
                    print(f"Skipping {video_id} ({condition_name}) - already done")
                    continue

                print(f"\n--- {video_id} ({condition_name}) ---")
                video_path = os.path.join(video_dir, f"{video_id}{video_suffix}")
                video_frames = extract_frames_from_video(video_path, CAUSAL_NUM_FRAMES)
                if video_frames is None:
                    print(f"Failed to extract frames from: {video_path}")
                    continue
                print(f"Extracted {len(video_frames)} frames from video")

                contact_path = os.path.join(frames_dir, f"{video_id}_contact.jpg")
                if not os.path.exists(contact_path):
                    print(f"Contact frame not found: {contact_path}")
                    continue
                contact_frame = Image.open(contact_path).convert("RGB")

                content = [
                    {"type": "video", "video": video_frames, "fps": 4},
                    {"type": "text", "text": "Did this frame appear in the video?"},
                    {"type": "image", "image": contact_frame},
                    {"type": "text", "text": "Answer only Yes or No."},
                ]

                try:
                    _, inputs = prepare_inputs(
                        model, processor, content, videos=[video_frames], images=[contact_frame]
                    )
                    response = greedy_generate(model, processor, inputs, max_new_tokens=32)
                except Exception as e:
                    print(f"Error processing {video_id} ({condition_name}): {e}")
                    traceback.print_exc()
                    continue
                finally:
                    torch.cuda.empty_cache()

                print(f"Raw response: {response}")
                result = {
                    "video_id": video_id,
                    "condition": condition_name,
                    "video_path": video_path,
                    "contact_path": contact_path,
                    "ground_truth": answer,
                    "prediction": parse_yes_no(response),
                    "response": response,
                }
                all_results.append(result)
                completed.add((video_id, condition_name))
                new_trials += 1
                save_json(output, all_results)

                print(f"Ground truth: {result['ground_truth']}, Prediction: {result['prediction']}")
                print(f"Progress: {len(all_results)} trials completed")


# ---------------------------------------------------------------------
# Temporal Frame Retrieval (Uses Charades)
# ---------------------------------------------------------------------
CHARADES_NUM_MAIN_FRAMES = 16
CHARADES_NUM_OPTION_FRAMES = 8
CHARADES_TOP_K_LOGPROBS = 4


def get_input_token_counts(processor, text, inputs):
    """Text-only and full multimodal token counts for a single example."""
    text_token_count = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
    total_input_token_count = int(inputs.attention_mask[0].sum().item())
    return {
        "text_token_count": text_token_count,
        "visual_token_count": max(total_input_token_count - text_token_count, 0),
        "total_input_token_count": total_input_token_count,
    }


def generate_with_logprobs(model, processor, inputs, max_new_tokens, top_k):
    """Greedy generation, then chosen-token and top-k logprobs from the step scores."""
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = output_ids.sequences[:, inputs.input_ids.shape[1]:]
    logprobs_data = {"chosen_logprobs": [], "top_candidates": []}

    for token_id, logits in zip(generated_ids[0], output_ids.scores):
        log_probs = F.log_softmax(logits[0], dim=-1)
        logprobs_data["chosen_logprobs"].append({
            "token": processor.tokenizer.decode([token_id.item()], skip_special_tokens=False),
            "token_id": token_id.item(),
            "logprob": log_probs[token_id].item(),
        })

        top_logprobs, top_indices = torch.topk(log_probs, top_k)
        logprobs_data["top_candidates"].append([
            {
                "token": processor.tokenizer.decode([tid.item()], skip_special_tokens=False),
                "token_id": tid.item(),
                "logprob": lp.item(),
            }
            for lp, tid in zip(top_logprobs, top_indices)
        ])

    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, logprobs_data


def run_charades(model, processor, dataset_path, output, limit):
    json_path = os.path.join(dataset_path, "charades_mcq_before_after.json")
    video_dir = os.path.join(dataset_path, "Charades_v1")
    frames_root = os.path.join(video_dir, "frames")
    require_paths(json_path, video_dir, frames_root)

    print("Loading dataset...")
    dataset = load_json(json_path, None)
    results = load_json(output, {})
    if results:
        print(f"Resuming with {len(results)} videos already processed")

    video_ids = list(dataset.keys())
    print(f"Total videos: {len(video_ids)}")

    new_videos = 0
    for idx, vid in enumerate(video_ids):
        if limit is not None and new_videos >= limit:
            print(f"Reached --limit {limit}")
            return
        if vid in results:
            print(f"Skipping {vid} (already processed)")
            continue

        results[vid] = []
        if not dataset[vid]:
            # Most Charades videos have no MCQs; record them without counting toward --limit.
            continue

        print("\n------------------------------------------------------------")
        print(f"Processing {vid} ({idx+1}/{len(video_ids)})")
        new_videos += 1

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            save_json(output, results)
            continue

        main_video_frames = extract_frames_from_video(video_path, CHARADES_NUM_MAIN_FRAMES)
        if main_video_frames is None:
            print(f"Failed to extract frames from main video: {video_path}")
            save_json(output, results)
            continue
        print(f"  Main video: {len(main_video_frames)} frames extracted, resolution: {main_video_frames[0].size}")

        for mcq in dataset[vid]:
            mcq_id = mcq["id"]
            print(f"Question: {mcq_id}")

            prompt = (
                "You will be shown a video and a multiple-choice question about events in it.\n\n"
                f"Question: {mcq['question']}\n\n"
                "Options:"
            )
            content = [
                {"type": "text", "text": prompt},
                {"type": "video", "video": main_video_frames, "fps": 4},
            ]
            all_videos = [main_video_frames]

            for key in OPTION_KEYS:
                frames = get_frames_from_folder(
                    os.path.join(frames_root, vid, mcq_id, key), CHARADES_NUM_OPTION_FRAMES
                )
                if frames:
                    print(f"  Option {key}: {len(frames)} frames loaded, resolution: {frames[0].size}")
                    content.append({"type": "text", "text": f"\n{key}."})
                    content.append({"type": "video", "video": frames, "fps": 4})
                    all_videos.append(frames)
                else:
                    print(f"  Option {key}: MISSING or empty")

            content.append({
                "type": "text",
                "text": "Answer with the option's letter from the given choices directly, and answer only with the option's letter (A/B/C/D).",
            })

            record = {
                "id": mcq_id,
                "type": mcq["type"],
                "question": mcq["question"],
                "prediction": None,
                "correct_option": mcq["metadata"]["correct_option"],
                "input_token_counts": None,
                "logprobs": None,
            }
            try:
                text, inputs = prepare_inputs(model, processor, content, videos=all_videos)
                record["input_token_counts"] = counts = get_input_token_counts(processor, text, inputs)
                print(
                    "  Input tokens:"
                    f" text={counts['text_token_count']},"
                    f" visual={counts['visual_token_count']},"
                    f" total={counts['total_input_token_count']}"
                )
                record["prediction"], record["logprobs"] = generate_with_logprobs(
                    model, processor, inputs, max_new_tokens=512, top_k=CHARADES_TOP_K_LOGPROBS
                )
                print(f"Model response: {record['prediction'][-200:]}")
            except Exception as e:
                print(f"Error processing {vid}/{mcq_id}: {e}")
                traceback.print_exc()
                record["error"] = str(e)

            results[vid].append(record)
            torch.cuda.empty_cache()

        save_json(output, results)
        print(f">>> Checkpoint saved to {output}")


# ---------------------------------------------------------------------
# Video Future Prediction (Uses ShapeStacks)
# ---------------------------------------------------------------------
SHAPESTACKS_NUM_INPUT_FRAMES = 32
SHAPESTACKS_NUM_OPTION_FRAMES = 8

# Kept byte-identical to shapestacks/gemini_video_visual.py so the two are comparable.
SHAPESTACKS_TASK_PROMPT = (
    "TASK: Physical outcome prediction for a block tower.\n\n"
    "You will see an initial sequence of frames showing a block tower, then four\n"
    "candidate outcomes labeled A, B, C, and D. Select the one that is most likely\n"
    "given the tower's physical stability.\n\n"
    "--- INITIAL SEQUENCE ---"
)
SHAPESTACKS_CANDIDATE_LABEL = "--- CANDIDATE {key} ---"
SHAPESTACKS_OUTPUT_INSTRUCTION = "OUTPUT: a single letter, one of A, B, C, or D. No other text."


def run_shapestacks(model, processor, dataset_path, output, limit):
    data_dir = os.path.join(dataset_path, "shapestacks_final")
    options_json = os.path.join(dataset_path, "json", "options_shapestacks.json")
    require_paths(data_dir, options_json)

    print(f"Loading options from: {options_json}")
    options_data = load_json(options_json, None)
    video_dirs = sorted(s for s in options_data if os.path.isdir(os.path.join(data_dir, s)))
    print(f"Found {len(video_dirs)} valid sample directories")

    results = load_json(output, {})
    if results:
        print(f"Resuming with {len(results)} samples already processed")

    new_samples = 0
    for idx, video_name in enumerate(video_dirs):
        if limit is not None and new_samples >= limit:
            print(f"Reached --limit {limit}")
            return
        if video_name in results:
            print(f"Skipping {video_name} (already processed)")
            continue

        print("\n------------------------------------------------------------")
        print(f"Processing {video_name} ({idx+1}/{len(video_dirs)})")
        sample_dir = os.path.join(data_dir, video_name)

        input_frames = get_frames_from_folder(
            os.path.join(sample_dir, "input"), SHAPESTACKS_NUM_INPUT_FRAMES
        )
        if input_frames is None:
            print(f"Missing input folder: {os.path.join(sample_dir, 'input')}")
            continue
        print(f"  Input: {len(input_frames)} frames")

        option_frames = {
            key: get_frames_from_folder(os.path.join(sample_dir, key), SHAPESTACKS_NUM_OPTION_FRAMES)
            for key in OPTION_KEYS
        }
        missing = [key for key, frames in option_frames.items() if not frames]
        if missing:
            print(f"Skipping {video_name} due to missing option folders: {missing}")
            continue

        content = [
            {"type": "text", "text": SHAPESTACKS_TASK_PROMPT},
            {"type": "video", "video": input_frames, "fps": 4},
        ]
        for key in OPTION_KEYS:
            content.append(
                {"type": "text", "text": f"\n\n{SHAPESTACKS_CANDIDATE_LABEL.format(key=key)}"}
            )
            content.append({"type": "video", "video": option_frames[key], "fps": 4})
        content.append({"type": "text", "text": f"\n\n{SHAPESTACKS_OUTPUT_INSTRUCTION}"})

        try:
            _, inputs = prepare_inputs(
                model, processor, content,
                videos=[input_frames] + [option_frames[key] for key in OPTION_KEYS],
            )
            response = greedy_generate(model, processor, inputs, max_new_tokens=10)
            pred_letter = response[0] if response and response[0] in "ABCD" else response

            print(f"Model response: {response}")
            print(f"Predicted letter: {pred_letter}")
            results[video_name] = {
                "prediction": pred_letter,
                "raw_response": response,
                "correct_option": options_data[video_name]["correct_option"],
            }
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            traceback.print_exc()
            results[video_name] = {"prediction": None, "error": str(e)}

        new_samples += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_json(output, results)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--dataset_path", required=True, help="Dataset root (see layout above).")
    parser.add_argument(
        "--model_size",
        required=True,
        type=lambda s: s.upper().rstrip("B") + "B",
        choices=MODEL_SIZES,
        help="Qwen2.5-VL size, e.g. 72B (or just 72).",
    )
    parser.add_argument(
        "--output",
        help="Results JSON (default: <dataset_path>/results_qwen25_<size>_<dataset>_video.json). "
             "An existing file is resumed.",
    )
    parser.add_argument("--limit", type=int, help="Stop after this many new samples (for quick test runs).")
    parser.add_argument(
        "--seed", type=int,
        help="Causal only: seed for the complete/incomplete split (default: unseeded). "
             "The split is saved next to the results either way.",
    )
    parser.add_argument("--cache_dir", help="HuggingFace cache for model weights (default: HF_HOME or ~/.cache/huggingface).")
    parser.add_argument(
        "--attn_implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend (default: flash_attention_2 if flash-attn is installed, else sdpa).",
    )
    args = parser.parse_args()

    args.dataset_path = os.path.abspath(args.dataset_path)
    if args.output is None:
        args.output = os.path.join(
            args.dataset_path, f"results_qwen25_{args.model_size.lower()}_{args.dataset}_video.json"
        )
    return args


def main():
    args = parse_args()

    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    if torch.cuda.is_available():
        print("CUDA available: True")
        print("GPU Name:", torch.cuda.get_device_name(0))
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    else:
        print("CUDA available: False — running on CPU")
    print(f"Dataset: {args.dataset} at {args.dataset_path}")
    print(f"Results: {args.output}")

    model, processor = load_model(args.model_size, args.cache_dir, args.attn_implementation)

    if args.dataset == "causal":
        run_causal(model, processor, args.dataset_path, args.output, args.limit, args.seed)
    elif args.dataset == "charades":
        run_charades(model, processor, args.dataset_path, args.output, args.limit)
    else:
        run_shapestacks(model, processor, args.dataset_path, args.output, args.limit)

    print(f"\nInference complete. Saved: {args.output}")


if __name__ == "__main__":
    main()
