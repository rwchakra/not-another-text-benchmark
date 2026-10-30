# Not Another Text Benchmark

This repository contains code for creating datasets and running inference on three video understanding tasks, from our NeurIPS (Evaluations and Datasets) 2026 paper, ["Not Another Text Benchmark: Putting the “Visual" Back in Visual Question Answering for Large Video Models"](https://arxiv.org/abs/2609.17112). 

- Please ensure that you have [Charades](https://prior.allenai.org/projects/charades) downloaded, and have access to the [Shapestacks](https://ogroth.github.io/shapestacks/) source [code](https://github.com/ogroth/shapestacks).
- Please ensure that you have our dataset downloaded as well, which is available at: <https://www.kaggle.com/datasets/natbneurips2026/tfr-vfp-cmd>


## Abstract
Large video models have exhibited impressive performance on a wide range of visual question answering tasks, owing to the rise of powerful, pretrained text and vision encoders. The usefulness of such models have also been demonstrated on a wide range of benchmarks, with an important caveat - the dominant approach in these benchmarks evaluates multiple choice reasoning via text options. This is a natural way to test text-based reasoning in these models, and has led to significant insights regarding model behavior in the community. In this work, we ask a different question - what happens when the evaluation modality is visual, rather than text? We introduce three new vision-centric evaluation benchmarks in temporal frame retrieval, video future prediction, and causal memory distortion, all designed around evaluating visual understanding capabilities in large video models. Our approach complements the existing approaches to evaluate video understanding in frontier models. We show that current frontier models exhibit significant weakness when attempting to reason through visual queries, rather than text. We conclude with an extended analysis section that provides pointers for future improvements in visual understanding for large video models.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Creation

### Temporal Frame Retrieval (Charades)

- **charades_temporal_mcq.py** - Creates the dataset for the temporal frame retrieval task using Charades videos.

**Usage:** Reads `Charades_v1_test.csv` (only the test split is used) and `Charades_v1_classes.txt` when you download Charades.

```bash
python charades_temporal_mcq.py
```

Name the output `charades_mcq_before_after.json`. The frames for each option (`Charades_v1/frames/<video_id>/<mcq_id>/{A,B,C,D}/*.jpg`) are included in our dataset link, so they will automatically be loaded.

### Video Future Prediction (ShapeStacks)

- **extract_frames.sh** - Extracts and organizes frames from ShapeStacks videos into input/label directories.
- **make_shapestacks_distractors.py** - Generates distractor options (reversed, scrambled sequences) for the multiple-choice future prediction task.

We also provide `sampled_dirs.txt`, a list of directories we sampled from ShapeStacks. Please see the paper for details on the sampling procedure.

**Usage:** Run the scripts in this order.

1. Split frames 00–07 into `input/` and frames 08–15 into `label/`:

   ```bash
   bash extract_frames.sh sampled_dirs.txt
   ```

2. Build the answer options `A`–`D` and write `options_shapestacks.json`. Run this from the directory containing `sampled_dirs.txt`:

   ```bash
   python make_shapestacks_distractors.py
   ```

### Causal Memory Distortion

- **causal_benchmark_builder.py** - Annotation tool for creating the causal memory distortion benchmark following the experimental design from Strickland & Keil (2011).

**Usage:** Download `causal_ground_truth.json`, `videos/` and `frames/` from the dataset link into one folder and pass it to `run_qwen.py` as `--dataset_path`. `causal_benchmark_builder.py` is only needed if you want to annotate new videos.

**Annotating new videos:**

1. Run the builder on a Charades video. A viewer window opens, where you mark the contact moment and the segments to cut (arrow keys or `a`/`d` to step through frames, `SPACE` to mark a frame). The six videos are saved to `benchmark/videos/`.

   ```python
   from causal_benchmark_builder import VideoMemoryBenchmarkBuilder

   builder = VideoMemoryBenchmarkBuilder(output_dir="benchmark")
   item = builder.build_test_item("path/to/Charades_v1/75M1Z.mp4", video_id="75M1Z", action_label="throwing ball up")
   builder.save_benchmark([item])  # saves the frame annotations to benchmark/annotations/benchmark.json
   ```

2. Copy the six videos into your `videos/` folder, and save the contact frame (the frame you marked) as `frames/<video_id>_contact.jpg`. The builder does not save it for you.
3. Add `{"video_id": "<video_id>"}` to the `videos` list in `causal_ground_truth.json`.

## Configuration

Set these placeholders at the top of each script before running it:

| Script | Variable | Set to |
|---|---|---|
| `charades_temporal_mcq.py` | `BASE_PATH = "path_to_charades"` | Directory containing the Charades CSV and classes file |
| `charades_temporal_mcq.py` | `OUT_JSON = "path_to_output.json"` | e.g. `<dataset_path>/charades_mcq_before_after.json` |
| `make_shapestacks_distractors.py` | `base_path = 'path_to_data'` | Directory that the paths in `sampled_dirs.txt` are relative to |

`extract_frames.sh` and `causal_benchmark_builder.py` don't have placeholders.

## Inference

We provide an example inference script for Qwen2.5-VL, which was at the time the best performing open-weight model. For all the other models tested in the paper, please refer to their individual documentation, and API usage details. 

**run_qwen.py** runs Qwen2.5-VL (3B, 7B, 32B or 72B) on any of the three tasks. Model weights are downloaded from the HuggingFace Hub.

```bash
python run_qwen.py --dataset {causal,charades,shapestacks} --dataset_path /data/<task> --model_size 7B
```

It expects this layout under `--dataset_path`:

```text
causal:       causal_ground_truth.json
              videos/<id>_<condition>.mp4
              frames/<id>_contact.jpg
charades:     charades_mcq_before_after.json
              Charades_v1/<id>.mp4
              Charades_v1/frames/<id>/<mcq_id>/{A,B,C,D}/*.jpg
shapestacks:  json/options_shapestacks.json
              shapestacks_final/<sample>/{input,A,B,C,D}/*.jpg
```

Options:

- `--output`: path for the results JSON. If the file already exists, the run resumes from it.
- `--limit N`: stop after N samples, for quick test runs.
- `--seed`: seed for the complete/incomplete split (causal task only).
- `--cache_dir`: where to store the model weights.
- `--attn_implementation`: attention backend (`flash_attention_2`, `sdpa` or `eager`).

## Citation

If you find this dataset and paper useful, please consider citing our work:

```bibtex
@misc{chakraborty2026textbenchmarkputtingvisual,
  title={Not Another Text Benchmark: Putting the "Visual" Back in Visual Question Answering for Large Video Models},
  author={Rwiddhi Chakraborty and Yinong Oliver Wang and Cheng Zhang and Fan Bai and Zhuoran Yu and Michael Kampffmeyer and Yong Jae Lee and Fernando De la Torre and Robert Jenssen},
  year={2026},
  eprint={2609.17112},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.17112},
}
```
