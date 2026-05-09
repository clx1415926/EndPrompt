# EndPrompt

This repository contains the experimental code for **“EndPrompt: Efficient Long-Context Extension via Terminal Anchoring”**. EndPrompt simulates long-context positional supervision with short physical sequences: it keeps the original short text as the first segment, appends a short terminal prompt as the second segment, and assigns the terminal segment position indices near the target context boundary. This exposes long-range RoPE relative-position relationships without full-length sequence training.

## Key Findings

- EndPrompt extends context length using short training sequences, avoiding the high memory and compute cost of full-length fine-tuning
- The method preserves the continuity of the original context while using a terminal prompt as an anchor for long-range positional supervision.
- The theoretical analysis is based on RoPE and Position Interpolation, showing that sparse long-range supervision can generalize to unobserved intermediate distances through smooth positional variation and shared Transformer parameters.
- On LLaMA-family models, the paper extends the context window from 8K to 64K and reports an average RULER score of 76.03, with the best average performance on LongBench.
- Compared with LCEG, LongLoRA, and full-length fine-tuning, EndPrompt achieves comparable or better performance with substantially lower training cost.

## Repository Structure

```text
./
├── README.md
├── requirements.txt
├── fine_tune_suffix.py
├── finetune_main_suffix.sh
├── finetune_suffix_random.py
├── fine-tune.py
├── fine_tune_new.py
├── fine_tune_half.py
├── fine_tune_half_new.py
├── occu.py
├── models/
│   └── modeling_llama.py
├── ds_configs/
│   ├── stage3.json
│   └── stage3_offload.json
├── LLAMA3_ADAPTATION_README.md
└── ADAPTATION_CHECKLIST.md
```

## Setup

Eight A800 GPUs or a stronger multi-GPU setup are recommended.

```bash
pip install -r ./requirements.txt
```

## Usage

### Recommended Entry Point

The main training entry point referenced by the LLaMA3 8B adaptation notes is:

```bash
bash ./finetune_main_suffix.sh
```

Default configuration in this script:

```text
METHOD_NAME=pi
TRAINING_LENGTH=65536
CHUNK_2_LENGTH=18
pose_final_position_id=65535
model_name_or_path=Llama-3-8B
```

Before running, update the model path, dataset path, output path, and DeepSpeed config path according to your local environment.

### Training Script Status

- `finetune_main_suffix.sh`: Main LLaMA3 suffix training entry point. It uses `Llama-3-8B`.
- `finetune_main_suffix_random.sh`: LLaMA3 random suffix-position training entry point. It also uses `Llama-3-8B`.
- `finetune.sh`: Legacy LLaMA2 training examples for original LLaMA, PI, LongLoRA, and Landmark. It uses `Llama-2-7b-hf`.
- `finetune_suffix.sh`: Legacy LLaMA2 suffix / terminal-prompt example. It uses `Llama-2-7b-hf`.
- `finetune_ntk_main.sh`: Legacy LLaMA2 NTK entry point. It uses `Llama-2-7B-32k`.
- `finetune_ntk_main_new.sh`: Continuation / checkpoint-based NTK-related entry point. It starts from `suffix_pi_32k` rather than a base LLaMA3 model.

For LLaMA3 experiments, use `finetune_main_suffix.sh` by default. The other legacy scripts should be treated as LLaMA2 baselines or templates unless their `--model_name_or_path`, RoPE-related settings, dataset path, and output path are explicitly updated.

## Key Scripts

- `fine_tune_suffix.py`: Core EndPrompt / suffix training script, supporting data processing modes such as `pose_fixed_split` and RoPE scaling configuration.
- `finetune_suffix_random.py`: Random terminal-prompt position variant.
- `fine-tune.py`: Base long-context training script supporting PI, NTK, YaRN, LongLoRA, Landmark, and related methods.
- `models/modeling_llama.py`: Project-local custom LLaMA modeling code.
- `ds_configs/stage3.json`, `ds_configs/stage3_offload.json`: DeepSpeed ZeRO Stage 3 configurations.
- `occu.py`: Utility script.

## Data and Outputs

Training scripts typically require:

```text
Model path: --model_name_or_path
Dataset path: --dataset_dir
Output path: --output_dir
DeepSpeed config: --deepspeed
```

The current shell scripts may contain cluster-specific paths. Replace them with local paths when migrating to a new environment.

## Notes

- This project focuses on training and method validation; it does not include full model weights or datasets.
- Training defaults to multi-GPU, bf16, FlashAttention, and DeepSpeed ZeRO Stage 3. Single-GPU or non-NVIDIA environments require configuration changes.
- `requirements.txt` does not pin all CUDA/DeepSpeed runtime dependencies; install versions compatible with your target machine.
