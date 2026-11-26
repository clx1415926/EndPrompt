# EndPrompt

This repository contains the experimental code for **“EndPrompt: Efficient Context Window Extension via Optimally Shifted Terminal Cue”** (DASFAA 2027 submission). EndPrompt (ET) introduces long-range positional supervision with short training sequences: it keeps an intact short context as the first segment, appends a fixed terminal cue, and shifts only the cue's positional indices by a blockwise offset drawn from an offline D-optimal design over RoPE features. This exposes long-range RoPE relative-position relationships without processing target-length inputs.

## Key Findings

- Fine-tuning with 8K training inputs extends LLaMA-3-8B from 8K to 64K: **81.39** average RULER (13 tasks, averaged over 4K–64K evaluations) and **39.03** on LongBench (16 English tasks), using 1B LCEG training tokens and one epoch.
- ET beats PoSE (79.31 / 37.95), RandPos (48.34 / 32.59), LCEG (72.24 / 36.61), LongLoRA (72.95 / 36.84), and full-length fine-tuning (69.23 / 35.63) on the same 64K setting.
- The method transfers to Mistral-7B-v0.3 (72.39 RULER / 37.29 LongBench) by recomputing the offset table from the corresponding RoPE frequencies.
- At the 64K target, ET uses 36.52 GB of peak training memory, 52% less than full-length fine-tuning, and trains 1.41x faster.
- The D-optimal offset distribution adds 1.17 RULER / 0.56 LongBench points over uniform offset sampling; the fixed-maximum offset does worst, so spreading samples across the offset range matters.
- After SFT on UltraChat-12K, the ET-adapted model averages 53.56 across MMLU, GSM8K, HellaSwag, and HumanEval (PoSE: 52.32; full FT: 50.98).

## Repository Structure

```text
./
├── README.md
├── requirements.txt
├── fine_tune_suffix.py          # core ET training script (final scheme, mode "et")
├── finetune_main_suffix.sh      # main LLaMA-3-8B ET entry point (D-optimal offsets)
├── finetune_main_suffix_random.sh  # uniform-offset ablation of the same construction
├── fine_tune_new.py             # legacy: half-context self-distillation variant
├── fine_tune_half.py            # legacy: half-context self-distillation variant
├── fine_tune_half_new.py        # legacy: half-context continuation variant
├── finetune_suffix_random.py    # legacy: random suffix-position training
├── fine-tune.py                 # base long-context training (PI/NTK/YaRN/LongLoRA/Landmark)
├── finetune.sh                  # legacy LLaMA-2 recipes
├── finetune_suffix.sh           # legacy LLaMA-2 suffix / terminal-prompt example
├── finetune_ntk_main.sh         # legacy LLaMA-2 NTK entry point
├── finetune_ntk_main_new.sh     # legacy checkpoint-continuation NTK entry point
├── models/
│   └── modeling_llama.py        # project-local LLaMA modeling (RoPE position_ids + linear PI)
├── ds_configs/
│   ├── stage3.json
│   └── stage3_offload.json
└── skip_design/
    ├── gate1_doptimal.py        # offline D-optimal offset design (paper Appendix)
    ├── gate1_w1_p_star.npy      # D-optimal offset distribution (LLaMA-3-8B, 8K->64K)
    ├── gate1_w1_p_star_cdf.npy  # its cumulative probabilities (length 57345 = L-a-b+1)
    ├── gate1_w1_summary.json    # design diagnostics (r=74, efficiency bounds, ...)
    └── gate2_ggn_rank.py        # GGN rank-consistency check for the design surrogate
```

## Setup

Eight A800 GPUs or a stronger multi-GPU setup are recommended.

```bash
pip install -r ./requirements.txt
```

## Usage

### Recommended Entry Point

The main ET training entry point is the DASFAA default configuration:

```bash
bash ./finetune_main_suffix.sh
```

Default configuration in this script:

```text
MODEL_NAME_OR_PATH  = Llama-3-8B
TRAINING_LENGTH     = 8192     # physical input: 8182 context + 10 cue tokens
TARGET_LENGTH       = 65536    # logical target window L (RoPE config; s = 8 linear PI)
CHUNK_2_LENGTH      = 10       # cue token count b (EP_1 below tokenizes to 10 tokens)
CUE_TEXT            = "This is the end of text, please pay attention here"   # EP_1
SKIP_MODE           = doptimal # offsets: doptimal | uniform | fixed_max
SKIP_CDF            = skip_design/gate1_w1_p_star_cdf.npy
DATASET             = pre-tokenized LCEG (jsonl with an "input_ids" field, ~8K-token sequences)
```

Training details matched to the paper: all parameters fine-tuned, one epoch over the LCEG corpus (about 1B tokens), global batch 64 (4 per GPU, 2 accumulation steps, 8 A800 GPUs), learning rate 2e-5 constant after 20 warmup steps, no weight decay, BF16, DeepSpeed ZeRO-3, FlashAttention, and gradient checkpointing.

Before running, update the model path, dataset path, output path, and the DeepSpeed config path to your local environment. The dataset must be pre-tokenized and expose an `input_ids` field; the script asserts that the cue text tokenizes to exactly `CHUNK_2_LENGTH` tokens.

### Offset Sampling Ablations

- `finetune_main_suffix_random.sh`: identical construction with `--et_skip_mode uniform` (Delta uniform over [0, L-a-b]); the paper's uniform-offset ablation.
- `--et_skip_mode fixed_max`: the paper's fixed-maximum-offset ablation (cue always at the window limit).

### Offset Design (skip_design)

The D-optimal table is configuration-specific. For a new (model, L, s, a, b) configuration, recompute it offline:

```bash
cd skip_design
python gate1_doptimal.py --a0 8182 --b 10 --wcue 1.0 --rank_sensitivity --out_prefix gate1_w1
```

This reproduces the shipped LLaMA-3-8B 8K-to-64K table: r = 74 retained directions, uniform
efficiency in [0.8889, 0.8898], and q-hat efficiency in [0.9990, 1.0]. `gate1_w1_p_star_cdf.npy`
is the cumulative table used during training via `delta = np.searchsorted(cdf, u, side="right")`.
`gate2_ggn_rank.py` is the auxiliary GGN rank-consistency check over the design surrogate; see the
script docstring for the pre-registered protocol.

## Training Script Status

- `finetune_main_suffix.sh`: main LLaMA-3-8B ET entry point (D-optimal offsets, final scheme).
- `finetune_main_suffix_random.sh`: uniform-offset ET ablation (final scheme construction).
- `fine_tune_suffix.py`: core ET script; `data_processing_mode="et"` implements the final construction.
- `finetune_suffix_random.py`: legacy random suffix-position variant (pre-table uniform sampling).
- The remaining scripts are legacy LLaMA-2 baselines or templates (PI, NTK, YaRN, LongLoRA, Landmark) and should be treated as reference unless their paths and RoPE settings are updated explicitly.

## Evaluation

All checkpoints in the paper are evaluated with contiguous position indices and no appended cue: RULER (all samples of all 13 tasks at 4K, 8K, 16K, 32K, and 64K) and the 16 English LongBench tasks with the official prompts.

## Notes

- This project focuses on training and method validation; it does not include full model weights or datasets.
- Training defaults to multi-GPU, bf16, FlashAttention, and DeepSpeed ZeRO Stage 3. Single-GPU or non-NVIDIA environments require configuration changes.
- `requirements.txt` does not pin all CUDA/DeepSpeed runtime dependencies; install versions compatible with your target machine.