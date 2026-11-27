# EndPrompt

Code for **"EndPrompt: Efficient Context Window Extension via Optimally Shifted Terminal Cue"** (DASFAA 2027 submission).

EndPrompt (ET) introduces long-range positional supervision with short training inputs: an intact short context is followed by a fixed terminal cue, and only the cue's positional indices are shifted by a blockwise offset drawn from an offline D-optimal design over RoPE features.

## Key Results (LLaMA-3-8B, 8K -> 64K, 1B LCEG tokens)

- **81.39** average RULER (13 tasks, averaged over 4K-64K evaluations) and **39.03** LongBench (16 English tasks)
- 36.52 GB peak training memory, **52% less** than full-length fine-tuning, and **1.41x faster**
- Also validated on Mistral-7B-v0.3 (72.39 RULER / 37.29 LongBench)

## Repository Structure

```text
./
├── README.md
├── requirements.txt
├── fine_tune_suffix.py          # core ET training script
├── finetune_main_suffix.sh      # main LLaMA-3-8B entry point
├── models/
│   └── modeling_llama.py        # LLaMA modeling: position_ids-based RoPE + linear PI
├── ds_configs/
│   └── stage3.json              # DeepSpeed ZeRO-3
└── skip_design/
    ├── gate1_doptimal.py        # offline D-optimal offset design (paper Appendix)
    ├── gate1_w1_p_star.npy      # offset distribution for LLaMA-3-8B 8K->64K
    ├── gate1_w1_p_star_cdf.npy  # cumulative table used during training
    └── gate1_w1_summary.json    # design diagnostics
```

## Setup

Eight A800 GPUs are recommended.

```bash
pip install -r ./requirements.txt
```

## Usage

```bash
bash ./finetune_main_suffix.sh
```

Default configuration:

```text
TRAINING_LENGTH = 8192      # physical input: 8182 context + 10 cue tokens
TARGET_LENGTH   = 65536     # logical window L (RoPE config; linear PI, s = 8)
CUE_TEXT        = "This is the end of text, please pay attention here"  # EP_1, 10 tokens
SKIP_CDF        = skip_design/gate1_w1_p_star_cdf.npy
```

Training follows the paper: all parameters fine-tuned, one epoch over the LCEG corpus (~1B tokens), global batch 64 (4 per GPU, 2 accumulation steps, 8 A800 GPUs), learning rate 2e-5 constant after 20 warmup steps, BF16, DeepSpeed ZeRO-3, FlashAttention, gradient checkpointing.

Before running, update the model path, dataset path, and output path. The dataset must be pre-tokenized with an `input_ids` field (~8K-token sequences).

For a new (model, L, s, a, b) configuration, recompute the offset table offline:

```bash
cd skip_design
python gate1_doptimal.py --a0 8182 --b 10 --rank_sensitivity --out_prefix gate1_w1
```

Training samples offsets via `delta = np.searchsorted(cdf, u, side="right")`.

## Notes

- Evaluation uses contiguous positions and no appended cue (RULER at 4K-64K; the 16 English LongBench tasks).
- The repository does not include model weights or datasets.