#!/bin/bash
# ----------------- EndPrompt (ET) main training entry -----------------
# Physical input: 8182 context tokens + 10 cue tokens (EP_1) = 8192 tokens.
# Logical window: 65536 with linear PI (s = 8); the cue offset Delta is sampled from
# the precomputed D-optimal cumulative table (skip_design/gate1_w1_p_star_cdf.npy).
RECIPE_NAME=suffix
TRAINING_LENGTH=8192           # physical input length (a + b)
TARGET_LENGTH=65536            # logical target window L
WANDB_NAME=${RECIPE_NAME}_${TARGET_LENGTH}

CHUNK_2_LENGTH=10              # cue token count b (EP_1 tokenizes to 10 tokens)
CUE_TEXT="This is the end of text, please pay attention here"
SKIP_CDF=skip_design/gate1_w1_p_star_cdf.npy

torchrun  --nproc_per_node=8 \
        fine_tune_suffix.py  \
        --model_name_or_path "/path/to/Llama-3-8B" \
        --bf16 True \
        --output_dir ./ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --target_length ${TARGET_LENGTH} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --gradient_checkpointing True \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 30 \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed ds_configs/stage3.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 3     \
        --tf32 True \
        --report_to "none" \
        --dataset_dir /path/to/slim_8k_1b_llama3_converted.jsonl \
        --pose_chunk2_length ${CHUNK_2_LENGTH} \
        --pose_final_position_id 65535 \
        --et_cue_text "${CUE_TEXT}" \
        --et_skip_cdf ${SKIP_CDF}