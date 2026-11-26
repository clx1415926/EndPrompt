#!/bin/bash
# ----------------- Uniform-offset ET ablation (paper ablation) -----------------
# Same construction as finetune_main_suffix.sh (fixed EP_1 cue, a=8182, b=10,
# PI s=8, L=65536), but Delta is drawn uniformly over [0, L-a-b] instead of from
# the D-optimal table.
RECIPE_NAME=suffix
METHOD_NAME=pi # option:[origin, pi, ntk, yarn]
TRAINING_LENGTH=8192           # physical input length (a + b)
TARGET_LENGTH=65536            # logical target window L
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TARGET_LENGTH}_uniform

CHUNK_2_LENGTH=10              # cue token count b (EP_1 tokenizes to 10 tokens)
CUE_TEXT="This is the end of text, please pay attention here"

torchrun  --nproc_per_node=8 \
        fine_tune_suffix.py  \
        --model_name_or_path "/path/to/Llama-3-8B" \
        --bf16 True \
        --output_dir ./ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --target_length ${TARGET_LENGTH} \
        --use_flash_attn True \
        --low_rank_training False \
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
        --use_wandb False \
        --dataset_dir /path/to/slim_8k_1b_llama3_converted.jsonl \
        --method_name ${METHOD_NAME} \
        --data_processing_mode "et" \
        --pose_chunk2_length ${CHUNK_2_LENGTH} \
        --pose_final_position_id 65535 \
        --et_cue_text "${CUE_TEXT}" \
        --et_skip_mode uniform \
        --wandb_name ${WANDB_NAME}