#!/bin/bash
# ----------------- Scripts for origin Llama, PI, NTK and YaRN Methos-------------------
RECIPE_NAME=suffix
METHOD_NAME=pi # option:[origin, pi, ntk, yarn]
TRAINING_LENGTH=65536 
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}

CHUNK_2_LENGTH=20 

torchrun  --nproc_per_node=8 \
        /root/paddlejob/workspace/env_run/ct_new1/fine_tune_suffix.py  \
        --model_name_or_path "/root/paddlejob/workspace/env_run/Llama-3-8B" \
        --bf16 True \
        --output_dir /root/paddlejob/workspace/env_run/ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
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
        --deepspeed /root/paddlejob/workspace/env_run/ct_new1/ds_configs/stage3.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 3     \
        --tf32 True \
        --report_to "none" \
        --use_wandb False \
        --dataset_dir /root/paddlejob/workspace/env_run/slim/llama3/slim_8k_1b_llama3_converted.jsonl \
        --method_name ${METHOD_NAME} \
        --data_processing_mode "pose_fixed_split" \
        --pose_chunk2_length ${CHUNK_2_LENGTH} \
        --pose_final_position_id 65535 \
        --wandb_name ${WANDB_NAME}