#!/bin/bash
RECIPE_NAME=suffix_random
METHOD_NAME=pi 
# 物理训练长度 (GPU看到的长度)
PHYSICAL_LENGTH=4116
# 逻辑位置上限 (模拟的长距离)
LOGICAL_LENGTH=65536
# Suffix 的固定长度
CHUNK_2_LENGTH=20

WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}

torchrun  --nproc_per_node=8 \
        /root/paddlejob/workspace/env_run/continuous_finetuning/finetune_suffix_random.py  \
        --model_name_or_path "/root/paddlejob/workspace/env_run/Llama-2-7b-hf" \
        --bf16 True \
        --output_dir /root/paddlejob/workspace/env_run/ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${PHYSICAL_LENGTH} \
        --pose_final_position_id ${LOGICAL_LENGTH} \
        --pose_chunk2_length ${CHUNK_2_LENGTH} \
        --use_flash_attn True \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 16 \
        --gradient_checkpointing True \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed /root/paddlejob/workspace/env_run/continuous_finetuning/ds_configs/stage3_offload.json \
        --logging_steps 1 \
        --tf32 True \
        --report_to "none" \
        --use_wandb False \
        --dataset_dir /root/paddlejob/workspace/env_run/data/slim_suffix_4k.jsonl \
        --method_name ${METHOD_NAME} \
        --data_processing_mode "pose_random" \
        --wandb_name ${WANDB_NAME}