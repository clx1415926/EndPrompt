import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import wandb


from datasets import load_dataset, load_from_disk

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
# Llama3使用原生token，不需要定义EOS/BOS/UNK


@dataclass
class ModelArguments:
    dataset_dir: Optional[str] = field(default=None)
    method_name: Optional[str] = field(default="pi") # pi, ntk-dynamic, longlora, longlora-ft, yarn, landmark
    model_name_or_path: Optional[str] = field(default="")
    model_type: Optional[str] = field(default="llama")
    scaling_type: Optional[str] = field(default="linear")
    scaling_factor: Optional[int] = field(default=1)
    use_wandb: bool = field(
        default=False,
    )
    wandb_name: str = field(
        default='test',
    )
    # ------------------ 修改后的参数 ------------------
    data_processing_mode: str = field(
        default="default",
        metadata={"help": "数据处理模式: 'default' (连续) or 'pose_fixed_split' (不连续)"}
    )
    # 将 split_index 替换为 chunk2_length
    pose_chunk2_length: int = field(
        default=-1,
        metadata={"help": "用于 PoSE 模式的第二块的固定长度 (例如: 20)，从序列末尾计算"}
    )
    pose_final_position_id: int = field(
        default=32767,
        metadata={"help": "序列中最后一个 token 对应的 position_id"}
    )
    # ----------------------------------------------


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=False,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    mem_freq: int = field(default=63)

# ------------------ (DataCollatorForSupervisedDataset 保持不变) ------------------
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 从 instance 字典中提取 'input_ids', 'labels', 'position_ids'
        input_ids, labels, position_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids"))
        
        # 填充 input_ids
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        # 填充 labels
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # 填充 position_ids
        position_ids = [torch.tensor(x) for x in position_ids]
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=0)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
# ---------------------------------------------------------------------------------

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

# ------------------ 针对您的场景更新的预处理函数 ------------------
def preprocess_fixed_split_pose(example, tokenizer, max_length, chunk2_length, final_pos_id):
    """
    根据固定的 'chunk2_length' (从末尾计算) 创建不连续的 position_ids。
    假设 'example' 已经有一个 'input_ids' 字段。
    """
    # 1. 获取 input_ids 并截断到最大长度
    input_ids = example['input_ids']
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    
    current_length = len(input_ids)
    
    # 2. 验证 chunk2_length 是否有效
    if chunk2_length <= 0 or chunk2_length >= current_length:
        # 如果 chunk2_length 无效 (例如, 序列太短),
        # 作为后备，创建连续的 position_ids
        print(f"警告: chunk2_length {chunk2_length} 对序列长度 {current_length} 无效。将使用连续 position_ids。")
        position_ids = list(range(current_length))
        labels = list(input_ids)
        return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}

    # 3. 根据 chunk2_length 计算分割点
    split_index = current_length - chunk2_length
    chunk1_len = split_index
    # ( chunk2_len 已经由参数传入 )
    
    # 4. 生成不连续的 position_ids
    position_ids = []
    
    # 块 1: [0, 1, ..., split_index - 1]
    if chunk1_len > 0:
        position_ids.extend(list(range(chunk1_len)))
    
    # 块 2: [..., final_pos_id]
    # 计算块2的起始 position_id
    # 例如: final_pos_id=32767, chunk2_length=20
    # start_pos = 32767 - 20 + 1 = 32758
    # 范围: [32758, ..., 32767] (长度为 20)
    chunk2_start_pos = final_pos_id - chunk2_length + 1
    position_ids.extend(list(range(chunk2_start_pos, final_pos_id + 1)))
    
    # 5. 创建 labels (与 input_ids 相同)
    labels = list(input_ids)
    
    # 6. 健全性检查
    assert len(input_ids) == len(position_ids) == len(labels), \
        f"长度不匹配: input_ids={len(input_ids)}, position_ids={len(position_ids)}, labels={len(labels)}"
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids
    }
# ---------------------------------------------------------------


# 这是原始的 tokenize_fn，用于 'default' 模式
def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}



def reshape_fn(tokenizer, examples):
    context_length = tokenizer.model_max_length
    examples['input_ids']=torch.tensor(examples['input_ids'])
    return {'input_ids': examples['input_ids'].view(-1, context_length)}


def add_mem_tokens(example, mem_freq, mem_id):
    x = example["input_ids"]
    ret = []
    prev_idx = 0
    for t_idx in range(mem_freq, len(x), mem_freq):
        ret.extend(x[prev_idx:t_idx])
        ret.append(mem_id)
        prev_idx = t_idx
    ret.extend(x[prev_idx:])
    # drop attention_mask
    return {"input_ids": ret}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # --- 模型加载逻辑 (与PI/NTK/YaRN相关, 保持不变) ---
    if model_args.method_name == "pi":
        from models.modeling_llama import LlamaForCausalLM
        print('training pi')
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        context_size = training_args.model_max_length
        orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models, 8192 for LLaMA3 models
        config.max_position_embeddings = context_size
        
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(f"RoPE scaling config: {config.rope_scaling}, orig_ctx_len: {orig_ctx_len}, context_size: {context_size}", flush=True)
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
        )
    elif model_args.method_name == "ntk":
        print('training ntk-dynamic')
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        context_size = training_args.model_max_length
        from models.modeling_llama import LlamaForCausalLM
        orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models, 8192 for LLaMA3 models
        if orig_ctx_len and context_size > orig_ctx_len:
            # set the ntk-dynamic half the scale
            scaling_factor = float(math.ceil(context_size / orig_ctx_len / 2))
            config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
            print(f"RoPE scaling config: {config.rope_scaling}, orig_ctx_len: {orig_ctx_len}, context_size: {context_size}", flush=True)
        config.max_position_embeddings = context_size
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
        )

    elif model_args.method_name == "yarn":
        print('training yarn')
        from models.modeling_llama import LlamaForCausalLM
        from models.llama_yarn.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
        config = config_cls.from_pretrained(model_args.model_name_or_path)
        # Detect original max position embeddings from config (4096 for Llama2, 8192 for Llama3)
        original_max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
        # Preserve original rope_theta if exists (10000 for Llama2, 500000 for Llama3)
        original_rope_theta = getattr(config, "rope_theta", 10000)
        context_size = training_args.model_max_length
        scaling_factor = float(math.ceil(context_size / original_max_position_embeddings))
        config.rope_scaling = {
            "type": "yarn",
            "factor": scaling_factor,
            "original_max_position_embeddings": original_max_position_embeddings
        }
        config.rope_theta = original_rope_theta  # Use original rope_theta from model config
        config.max_position_embeddings = training_args.model_max_length
        print(f"YaRN config: scaling_factor={scaling_factor}, original_max_pos={original_max_position_embeddings}, rope_theta={config.rope_theta}", flush=True)

        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            config=config,
            use_flash_attention_2=True
        )
    elif model_args.method_name == "longlora":
        # replace sparse shift attention for longlora
        from models.llama_longlora.llama_attn_replace import replace_llama_attn
        if training_args.low_rank_training:
            print('training longlora with lora and sparse shift attention')
        else:
            print('training longlora with full-finetuning and sparse shift attention')
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)
                # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        context_size = training_args.model_max_length
        orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models, 8192 for LLaMA3 models
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(f"RoPE scaling config: {config.rope_scaling}, orig_ctx_len: {orig_ctx_len}, context_size: {context_size}", flush=True)
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
        )

        # use lora with "embed,norm" or use fullfinetuning
        if training_args.low_rank_training:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=targets,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            # enable trainable params
            [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
    elif model_args.method_name == "landmark":
        from models.llama_landmark.llama_mem import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            mem_freq=training_args.mem_freq,
            include_landmark_in_loss=True
        )
    elif model_args.method_name == "origin":
        print('training origin llama')
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
        )
    
    # --- (Tokenizer 和 Special Tokens 逻辑保持不变) ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    
    # 检测模型类型（Llama3词汇表>100K）
    is_llama3 = len(tokenizer) > 100000
    
    if is_llama3:
        # Llama3：只添加PAD token，保留原生BOS/EOS
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        print(f"✓ 检测到Llama3 (vocab_size={len(tokenizer)})")
        print(f"  保留原生BOS: {tokenizer.bos_token}")
        print(f"  保留原生EOS: {tokenizer.eos_token}")
    else:
        # Llama2：使用传统逻辑
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "</s>"
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<s>"
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = "<unk>"
        print(f"✓ 检测到Llama2 (vocab_size={len(tokenizer)})")
    
    if model_args.method_name == "landmark":
        mem_token = "<landmark>"
        special_tokens_dict["additional_special_tokens"] = [mem_token]

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # 验证tokenizer配置
    print("\n" + "="*60)
    print("📋 Tokenizer配置验证")
    print("="*60)
    print(f"词汇表大小: {len(tokenizer)}")
    print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    if tokenizer.unk_token:
        print(f"UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"新增特殊token数: {len(special_tokens_dict)}")
    print("="*60 + "\n")
    
    # Llama3词汇表检查
    if is_llama3 and len(tokenizer) != 128257:
        print(f"⚠️  警告：Llama3词汇表大小异常！预期128257，实际{len(tokenizer)}")
    # --- (Barrier 和 load_dataset 逻辑保持不变) ---
    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset = load_dataset('json',data_files=model_args.dataset_dir,)
    
    
    if rank == 0:
        barrier()

    print(dataset)
    
    # ------------------ 修改的数据处理和收集器逻辑 ------------------

    # 根据传入的参数选择数据处理方式
    if model_args.data_processing_mode == "pose_fixed_split":
        print(f"使用 'pose_fixed_split' 数据处理模式。")
        # 更新: 打印 chunk2_length
        print(f"  Chunk 2 Length (from end): {model_args.pose_chunk2_length}")
        print(f"  Final Position ID: {model_args.pose_final_position_id}")
        
        # 更新: 检查 chunk2_length
        if model_args.pose_chunk2_length <= 0:
            raise ValueError("'pose_fixed_split' 模式需要设置 '--pose_chunk2_length' 为正数 (例如 20)。")
        
        # 假设数据集已有 'input_ids' 字段
        if "input_ids" not in dataset['train'].column_names:
            raise ValueError("'pose_fixed_split' 模式需要数据集中有 'input_ids' 字段。")
            
        column_names = dataset['train'].column_names
        
        dataset = dataset.map(
            partial(
                # 更新: 传递新参数
                preprocess_fixed_split_pose, 
                tokenizer=tokenizer, 
                max_length=training_args.model_max_length,
                chunk2_length=model_args.pose_chunk2_length,
                final_pos_id=model_args.pose_final_position_id
            ),
            batched=False, # 我们的函数一次处理一个样本
            num_proc=8, # 可以根据需要调整
            remove_columns=column_names # 移除所有旧字段，只保留新字段
        )
        # 使用新的数据收集器
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
    elif model_args.method_name == "landmark":
        print("使用 'landmark' 数据处理模式。")
        mem_id = tokenizer.convert_tokens_to_ids(mem_token)
        model.set_mem_id(mem_id)
        remove_columns = [col for col in dataset['train'].column_names if col != 'input_ids']
        dataset = dataset.map(partial(reshape_fn,tokenizer),batched=True, num_proc=8, remove_columns=remove_columns)
        dataset = dataset.map(
            partial(
                add_mem_tokens, 
                mem_freq=training_args.mem_freq, 
                mem_id=mem_id
            ), batched=False, num_proc=8)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
    else: # 默认的、连续的
        print("使用 'default' 数据处理模式 (标准 tokenization)。")
        if "text" not in dataset['train'].column_names:
            raise ValueError("Default 模式需要数据集中有 'text' 字段。")
            
        dataset = dataset.map(
            partial(tokenize_fn, tokenizer),
            batched=True,
            num_proc=8,
            remove_columns=["text"]
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # -------------------------------------------------------------
        
    # --- (Wandb, Gradient Checkpointing, Trainer 逻辑保持不变) ---
    if model_args.use_wandb:
        project_name = f'long_extension'
        wandb.init(project=project_name, entity='', name=model_args.wandb_name, sync_tensorboard=False,
                job_type="CleanRepo", group=model_args.wandb_name)

    
    if model_args.method_name != "landmark":
        model.config.use_cache = False         # required for gradient checkpointing
        # model.enable_input_require_grads()     # required for gradient checkpointing
        # model.gradient_checkpointing_enable()  # enable gradient checkpointing
    
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=None,
        data_collator=data_collator) # <-- collator 已经被正确设置
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    # --- (LongLoRA 保存逻辑保持不变) ---
    if model_args.method_name == "longlora" and training_args.low_rank_training:
        print('saveing pytorch_model.bin with lora')
        save_finetuned_model_dir = os.path.join(training_args.output_dir, 'finetuned_model')
        if save_finetuned_model_dir is not None:
            os.makedirs(save_finetuned_model_dir, exist_ok=True)
        model.base_model.save_pretrained(save_finetuned_model_dir)
        # save merged model
        model = model.merge_and_unload()
        save_finetuned_model_merged_dir = os.path.join(training_args.output_dir, 'finetuned_model_merged')
        if save_finetuned_model_merged_dir is not None:
            os.makedirs(save_finetuned_model_merged_dir, exist_ok=True)
        model.save_pretrained(save_finetuned_model_merged_dir)


if __name__ == "__main__":
    train()