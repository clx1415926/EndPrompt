import os
import math
import random
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
from datasets import load_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]" 
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    dataset_dir: Optional[str] = field(default=None)
    method_name: Optional[str] = field(default="pi") 
    model_name_or_path: Optional[str] = field(default="")
    model_type: Optional[str] = field(default="llama")
    scaling_type: Optional[str] = field(default="linear")
    scaling_factor: Optional[int] = field(default=1)
    use_wandb: bool = field(default=False)
    wandb_name: str = field(default='test')
    
    # ------------------ 模式控制参数 ------------------
    data_processing_mode: str = field(
        default="default",
        metadata={"help": "模式选择: 'default', 'pose_fixed_split' (旧), 'pose_random' (新)"}
    )
    pose_chunk2_length: int = field(
        default=20,
        metadata={"help": "Suffix 的固定长度 (例如 20)。在 pose_random 模式下也使用此长度。"}
    )
    pose_final_position_id: int = field(
        default=32768,
        metadata={"help": "最大逻辑上下文长度 (位置ID随机采样的上限)"}
    )
    # ------------------------------------------------


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "物理训练长度 (GPU实际输入长度)."},
    )
    use_flash_attn: bool = field(default=True)
    use_full_attn: bool = field(default=False)
    low_rank_training: bool = field(default=False)
    trainable_params: str = field(default="embed,norm")
    mem_freq: int = field(default=63)


# ----------------------------------------------------------------------
#  核心：Suffix 固定长度 + 位置随机 Collator
# ----------------------------------------------------------------------
@dataclass
class DataCollatorForFixedSuffixRandomPos(object):
    """
    实现:
    1. 长度固定：Suffix 长度 = suffix_fixed_len (如20)。
    2. 内容来源：Input 取头部，Suffix 取数据最尾部。
    3. 位置随机：Suffix 的 Position ID 在 [Input结束, logical_max_len] 间随机跳跃。
    """
    tokenizer: transformers.PreTrainedTokenizer
    physical_max_len: int   # 2048
    logical_max_len: int    # 32768
    suffix_fixed_len: int   # 20

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        position_ids_list = []

        for instance in instances:
            raw_ids = instance['input_ids']
            if isinstance(raw_ids, torch.Tensor):
                raw_ids = raw_ids.tolist()
            
            # --- 切分逻辑 ---
            # 目标: Suffix 长度固定为 suffix_fixed_len
            s_len = self.suffix_fixed_len
            # Input 长度 = 总物理容量 - Suffix 长度
            i_len = self.physical_max_len - s_len

            if len(raw_ids) <= self.physical_max_len:
                # 长度不够，不切，直接用
                chunk_ids = raw_ids
                actual_input_len = len(raw_ids)
                actual_suffix_len = 0
            else:
                # 正常情况：取头 + 取尾
                # 头部: 0 到 i_len
                # 尾部:倒数 s_len 到 最后
                chunk_ids = raw_ids[:i_len] + raw_ids[-s_len:]
                actual_input_len = i_len
                actual_suffix_len = s_len

            # --- 位置 ID 生成 ---
            pos_ids = torch.arange(len(chunk_ids), dtype=torch.long)

            if actual_suffix_len > 0:
                # Input 部分位置: 0, 1, ..., i_len-1 (自然对齐，不动)
                
                # Suffix 部分位置: 随机偏移
                # 范围: [紧接Input之后, 逻辑上限 - Suffix长度]
                min_pos = actual_input_len
                max_pos = self.logical_max_len - actual_suffix_len
                
                if max_pos > min_pos:
                    # 随机采样起点
                    random_start = random.randint(min_pos, max_pos)
                    # 计算 Offset
                    offset = random_start - actual_input_len
                    # 应用 Offset 到 suffix 部分
                    pos_ids[actual_input_len:] += offset
            
            input_ids_list.append(torch.tensor(chunk_ids, dtype=torch.long))
            labels_list.append(torch.tensor(chunk_ids, dtype=torch.long))
            position_ids_list.append(pos_ids)

        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = torch.nn.utils.rnn.pad_sequence(
            position_ids_list, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# ------------------ 旧的 Collator (兼容) ------------------
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = [torch.tensor(x) for x in position_ids]
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=0)
        return dict(
            input_ids=input_ids, labels=labels, position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

# ------------------ 预处理函数 ------------------
def preprocess_fixed_split_pose(example, tokenizer, max_length, chunk2_length, final_pos_id):
    # (保持不变) 旧的固定位置逻辑
    input_ids = example['input_ids']
    if len(input_ids) > max_length: input_ids = input_ids[:max_length]
    current_length = len(input_ids)
    if chunk2_length <= 0 or chunk2_length >= current_length:
        return {"input_ids": input_ids, "labels": input_ids, "position_ids": list(range(current_length))}
    split_index = current_length - chunk2_length
    position_ids = list(range(split_index))
    chunk2_start_pos = final_pos_id - chunk2_length + 1
    position_ids.extend(list(range(chunk2_start_pos, final_pos_id + 1)))
    return {"input_ids": input_ids, "labels": input_ids, "position_ids": position_ids}

def tokenize_fn(tokenizer, example):
    # Default 模式
    context_length = tokenizer.model_max_length
    outputs = tokenizer(tokenizer.eos_token.join(example["text"]), truncation=False, return_tensors="pt", pad_to_multiple_of=context_length, padding=True)
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def tokenize_only(tokenizer, example):
    # 新模式专用：只转 ID，不截断
    outputs = tokenizer(tokenizer.eos_token.join(example["text"]), truncation=False, padding=False, return_tensors=None)
    return {"input_ids": outputs["input_ids"]}

def reshape_fn(tokenizer, examples):
    context_length = tokenizer.model_max_length
    examples['input_ids']=torch.tensor(examples['input_ids'])
    return {'input_ids': examples['input_ids'].view(-1, context_length)}

def add_mem_tokens(example, mem_freq, mem_id):
    x = example["input_ids"]
    ret = []
    prev_idx = 0
    for t_idx in range(mem_freq, len(x), mem_freq):
        ret.extend(x[prev_idx:t_idx]); ret.append(mem_id); prev_idx = t_idx
    ret.extend(x[prev_idx:])
    return {"input_ids": ret}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # =========================================================================
    # 模型加载 (PI/NTK/YaRN/LongLoRA 逻辑完全保留)
    # =========================================================================
    if model_args.method_name == "pi":
        from models.modeling_llama import LlamaForCausalLM
        print('training pi')
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
        
        # 确定 Scaling 目标
        if model_args.data_processing_mode == "pose_random":
            scaling_target = model_args.pose_final_position_id
        else:
            scaling_target = training_args.model_max_length

        orig_ctx_len = getattr(config, "max_position_embeddings", None) 
        config.max_position_embeddings = scaling_target
        if orig_ctx_len and scaling_target > orig_ctx_len:
            scaling_factor = float(math.ceil(scaling_target / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(f"PI Scaling: {config.rope_scaling}", flush=True)

        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config, cache_dir=training_args.cache_dir,
            use_flash_attention_2=True, torch_dtype=torch.bfloat16,
        )

    elif model_args.method_name == "ntk":
        print('training ntk-dynamic')
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
        from models.modeling_llama import LlamaForCausalLM
        
        if model_args.data_processing_mode == "pose_random":
            scaling_target = model_args.pose_final_position_id
        else:
            scaling_target = training_args.model_max_length

        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and scaling_target > orig_ctx_len:
            scaling_factor = float(math.ceil(scaling_target / orig_ctx_len / 2))
            config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
            print(f"NTK Scaling: {config.rope_scaling}", flush=True)
        config.max_position_embeddings = scaling_target
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config, cache_dir=training_args.cache_dir,
            use_flash_attention_2=True, torch_dtype=torch.bfloat16,
        )

    elif model_args.method_name == "yarn":
        print('training yarn')
        from models.llama_yarn.modeling_llama_yarn import LlamaForCausalLM
        from models.llama_yarn.configuration_llama import LlamaConfig
        
        if model_args.data_processing_mode == "pose_random":
            scaling_target = model_args.pose_final_position_id
        else:
            scaling_target = training_args.model_max_length
            
        original_max_position_embeddings = 4096
        scaling_factor = float(math.ceil(scaling_target / original_max_position_embeddings))
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        config.rope_scaling = {"type": "yarn", "factor": scaling_factor, "original_max_position_embeddings": original_max_position_embeddings}
        config.rope_theta = 10000
        config.max_position_embeddings = scaling_target
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config, use_flash_attention_2=True
        )

    elif model_args.method_name == "longlora":
        from models.llama_longlora.llama_attn_replace import replace_llama_attn
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
        
        if model_args.data_processing_mode == "pose_random":
             scaling_target = model_args.pose_final_position_id
        else:
             scaling_target = training_args.model_max_length

        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and scaling_target > orig_ctx_len:
            scaling_factor = float(math.ceil(scaling_target / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(config.rope_scaling, flush=True)
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, config=config, cache_dir=training_args.cache_dir, torch_dtype=torch.bfloat16,
        )
        if training_args.low_rank_training:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]
            config = LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, config)
            [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    elif model_args.method_name == "landmark":
        from models.llama_landmark.llama_mem import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, mem_freq=training_args.mem_freq, include_landmark_in_loss=True)

    elif model_args.method_name == "origin":
        print('training origin llama')
        model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, use_flash_attention_2=True, torch_dtype=torch.bfloat16)
    
    # =========================================================================
    # Tokenizer & Data
    # =========================================================================
    tokenizer_max_len = model_args.pose_final_position_id if "pose" in model_args.data_processing_mode else training_args.model_max_length
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=tokenizer_max_len, padding_side="right", use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None: special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None: special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None: special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None: special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if model_args.method_name == "landmark": special_tokens_dict["additional_special_tokens"] = ["<landmark>"]
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    
    rank = int(os.environ.get('RANK', -1))
    if rank > 0: barrier()
    dataset = load_dataset('json', data_files=model_args.dataset_dir)
    if rank == 0: barrier()
    
    # =========================================================================
    # 数据处理选择 (NEW: pose_random)
    # =========================================================================

    if model_args.data_processing_mode == "pose_random":
        print(f"模式: pose_random")
        print(f" - 物理长度(Block): {training_args.model_max_length}")
        print(f" - Suffix固定长度: {model_args.pose_chunk2_length}")
        print(f" - 位置随机范围: Input结束 ~ {model_args.pose_final_position_id}")
        
        # 1. 预处理：只做 Tokenization，不截断，保留完整原文以便 Collator 切尾巴
        if "input_ids" not in dataset['train'].column_names:
            dataset = dataset.map(
                partial(tokenize_only, tokenizer),
                batched=False, 
                num_proc=8,
                remove_columns=["text"] if "text" in dataset['train'].column_names else []
            )
        
        # 2. 调用新 Collator
        data_collator = DataCollatorForFixedSuffixRandomPos(
            tokenizer=tokenizer,
            physical_max_len=training_args.model_max_length,         
            logical_max_len=model_args.pose_final_position_id,
            suffix_fixed_len=model_args.pose_chunk2_length  # <--- 使用传入的固定长度 (如 20)
        )

    elif model_args.data_processing_mode == "pose_fixed_split":
        # 旧模式
        print(f"模式: pose_fixed_split (Suffix位置固定在 {model_args.pose_final_position_id})")
        dataset = dataset.map(
            partial(preprocess_fixed_split_pose, tokenizer=tokenizer, max_length=training_args.model_max_length, 
                    chunk2_length=model_args.pose_chunk2_length, final_pos_id=model_args.pose_final_position_id),
            batched=False, num_proc=8, remove_columns=dataset['train'].column_names
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    elif model_args.method_name == "landmark":
        # Landmark 逻辑
        mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
        model.set_mem_id(mem_id)
        remove_columns = [col for col in dataset['train'].column_names if col != 'input_ids']
        dataset = dataset.map(partial(reshape_fn,tokenizer), batched=True, num_proc=8, remove_columns=remove_columns)
        dataset = dataset.map(partial(add_mem_tokens, mem_freq=training_args.mem_freq, mem_id=mem_id), batched=False, num_proc=8)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
    else: # Default
        print("模式: default (Continuous)")
        dataset = dataset.map(partial(tokenize_fn, tokenizer), batched=True, num_proc=8, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    
    if model_args.use_wandb:
        wandb.init(project='long_extension', name=model_args.wandb_name, sync_tensorboard=False, job_type="CleanRepo", group=model_args.wandb_name)

    if model_args.method_name != "landmark":
        model.config.use_cache = False         

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset['train'], eval_dataset=None, data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    if model_args.method_name == "longlora" and training_args.low_rank_training:
        save_finetuned_model_dir = os.path.join(training_args.output_dir, 'finetuned_model')
        model.base_model.save_pretrained(save_finetuned_model_dir)

if __name__ == "__main__":
    train()