import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
# QLoRA/BitsAndBytesConfig 已被删除
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import wandb


from datasets import load_dataset, load_from_disk

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]" 
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


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
    
    # --- ARGUMENTS FOR DISTILLATION (TOP-K) ---
    distil_alpha: float = field(
        default=0.3,
        metadata={"help": "Weight for the standard Language Modeling loss. The KD loss weight will be (1 - distil_alpha)."}
    )
    distil_temp: float = field(
        default=2.0,
        metadata={"help": "Temperature for the Knowledge Distillation loss."}
    )
    distil_top_k: int = field(
        default=256,
        metadata={"help": "Compute KD loss only on the top-k teacher logits to save memory."}
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

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


# --- CUSTOM TRAINER (WITH TOP-K) ---
class CustomTrainer(Trainer):
    
    @staticmethod
    def _my_concat(l1: torch.Tensor, l2: torch.Tensor, dim: int = -2) -> torch.Tensor:
        """
        按照奇偶交叉，把两个tensor的dim维度拼起来
        """
        s1 = l1.shape[dim]
        s2 = l2.shape[dim]
        s_out = s1 + s2

        out_shape = list(l1.shape)
        if s1 == 0:
            out_shape = list(l2.shape)
        out_shape[dim] = s_out
        
        out = torch.empty(*out_shape, device=l1.device, dtype=l1.dtype)

        indices = [slice(None)] * l1.ndim
        
        if s1 > 0:
            indices[dim] = slice(0, s_out, 2) # Even indices
            out[tuple(indices)] = l1
        
        if s2 > 0:
            indices[dim] = slice(1, s_out, 2) # Odd indices
            out[tuple(indices)] = l2
            
        return out

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation with self-distillation using Top-K.
        """
        # --- Get Hyperparameters ---
        alpha = self.args.distil_alpha
        temp = self.args.distil_temp
        k = self.args.distil_top_k
        
        # --- 1. Student Pass (Full Context) ---
        outputs = model(**inputs)
        lm_loss = outputs.loss
        logits_student = outputs.logits  # [B, S, V]

        # --- 2. Teacher Pass (Half Context) ---
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        x1_ids = input_ids[..., ::2]
        x2_ids = input_ids[..., 1::2]
        x1_mask = attention_mask[..., ::2] if attention_mask is not None else None
        x2_mask = attention_mask[..., 1::2] if attention_mask is not None else None
        
        with torch.no_grad():
            logits1 = model(input_ids=x1_ids, attention_mask=x1_mask).logits
            logits2 = model(input_ids=x2_ids, attention_mask=x2_mask).logits
        
        logits_teacher = self._my_concat(logits1, logits2, dim=-2) # [B, S, V]
        del logits1, logits2

        # --- 4. Top-K KD Loss Calculation ---
        
        labels = inputs.get("labels")
        if labels is None:
            labels = input_ids
        
        labels_for_kd = labels[..., 1:] # Shifted labels
        loss_mask = (labels_for_kd != IGNORE_INDEX) # [B, S-1]
        
        student_logits_for_kd = logits_student[..., :-1, :] # View, [B, S-1, V]
        teacher_logits_for_kd = logits_teacher[..., :-1, :] # View, [B, S-1, V]
        del logits_teacher 

        # --- TOP-K LOGIC ---
        with torch.no_grad():
            top_k_teacher_logits, top_k_indices = torch.topk(
                teacher_logits_for_kd, k=k, dim=-1
            ) # [B, S-1, k]
        del teacher_logits_for_kd 

        top_k_student_logits = torch.gather(
            student_logits_for_kd, dim=-1, index=top_k_indices
        ) # [B, S-1, k]

        log_p_student_topk = F.log_softmax(top_k_student_logits / temp, dim=-1)
        del top_k_student_logits 

        with torch.no_grad():
            p_teacher_topk = F.softmax(top_k_teacher_logits / temp, dim=-1)
        del top_k_teacher_logits 

        kd_loss_fn = torch.nn.KLDivLoss(reduction='none')
        kd_loss_unreduced = kd_loss_fn(log_p_student_topk, p_teacher_topk) * (temp * temp)
        del log_p_student_topk, p_teacher_topk

        kd_loss_per_token = kd_loss_unreduced.sum(dim=-1) # [B, S-1]
        del kd_loss_unreduced

        kd_loss = (kd_loss_per_token * loss_mask).sum() / loss_mask.sum()
        del kd_loss_per_token, loss_mask
        
        # --- 5. Combine Losses ---
        loss = alpha * lm_loss + (1 - alpha) * kd_loss
        
        return (loss, outputs) if return_outputs else loss
# --- END OF CUSTOM TRAINER CLASS ---


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # --- MODEL LOADING (NO QLoRA) ---
    
    if model_args.method_name == "pi":
        print('training pi')
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        context_size = training_args.model_max_length
        orig_ctx_len = getattr(config, "max_position_embeddings", None) 
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(config.rope_scaling, flush=True)
            
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
            # QLoRA parameters removed
        )
        
    elif model_args.method_name == "ntk":
        print('training ntk-dynamic') # <--- MODIFIED (Removed "with QLoRA")
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        context_size = training_args.model_max_length
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len / 2))
            config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
            print(config.rope_scaling, flush=True)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
            # QLoRA parameters removed
        )
        
    elif model_args.method_name == "yarn":
        print('training yarn') # <--- MODIFIED (Removed "with QLoRA")
        from models.modeling_llama import LlamaForCausalLM
        from models.llama_yarn.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
        original_max_position_embeddings = 4096
        context_size = training_args.model_max_length
        scaling_factor = float(math.ceil(context_size / original_max_position_embeddings))
        config = config_cls.from_pretrained(model_args.model_name_or_path)
        config.rope_scaling = {
            "type": "yarn",
            "factor": scaling_factor,
            "original_max_position_embeddings": original_max_position_embeddings
        }
        config.rope_theta = 10000
        config.max_position_embeddings = training_args.model_max_length

        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            config=config,
            use_flash_attention_2=True,
            # QLoRA parameters removed
        )
        
    elif model_args.method_name == "longlora":
        from models.llama_longlora.llama_attn_replace import replace_llama_attn
        if training_args.low_rank_training:
            print('training longlora with lora and sparse shift attention')
        else:
            print('training longlora with full-finetuning and sparse shift attention')
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        context_size = training_args.model_max_length
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(config.rope_scaling, flush=True)
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            # QLoRA parameters removed
        )

        # --- Reverted to original LoRA logic ---
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
        print('training origin llama') # <--- MODIFIED (Removed "with QLoRA")
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
            # QLoRA parameters removed
        )
        
    # --- LoRA/PEFT logic is now ONLY in the `longlora` branch ---
    # (Or other branches if you set --low_rank_training True)
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if model_args.method_name == "landmark":
        mem_token = "<landmark>"
        special_tokens_dict["additional_special_tokens"] = [mem_token]

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset = load_dataset('json',data_files=model_args.dataset_dir,)
    
    
    if rank == 0:
        barrier()

    print(dataset)

    # add special config for landmark
    if model_args.method_name == "landmark":
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
        
    if model_args.use_wandb:
        project_name = f'long_extension'
        wandb.init(project=project_name, entity='', name=model_args.wandb_name, sync_tensorboard=False,
                job_type="CleanRepo", group=model_args.wandb_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    if model_args.method_name != "landmark":
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=None,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    # save model for longlora
    if model_args.method_name == "longlora" and training_args.low_rank_training:
        print('saving pytorch_model.bin with lora')
        save_finetuned_model_dir = os.path.join(training_args.output_dir, 'finetuned_model')
        if save_finetuned_model_dir is not None:
            os.makedirs(save_finetuned_model_dir, exist_ok=True)
        
        # This will now save the LoRA adapter
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()