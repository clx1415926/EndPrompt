import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from torch.distributed import barrier

from datasets import load_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
# Llama3使用原生token，不需要定义EOS/BOS/UNK


@dataclass
class ModelArguments:
    dataset_dir: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="")
    target_length: int = field(
        default=65536,
        metadata={"help": "逻辑目标窗口 L;决定 RoPE 配置 (PI scale = ceil(L / 模型原窗口))。"}
    )
    pose_chunk2_length: int = field(
        default=10,
        metadata={"help": "terminal cue 的 token 数 b (EP_1 分词后, 默认 10)。"}
    )
    pose_final_position_id: int = field(
        default=65535,
        metadata={"help": "目标窗口内最后一个位置 (L-1)。"}
    )
    et_cue_text: str = field(
        default="This is the end of text, please pay attention here",
        metadata={"help": "固定的 terminal cue 文本 (论文默认 EP_1);需恰好分词为 pose_chunk2_length 个 token。"}
    )
    et_skip_cdf: str = field(
        default="skip_design/gate1_w1_p_star_cdf.npy",
        metadata={"help": "D-optimal 偏移分布的累积概率表 (npy, 单调不减, 末项为 1.0, 长度为 L-a-b+1)。"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=8192,
        metadata={"help": "物理输入长度 (context a + cue b); 逻辑窗口由 --target_length 决定。"},
    )


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

# ------------------ ET (最终方案) 样本构造 ------------------
def preprocess_et(example, idx, cue_ids, context_length, pad_id, final_pos_id, skip_cdf, base_seed):
    """
    按论文构造 ET 训练样本: 完整短 context 后接固定的 terminal cue,
    仅对 cue 做整块位置偏移 (p_l = l for l<a; p_l = l+Delta for a<=l<a+b),
    偏移 Delta 从 D-optimal 分布的累积表逆 CDF 采样。
    """
    input_ids = example['input_ids']
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if len(input_ids) < context_length:
        input_ids = input_ids + [pad_id] * (context_length - len(input_ids))
    context = input_ids[:context_length]

    b = len(cue_ids)
    delta_max = final_pos_id + 1 - context_length - b  # L - a - b
    rng = np.random.default_rng((base_seed + idx) % (2**32))
    delta = int(np.searchsorted(skip_cdf, float(rng.random()), side="right"))
    assert 0 <= delta <= delta_max, f"采样的偏移越界: delta={delta}, delta_max={delta_max}"

    seq = context + list(cue_ids)
    position_ids = list(range(context_length))
    position_ids.extend(range(context_length + delta, context_length + delta + b))
    return {"input_ids": seq, "labels": seq, "position_ids": position_ids}
# ---------------------------------------------------------------


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # --- 模型加载 (linear PI, 与论文一致) ---
    from models.modeling_llama import LlamaForCausalLM
    print('training pi')
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # 原窗口 (LLaMA-3: 8192)
    config.max_position_embeddings = model_args.target_length
    if orig_ctx_len and model_args.target_length > orig_ctx_len:
        scaling_factor = float(math.ceil(model_args.target_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    print(f"RoPE scaling config: {config.rope_scaling}, orig_ctx_len: {orig_ctx_len}, target_length: {model_args.target_length}", flush=True)
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
    )
    
    # --- Tokenizer ---
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

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # 验证tokenizer配置
    print(f"词汇表大小: {len(tokenizer)}; PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # --- 数据加载 ---
    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset = load_dataset('json', data_files=model_args.dataset_dir)
    
    if rank == 0:
        barrier()

    print(dataset)
    
    # --- ET 样本构造 ---
    if model_args.pose_chunk2_length <= 0:
        raise ValueError("'--pose_chunk2_length' 需为正数 (cue 的 token 数, 默认 10)。")
    if "input_ids" not in dataset['train'].column_names:
        raise ValueError("数据集中需要 'input_ids' 字段 (预分词)。")

    cue_ids = tokenizer(model_args.et_cue_text, add_special_tokens=False)["input_ids"]
    if len(cue_ids) != model_args.pose_chunk2_length:
        raise ValueError(
            f"cue 文本分词为 {len(cue_ids)} 个 token, 与 --pose_chunk2_length={model_args.pose_chunk2_length} 不符。"
            f"请调整 --et_cue_text 或 --pose_chunk2_length, 并确保 skip 表按 (a, b) 重新计算。"
        )
    context_length = training_args.model_max_length - model_args.pose_chunk2_length
    if context_length < 1:
        raise ValueError("--model_max_length 必须大于 --pose_chunk2_length。")

    skip_cdf = np.load(model_args.et_skip_cdf)
    delta_max = model_args.pose_final_position_id + 1 - context_length - len(cue_ids)
    if skip_cdf.ndim != 1 or skip_cdf.shape[0] != delta_max + 1 or skip_cdf[-1] != 1.0 or (np.diff(skip_cdf) < 0).any():
        raise ValueError(
            f"--et_skip_cdf 应为长度 {delta_max + 1}、单调不减且末项为 1 的一维累积概率表。"
        )
    print(f"  Context: {context_length} tokens, Cue: {len(cue_ids)} tokens, "
          f"target L: {model_args.pose_final_position_id + 1}, delta_max: {delta_max}")
    print(f"  Cue text: '{model_args.et_cue_text}'")

    column_names = dataset['train'].column_names
    dataset = dataset.map(
        partial(
            preprocess_et,
            cue_ids=cue_ids,
            context_length=context_length,
            pad_id=tokenizer.pad_token_id,
            final_pos_id=model_args.pose_final_position_id,
            skip_cdf=skip_cdf,
            base_seed=training_args.seed,
        ),
        with_indices=True,
        batched=False,
        num_proc=8,
        remove_columns=column_names,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # --- 训练 ---
    model.config.use_cache = False  # required for gradient checkpointing
    
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=None,
        data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()