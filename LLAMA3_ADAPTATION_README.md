# Llama3 8B 适配说明

## 修改概述

本文档说明了将训练脚本从 Llama2 7B 适配到 Llama3 8B 所做的修改。

## 主要差异对比

### 1. 模型配置差异

| 配置项 | Llama2 7B | Llama3 8B | 影响 |
|--------|-----------|-----------|------|
| `max_position_embeddings` | 4096 | 8192 | RoPE scaling 计算基准 |
| `rope_theta` | 10000 (默认) | 500000 | RoPE 频率基数 |
| `vocab_size` | 32000 | 128256 | 词表大小 |
| `intermediate_size` | 11008 | 14336 | FFN 中间层大小 |
| `num_key_value_heads` | 32 (MHA) | 8 (GQA) | 注意力机制 |
| `bos_token_id` | 1 | 128000 | 起始token |
| `eos_token_id` | 2 | 128001 | 结束token |

### 2. 代码修改详情

#### 2.1 Position Interpolation (PI) 方法

**修改位置**: `fine_tune_suffix.py` 第 228-249 行

**修改内容**:
- 更新注释，说明 `orig_ctx_len` 对 Llama2 是 4096，对 Llama3 是 8192
- 增强日志输出，显示 `orig_ctx_len` 和 `context_size` 的具体值

**原因**: 
- Llama3 的原始上下文长度是 8192，与 Llama2 的 4096 不同
- 当训练长度为 65536 时，Llama3 的 scaling factor 为 ceil(65536/8192) = 8
- 而 Llama2 的 scaling factor 为 ceil(65536/4096) = 16
- 更详细的日志有助于调试和验证配置

#### 2.2 NTK-Dynamic 方法

**修改位置**: `fine_tune_suffix.py` 第 250-271 行

**修改内容**:
- 更新注释，说明 `orig_ctx_len` 的不同
- 增强日志输出

**原因**:
- NTK-dynamic 方法使用 `scaling_factor / 2`
- 对 Llama3: scaling_factor = ceil(65536/8192/2) = 4
- 对 Llama2: scaling_factor = ceil(65536/4096/2) = 8
- 正确识别原始长度对计算 scaling factor 至关重要

#### 2.3 YaRN 方法

**修改位置**: `fine_tune_suffix.py` 第 273-296 行

**修改内容**:
- 从配置中动态读取 `original_max_position_embeddings`（不再硬编码为 4096）
- 从配置中动态读取 `rope_theta`（保留模型原始值）
- 增强日志输出，显示 scaling_factor、original_max_pos 和 rope_theta

**原因**:
- **关键修改**: Llama3 的 `rope_theta` 是 500000，而不是 Llama2 的 10000
- 硬编码 `rope_theta=10000` 会破坏 Llama3 的长上下文能力
- YaRN 方法依赖正确的 `rope_theta` 来进行频率插值
- 动态读取配置确保了对不同模型的通用性

#### 2.4 LongLoRA 方法

**修改位置**: `fine_tune_suffix.py` 第 306-321 行

**修改内容**:
- 更新注释
- 增强日志输出

**原因**:
- 与 PI 方法类似，需要正确识别原始上下文长度
- LongLoRA 使用线性插值，scaling factor 的计算与 PI 相同

## 使用建议

### 1. 训练脚本参数

原有的 shell 脚本 `finetune_main_suffix.sh` 已经正确指向 Llama3 8B 路径：
```bash
--model_name_or_path "/root/paddlejob/workspace/env_run/Llama-3-8B"
```

### 2. 预期行为

使用当前配置（`TRAINING_LENGTH=65536`）训练时：

**PI 方法**:
- Llama2: scaling_factor = 16
- Llama3: scaling_factor = 8 ✓

**NTK 方法**:
- Llama2: scaling_factor = 8
- Llama3: scaling_factor = 4 ✓

**YaRN 方法**:
- Llama2: scaling_factor = 16, rope_theta = 10000
- Llama3: scaling_factor = 8, rope_theta = 500000 ✓

### 3. 验证步骤

训练开始时检查日志输出，确认：
1. `orig_ctx_len` 应为 8192（Llama3）
2. `context_size` 应为 65536（训练长度）
3. `scaling_factor` 计算正确
4. 对于 YaRN 方法，`rope_theta` 应为 500000

示例日志：
```
training pi
RoPE scaling config: {'type': 'linear', 'factor': 8.0}, orig_ctx_len: 8192, context_size: 65536
```

## 重要注意事项

1. **rope_theta 的重要性**: Llama3 使用更大的 rope_theta (500000) 来支持更长的上下文。强制使用 Llama2 的 rope_theta (10000) 会严重损害模型性能。

2. **自动适配**: 修改后的代码会自动从模型配置中读取参数，因此同一份代码可以用于训练 Llama2 和 Llama3，只需改变 `model_name_or_path` 即可。

3. **向后兼容**: 所有修改都保持了向后兼容性，现有的 Llama2 训练配置仍然有效。

4. **其他模型**: 如果未来需要训练其他 Llama 系列模型，代码会自动适配，无需额外修改。

## 测试建议

1. 运行一个 epoch 较少的测试训练，验证：
   - 模型能正常加载
   - RoPE scaling 配置正确
   - 训练能正常进行
   - 没有 OOM 或其他错误

2. 检查训练日志中的配置输出，确保所有参数符合预期

3. 如果使用 YaRN 方法，特别注意 rope_theta 的值

## 修改日期

2026-01-31

## 作者

Zulu (Comate AI Assistant)