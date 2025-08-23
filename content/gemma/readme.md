# Gemma3: Google推出的高性能轻量级开源模型

Gemma3 是 Google 推出的轻量级、高性能开源模型，支持多种尺寸（1B、4B、12B 和 27B），专为单 GPU 或 TPU 设计。它基于 Transformer 架构，进行了多项重要改进。本指南将详细介绍如何从零开始实现 Gemma3 模型，包括其核心架构、关键创新点以及完整的实现代码。

Gemma3的模型结构如下图所示：

---

## 1. 构建 RMS 层

Gemma3 使用 **RMSNorm**（均方根标准化），与传统实现不同，Gemma3 将缩放因子从 `w` 改为 `1 + w`，其中 `w` 初始化为 0。这种设计使得初始状态下的标准化操作更加稳定，有助于模型训练过程中的收敛性和数值稳定性。
``` python
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())#1.0 + self.scale.float(): 使用(1 + scale)而不是直接使用scale,scale初始为0
         
        if self.shift is not None:
            out = out + self.shift.float()
         
        return out.to(input_dtype)
```        

## 2. FeedForward 层

Gemma3 采用 **门控前馈网络**（Gated Feed Forward Network）。该网络利用两个不同的线性变换，通过元素级乘法实现门控效果。这种设计不仅增强了网络的表达能力，还能够选择性地控制信息流动，提高模型在处理复杂语言模式时的效率和准确性。
```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        #门控机制的核心是使用两个并行路径，其中一个路径提供"内容"，另一个路径提供"门"                               
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)
 ```

## 3. Attention 计算

### 3.1 旋转位置编码（RoPE）

为了确保模型能够处理超长上下文并保持顺序信息，Gemma3 对全局注意力层中的旋转位置编码（RoPE）进行了优化。具体来说，RoPE   的基频从原本的 10k 增加到 1M。这一调整扩展了位置编码的有效周期，使得模型在处理如 128K    长度的序列时，依然能够精确捕捉到相对位置关系。与此同时，局部注意层保持较低的 RoPE 频率（10k    基频），从而更专注于处理局部区域的精细关系。
```python
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(x, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])#x形状为(batch_size, seq_len, num_heads,dim)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, x)
    x_rotated = torch.view_as_real(x * pos_cis).flatten(3)
    return x_rotated.to(dtype=x.dtype)
```

### 3.2 分组查询注意力（Grouped Query Attention）

Gemma3 引入了 **分组查询注意力** 机制，这是一项重要的效率优化技术：

- 将查询头进行分组处理
- 每组内的查询头共享相同的键值对
- 有效减少键值缓存的内存占用
- 在保持模型性能的同时显著提升计算效率

### 3.3 滑动窗口自注意力（Sliding Window Self-Attention）

为了支持长上下文处理，Gemma3 创新性地采用了 **局部-全局注意力层交替** 的混合架构：

#### 架构设计
- **层间配置**：每 5 个局部层之间插入一个全局层
- **局部层功能**：专门处理局部依赖关系，仅关注固定跨度（如 1024 个 token）范围内的上下文
- **全局层功能**：负责处理长距离依赖关系，能够跨越整个 128K 上下文进行注意力计算

#### 技术实现
- **局部层**：采用滑动窗口自注意力机制，只关注当前窗口内的 tokens，确保计算效率
- **全局层**：关注整个上下文范围，捕捉长距离语义依赖

#### 性能优势
通过合理配置局部层和全局层的比例，这种混合架构实现了计算成本与性能的最佳平衡：
- 在不显著增加计算成本的前提下，有效扩展上下文处理长度
- 将 KV 缓存内存开销从约 60% 大幅降低至不足 15%（基于 32K 上下文的测算结果）

## 4. 组装 Gemma3 模型

通过整合以上所有核心组件和创新技术，Gemma3 构建了一个高效、强大的语言模型架构。这种设计不仅保证了模型的高性能表现，还有效控制了计算资源消耗，使其特别适合在资源受限的环境中部署和应用。
