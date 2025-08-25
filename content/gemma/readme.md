# Gemma3

Gemma3 是 Google 推出的轻量级、高性能开源模型，支持多种尺寸（1B、4B、12B 和 27B），专为单 GPU 或 TPU 设计。它基于 Transformer 架构，进行了多项重要改进。本指南将详细介绍如何从零开始实现 Gemma3 模型，包括其核心架构、关键创新点以及完整的实现代码。

Gemma3的模型结构如下图所示：
![输入图片说明](/imgs/2025-08-24/OgLHsxFtkWxsZPBr.png)

---

## 1. 构建 RMS 层

Gemma3 使用 **RMSNorm**（均方根标准化）。LayerNorm 需要计算均值和标准差，计算量较大，且涉及减法操作，可能影响数值稳定性。RMSNorm 省略了均值计算，仅使用 均方根（RMS, Root Mean Square） 归一化：
$$
\text{rms}(x) = \sqrt{ \frac{1}{d} \sum_{i=1}^{d} x_i^2 }
$$

$$
\hat{x} = \frac{x}{\text{rms}(x) + \epsilon}
$$

$$
y = \gamma \hat{x}
$$
与传统实现不同，Gemma3 将缩放因子从 γ 改为 1 + γ，其中 γ 初始化为 0。这种设计使得初始状态下的标准化操作更加稳定，有助于模型训练过程中的收敛性和数值稳定性。
``` python
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps  # 小常数，防止除以零，增加数值稳定性
       
        # Gemma3使用零中心权重并在前向传播中使用(1 + weight)
        self.scale = nn.Parameter(torch.zeros(emb_dim))  # 缩放参数，初始化为0，实际使用(1+scale)
        
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None  # 可选的偏置参数，如果bias为True则启用

    def forward(self, x):
        input_dtype = x.dtype  # 保存输入的数据类型，以便最后转换回去
        x_f = x.float()  # 转换为float类型进行计算，提高数值精度
        
        var = x_f.pow(2).mean(dim=-1, keepdim=True)  # 计算均方值（不是方差，因为没有减去均值）
        x_norm = x_f * torch.rsqrt(var + self.eps)  # 归一化：x / sqrt(mean(x^2) + eps)
        
        out = x_norm * (1.0 + self.scale.float())  # 应用缩放：(1 + scale) * 归一化结果，scale初始为0
        
        if self.shift is not None:
            out = out + self.shift.float()  # 如果启用了偏置，加上偏置项
        
        return out.to(input_dtype) 
```        

## 2. FeedForward 层

Gemma使用 **GeGLU**，即一种门控线性单元变体。公式为：
$$
\text{output} = \text{FC3}\left( \text{GELU}(\text{FC1}(x)) \odot \text{FC2}(x) \right)
$$（其中 `⊙` 是逐元素乘法）。
    
输入 `x` 同时通过两个不同的线性层 (`fc1` 和 `fc2`)。fc1的输出经过 **GELU** 激活函数，产生一个"内容"或"值"。`fc2` 的输出作为一个"门"，控制哪些信息可以通过。 两者进行**逐元素相乘**，实现精细的信息流控制。门控后的结果通过 `fc3` 投影回原始嵌入维度。
GELU激活函数公式为：
$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$其中：$\Phi(x)$为标准正态分布的累积分布函数（CDF），$\text{erf}$为 误差函数（error function），$x$为输入值。
由于 GELU 的精确计算（涉及误差函数 `erf`）在计算上相对昂贵，一般使用高度精确的近似公式，该公式使用双曲正切函数 `tanh`。
$$
\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715x^3 \right) \right) \right)
$$
GELU可以理解为：根据输入值的大小，以一定的概率来决定让多少信息通过。
对于大的正输入：`tanh(...) ≈ 1`，因此 `GELU(x) ≈ 0.5x * (1 + 1) = x`。信息几乎完全通过。
       对于大的负输入：`tanh(...) ≈ -1`，因此 `GELU(x) ≈ 0.5x * (1 - 1) = 0`。信息几乎被完全抑制。
        对于接近零的输入：`tanh(...) ≈ 0`，因此 `GELU(x) ≈ 0.5x * (1 + 0) = 0.5x`。信息部分通过。

这种设计不仅增强了网络的表达能力，还能够选择性地控制信息流动，提高模型在处理复杂语言模式时的效率和准确性。
```python
class FeedForward(nn.Module):
    """GeGLU结构的门控前馈网络 (Gated Feed-Forward Network)"""
    def __init__(self, cfg):
        super().__init__()
        # 第一个线性层：用于生成"内容"或"值"
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 第二个线性层：用于生成"门"控信号
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 第三个线性层：将处理后的特征投影回原始维度
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        # 门控机制的核心是使用两个并行路径，其中一个路径提供"内容"，另一个路径提供"门"
        
        # 路径1：生成内容（将被GELU激活）
        x_fc1 = self.fc1(x)
        # 路径2：生成门控信号（直接作为权重）
        x_fc2 = self.fc2(x)
        
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        
        # 将门控后的结果投影回原始嵌入维度
        return self.fc3(x)
 ```

## 3. Attention 计算

### 3.1 旋转位置编码（RoPE）

为了确保模型能够处理超长上下文并保持顺序信息，Gemma3 对全局注意力层中的旋转位置编码（RoPE）进行了优化。具体来说，RoPE   的基频从原本的 10k 增加到 1M。这一调整扩展了位置编码的有效周期，使得模型在处理如 128K    长度的序列时，依然能够精确捕捉到相对位置关系。与此同时，局部注意层保持较低的 RoPE 频率（10k    基频），从而更专注于处理局部区域的精细关系。
```python
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算旋转位置编码的复数旋转向量（cis向量）
    Args:
        dim: 特征维度（通常是每个注意力头的维度）
        end: 最大序列长度（默认32K）
        theta: 旋转角度的基数，控制波长分布（默认10000.0）
    Returns:
        pos_cis: 复数张量，形状为 [end, dim//2]，包含所有位置的角度信息
    """
    # 计算频率向量
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)  
    
    # 计算外积：每个位置t乘以每个频率f，得到角度矩阵 [end, dim//2]
    freqs = torch.outer(t, freqs).float() 
    
    # 将角度转换为复数形式：cis(θ) = cos(θ) + i*sin(θ) = e^(iθ)
    pos_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return pos_cis


def apply_rotary_emb(x, pos_cis):
    """
    应用旋转位置编码到输入张量
    Args:
        x: 输入张量，形状为 [batch, seq_len, num_heads, dim]
        pos_cis: 预计算的旋转复数向量，形状为 [max_seq_len, dim//2]
    Returns:
        x_rotated: 应用旋转编码后的张量，形状与输入相同
    """
    def unite_shape(pos_cis, x):
        """调整pos_cis的形状以匹配x进行广播"""
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])  # 检查形状匹配：x形状为(batch_size, seq_len, num_heads, dim)
        
        # 创建广播形状：[1, seq_len, 1, dim] 以便与x [batch_size, seq_len, num_heads, dim] 相乘
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 将输入的最后两维重塑为复数形式：[batch, seq_len, num_heads, dim] -> [batch, seq_len, num_heads, dim//2, 2] -> 复数
    x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # 调整旋转向量的形状以便广播
    pos_cis = unite_shape(pos_cis, x)
    
    #复数乘法实现旋转 (a+bi) * (cosθ + i*sinθ) = 旋转后的向量
    x_rotated = torch.view_as_real(x * pos_cis).flatten(3)
	return x_rotated.to(dtype=x.dtype)
```

### 3.2 分组查询注意力（Grouped Query Attention）

Gemma3 引入了 **分组查询注意力** 机制，即将查询头进行分组处理，每组内的查询头共享相同的键值对。这样处理能够有效地减少键值缓存的内存占用，并且在保持模型性能的同时显著提升计算效率。
```python
class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
        query_pre_attn_scalar=None, dtype=None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "头数必须被kv组数整除"

        self.num_heads = num_heads          # 查询头的总数
        self.num_kv_groups = num_kv_groups  # 键值头的分组数
        self.group_size = num_heads // num_kv_groups  # 每组共享的查询头数量

        if head_dim is None:
            assert d_in % num_heads == 0, 
            head_dim = d_in // num_heads

        self.head_dim = head_dim    # 每个头的维度
        self.d_out = num_heads * head_dim  # 输出维度

        # 线性投影层
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)  
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)   
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype) 

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)  

        # 可选的查询和键归一化
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6) 
            self.k_norm = RMSNorm(head_dim, eps=1e-6) 
        else:
            self.q_norm = self.k_norm = None

        # 注意力分数缩放因子
        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5  # 使用自定义缩放
        else:
            self.scaling = (head_dim) ** -0.5  # 默认使用head_dim的平方根倒数

    def _prepare_grouped_kv(self, keys, values):
        """
        将KV头扩展到与查询头相同的数量
        keys: (batch_size, num_kv_groups, seq_len, head_dim)
        values: (batch_size, num_kv_groups, seq_len, head_dim)
        返回: 扩展后的keys和values, shape: (batch_size, num_heads, seq_len, head_dim)
        """
        # 使用repeat_interleave将每个KV头重复group_size次
        expanded_keys = keys.repeat_interleave(self.group_size, dim=1)
        expanded_values = values.repeat_interleave(self.group_size, dim=1)
        return expanded_keys, expanded_values

    def _compute_attention(self, queries, keys, values, mask):
        """
        计算注意力机制
        queries: (batch_size, num_heads, seq_len, head_dim)
        keys: (batch_size, num_heads, seq_len, head_dim)  
        values: (batch_size, num_heads, seq_len, head_dim)
        mask: 注意力掩码
        返回: 注意力上下文向量, shape: (batch_size, num_heads, seq_len, head_dim)
        """
        queries = queries * self.scaling
        
        attn_scores = queries @ keys.transpose(2, 3)
        
        # 应用掩码（将掩码位置设置为负无穷）
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        
        # 计算注意力权重（softmax归一化）
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 应用注意力权重到值上: 注意力权重 @ V
        context = attn_weights @ values
        return context

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape  # 获取batch大小和序列长度

        # 应用投影层获取查询、键、值
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # 重塑张量维度
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)    # (b, num_kv_groups, num_tokens, head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2) # (b, num_kv_groups, num_tokens, head_dim)

        # 可选的查询和键归一化
        if self.q_norm:
            queries = self.q_norm(queries)  # 归一化查询
        if self.k_norm:
            keys = self.k_norm(keys)        # 归一化键

        # 应用旋转位置编码（RoPE）
        queries = apply_rope(queries, cos, sin)  # 为查询添加位置信息
        keys = apply_rope(keys, cos, sin)        # 为键添加位置信息

        # 扩展K和V以匹配查询头数量
        keys, values = self._prepare_grouped_kv(keys, values)  # (b, num_heads, num_tokens, head_dim)

        context = self._compute_attention(queries, keys, values, mask)  # (b, num_heads, num_tokens, head_dim)

        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)  # (b, num_tokens, d_out)
        return self.out_proj(context)  # (b, num_tokens, d_in)
```
### 3.3 滑动窗口自注意力（Sliding Window Self-Attention）

为了支持长上下文处理，Gemma3 采用了 **局部-全局注意力层交替** 的混合架构：每 5 个局部层之间插入一个全局层，局部层采用，专门处理局部依赖关系，仅关注固定跨度（如 1024 个 token）范围内的上下文。全局层关注整个上下文范围，捕捉长距离语义依赖。
![滑动窗口自注意力机制](/imgs/2025-08-24/zGGxardamyUTnntY.png)
如图所示，左边是正常的causal attention，每个位置能看到自己和前面的位置，attention mask是个下三角矩阵。
右边则是滑动窗口自注意力的attention mask，这里的窗口大小为3。包括自己在内，每个位置只能往前看3个输入。
mask代码实现如下：
``` python
def _create_masks(seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    
        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)
    
        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
    
        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past
        return mask_global, mask_local
  ```
通过合理配置局部层和全局层的比例，这种混合架构实现了计算成本与性能的最佳平衡：在不显著增加计算成本的前提下，有效扩展上下文处理长度，将 KV 缓存内存开销从约 60% 大幅降低至不足 15%（基于 32K 上下文的测算结果）。



## 4. 组装 Gemma3 模型

通过整合以上所有核心组件和创新技术，Gemma3 构建了一个高效、强大的语言模型架构。这种设计不仅保证了模型的高性能表现，还有效控制了计算资源消耗，使其特别适合在资源受限的环境中部署和应用。
```python
class TransformerBlock(nn.Module):

    def __init__(self, cfg, attn_type):
        super().__init__()
        self.attn_type = attn_type 

        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x,
        mask_global,
        mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global
        
        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x

class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"]
        
        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, attn_type)for attn_type in cfg["layer_types"]
        ])

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.cfg = cfg

        # Reusable utilities    
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def forward(self, input_ids):
        # Forward pass
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
 ```
