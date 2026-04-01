# =============================================================================
# model/context_query.py
# ContextQueryExtractor：LENS 风格的 Context Query 模块
#
# 相比原 QueryExtractor 的改进：
#   KV 从单 [SEG] token 扩展为完整生成序列的 hidden states
#     - 训练：labels != -100 标记的所有 assistant token 的 hidden states
#       （包含 bbox 坐标 token、语义句子 token、[SEG] token）
#     - 推理：逐步生成时每步的 last-layer hidden state 拼成的序列
#   架构从单层 cross-attention 升级为 N 层 TransformerDecoder block
#     （Pre-LN：self-attn → cross-attn → FFN，各含残差连接）
#   queries 4 → 16（默认，可配置）
#
# 这使每个 query 能同时注意到：
#   bbox 坐标 token（空间先验）、语义句子 token、[SEG] token 本身，
#   而不只是 [SEG] 单个 token 的压缩表示，信息量大幅提升。
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FFN(nn.Module):
    """Position-wise Feed-Forward Network，GELU 激活。"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _DecoderLayer(nn.Module):
    """
    标准 Transformer Decoder block（Pre-LN 版本，训练更稳定）：

        LN → Self-Attn (queries attend to each other) → 残差
        LN → Cross-Attn (queries × context) → 残差，支持 key_padding_mask
        LN → FFN → 残差
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.self_attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = _FFN(embed_dim, ffn_dim)

    def forward(
            self,
            queries: torch.Tensor,               # (B, M, D)
            context: torch.Tensor,               # (B, ctx_len, D)
            ctx_key_padding_mask: torch.Tensor,  # (B, ctx_len)，True = padding（忽略）
    ) -> torch.Tensor:
        # Self-attention（queries 内部交互）
        normed = self.norm1(queries)
        attn_out, _ = self.self_attn(normed, normed, normed)
        queries = queries + attn_out

        # Cross-attention（queries 从 context 提取信息）
        normed = self.norm2(queries)
        attn_out, _ = self.cross_attn(
            normed, context, context,
            key_padding_mask=ctx_key_padding_mask,
        )
        queries = queries + attn_out

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class ContextQueryExtractor(nn.Module):
    """
    LENS 风格的 Context Query 提取器。

    从 LLM 生成序列的全部 hidden states 中提取丰富的上下文信息，
    通过 num_layers 层 TransformerDecoder 将信息聚合到 num_queries 个可学习
    query 向量上，输出 (B, num_queries, embed_dim) 作为 SAM2 sparse embeddings。

    训练模式（forward 调用）：
        context_hidden  = last_hidden[labels != -100]  -- 所有 assistant token
        ctx_key_padding_mask = 填充位为 True，真实位为 False

    推理模式（generate_with_mask 调用）：
        context_hidden  = stack([outputs.hidden_states[s][-1][:,-1,:] for s in steps])
        ctx_key_padding_mask = zeros (全部有效，无填充)
    """

    def __init__(
            self,
            llm_dim: int = 4096,
            embed_dim: int = 256,
            num_queries: int = 16,
            num_heads: int = 4,
            num_layers: int = 2,
            ffn_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        ffn_dim = int(embed_dim * ffn_ratio)

        # num_queries 个可学习 query 向量（trunc_normal 初始化，避免全零梯度消失）
        self.queries = nn.Parameter(torch.empty(num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # KV 投影：llm_dim → embed_dim（全部 context token 共享同一投影）
        self.kv_proj = nn.Linear(llm_dim, embed_dim, bias=False)

        # num_layers 层 TransformerDecoder block
        self.layers = nn.ModuleList([
            _DecoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            context_hidden: torch.Tensor,        # (B, ctx_len, llm_dim)
            ctx_key_padding_mask: torch.Tensor,  # (B, ctx_len)，True=填充（忽略）
    ) -> torch.Tensor:
        """
        返回: (B, num_queries, embed_dim) —— SAM2 sparse embeddings
        """
        B = context_hidden.shape[0]

        # 投影所有 context token 到 embed_dim
        context = self.kv_proj(context_hidden)  # (B, ctx_len, embed_dim)

        # 广播可学习 queries 到当前 batch
        queries = self.queries.to(dtype=context.dtype).unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, context, ctx_key_padding_mask)

        return self.out_norm(queries)  # (B, num_queries, embed_dim)