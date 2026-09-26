
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Large negative constant used to zero out masked positions after softmax.
# Using -inf produces NaN when an entire row is masked, which happens for a
# fully padded sequence; a large finite value degrades gracefully instead.
NEG_INF = -1e9


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), NEG_INF)

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    return torch.matmul(weights, value), weights


class MultiHeadSelfAttention(nn.Module):


    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, seq, d_model)`` to ``(batch, heads, seq, d_k)``."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, heads, seq, d_k)`` back to ``(batch, seq, d_model)``."""
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        attn_out, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )

        return self.w_o(self._merge_heads(attn_out)), weights


class SinusoidalPositionalEncoding(nn.Module):
    

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear, ReLU, Dropout, Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderBlock(nn.Module):
    """One Transformer encoder block.

    ``use_residual``, ``use_layernorm`` and ``use_ffn`` exist so that individual
    components can be switched off without touching the rest of the model. That
    is what the ablation study varies.

    ``norm_first`` selects pre-layer-norm (as in most modern implementations)
    rather than the post-layer-norm of the original paper.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layernorm: bool = True,
        use_ffn: bool = True,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.use_ffn = use_ffn
        self.norm_first = norm_first

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout) if use_ffn else None

        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def _combine(self, residual: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        out = self.dropout(sublayer_out)
        return residual + out if self.use_residual else out

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_first:
            attn_out, weights = self.self_attn(self.norm1(x), mask)
            x = self._combine(x, attn_out)
            if self.feed_forward is not None:
                x = self._combine(x, self.feed_forward(self.norm2(x)))
        else:
            attn_out, weights = self.self_attn(x, mask)
            x = self.norm1(self._combine(x, attn_out))
            if self.feed_forward is not None:
                x = self.norm2(self._combine(x, self.feed_forward(x)))

        return x, weights


def masked_mean(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    
    mask = pad_mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class TransformerClassifier(nn.Module):
    """Sequence classifier built from the components above.

    Args:
        vocab_size: size of the token vocabulary, including PAD and UNK.
        d_model: embedding and model width.
        num_heads: attention heads per block.
        d_ff: hidden width of the feed-forward network.
        num_layers: number of stacked encoder blocks.
        num_classes: number of output classes.
        dropout: dropout probability, applied in attention, the FFN, positional
            encoding and between sublayers.
        max_len: maximum supported sequence length.
        pad_idx: token id used for padding.
        use_positional_encoding: switch off to test whether order information
            matters for this task.
        use_residual, use_layernorm, use_ffn, norm_first: passed to each block.
        pooling: ``"masked_mean"``, ``"mean"`` (the unmasked version, kept so the
            ablation can measure what the bug costs) or ``"cls"``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_idx: int = 0,
        use_positional_encoding: bool = True,
        use_residual: bool = True,
        use_layernorm: bool = True,
        use_ffn: bool = True,
        norm_first: bool = False,
        pooling: str = "masked_mean",
    ) -> None:
        super().__init__()
        if pooling not in {"masked_mean", "mean", "cls"}:
            raise ValueError(f"unknown pooling strategy: {pooling}")

        self.pad_idx = pad_idx
        self.d_model = d_model
        self.pooling = pooling
        self.use_positional_encoding = use_positional_encoding

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = (
            SinusoidalPositionalEncoding(d_model, max_len, dropout)
            if use_positional_encoding
            else nn.Dropout(dropout)
        )

        self.blocks = nn.ModuleList(
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_residual=use_residual,
                use_layernorm=use_layernorm,
                use_ffn=use_ffn,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        )

        # A pre-LN stack leaves its output un-normalised, because every LayerNorm
        # sits inside a residual branch. The residual stream therefore grows with
        # depth and the classifier receives activations of uncontrolled scale.
        # Every working pre-LN implementation adds a final norm; omitting it here
        # made the pre-LN ablation collapse to chance accuracy, which looked like
        # a finding about pre-LN and was actually a bug in this file.
        self.final_norm = (
            nn.LayerNorm(d_model) if (norm_first and use_layernorm) else nn.Identity()
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, tokens: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Args:
            tokens: ``(batch, seq)`` integer token ids.
            return_attention: also return per-layer attention weights, each
                ``(batch, heads, seq, seq)``.
        """
        pad_mask = tokens != self.pad_idx  # (batch, seq), True where real
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)

        # Scale embeddings so the token signal is not swamped by positional
        # values, which lie in [-1, 1] regardless of d_model.
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x, attn_mask)
            if return_attention:
                attentions.append(weights)

        x = self.final_norm(x)

        if self.pooling == "masked_mean":
            pooled = masked_mean(x, pad_mask)
        elif self.pooling == "mean":
            pooled = x.mean(dim=1)
        else:  # "cls": first position, assumed to hold a prepended CLS token
            pooled = x[:, 0, :]

        logits = self.classifier(pooled)
        return (logits, attentions) if return_attention else logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        """Human-readable parameter breakdown, per module."""
        lines = [f"{'module':<44}{'shape':<24}{'params':>12}"]
        lines.append("-" * 80)
        for name, param in self.named_parameters():
            if param.requires_grad:
                shape = "x".join(str(d) for d in param.shape)
                lines.append(f"{name:<44}{shape:<24}{param.numel():>12,}")
        lines.append("-" * 80)
        lines.append(f"{'TOTAL trainable':<68}{self.count_parameters():>12,}")
        return "\n".join(lines)
