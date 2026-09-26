

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from attnablate.attention import (
    MultiHeadSelfAttention,
    SinusoidalPositionalEncoding,
    TransformerClassifier,
    masked_mean,
    scaled_dot_product_attention,
)

TOL = 1e-5


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)


# --------------------------------------------------------------------------
# Scaled dot-product attention
# --------------------------------------------------------------------------


def test_sdpa_matches_torch_functional() -> None:
    """Our attention equals ``F.scaled_dot_product_attention`` with no mask."""
    q, k, v = (torch.randn(2, 4, 7, 16) for _ in range(3))

    ours, weights = scaled_dot_product_attention(q, k, v)
    reference = F.scaled_dot_product_attention(q, k, v)

    assert torch.allclose(ours, reference, atol=TOL)
    assert weights.shape == (2, 4, 7, 7)


def test_sdpa_matches_torch_functional_with_mask() -> None:
    """Equality holds under a key padding mask too.

    ``F.scaled_dot_product_attention`` takes a boolean mask where ``True`` means
    *keep*, matching our convention.
    """
    q, k, v = (torch.randn(2, 4, 7, 16) for _ in range(3))
    mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    mask[:, :, :, 5:] = False  # last two positions are padding

    ours, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    assert torch.allclose(ours, reference, atol=TOL)


def test_sdpa_rows_are_probability_distributions() -> None:
    q, k, v = (torch.randn(3, 2, 6, 8) for _ in range(3))
    _, weights = scaled_dot_product_attention(q, k, v)

    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 2, 6), atol=TOL)


def test_sdpa_masked_positions_receive_no_weight() -> None:
    """Masked keys must end up with effectively zero attention weight."""
    q, k, v = (torch.randn(1, 1, 5, 8) for _ in range(3))
    mask = torch.ones(1, 1, 1, 5, dtype=torch.bool)
    mask[..., 3:] = False

    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    assert torch.all(weights[..., 3:] < 1e-8)
    assert torch.allclose(weights[..., :3].sum(dim=-1), torch.ones(1, 1, 5), atol=TOL)


def test_sdpa_scaling_actually_divides_by_sqrt_dk() -> None:
    """Check the 1/sqrt(d_k) factor."""

    d_k = 16
    q, k = torch.randn(1, 1, 4, d_k), torch.randn(1, 1, 4, d_k)
    v = torch.eye(4).view(1, 1, 4, 4)

    out, _ = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1)

    assert torch.allclose(out, expected, atol=TOL)


# --------------------------------------------------------------------------
# Multi-head self-attention
# --------------------------------------------------------------------------


def _copy_weights_into_torch_mha(
    ours: MultiHeadSelfAttention, reference: nn.MultiheadAttention
) -> None:
   
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight], dim=0)
        )
        reference.in_proj_bias.copy_(
            torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias], dim=0)
        )
        reference.out_proj.weight.copy_(ours.w_o.weight)
        reference.out_proj.bias.copy_(ours.w_o.bias)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_mha_matches_torch_multiheadattention(num_heads: int) -> None:
    d_model, seq, batch = 32, 9, 3

    ours = MultiHeadSelfAttention(d_model, num_heads, dropout=0.0).eval()
    reference = nn.MultiheadAttention(
        d_model, num_heads, dropout=0.0, batch_first=True
    ).eval()
    _copy_weights_into_torch_mha(ours, reference)

    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        our_out, _ = ours(x)
        ref_out, _ = reference(x, x, x, need_weights=False)

    assert torch.allclose(our_out, ref_out, atol=TOL)


def test_mha_matches_torch_with_padding_mask() -> None:
    
    d_model, num_heads, seq, batch = 32, 4, 9, 3

    ours = MultiHeadSelfAttention(d_model, num_heads, dropout=0.0).eval()
    reference = nn.MultiheadAttention(
        d_model, num_heads, dropout=0.0, batch_first=True
    ).eval()
    _copy_weights_into_torch_mha(ours, reference)

    keep = torch.ones(batch, seq, dtype=torch.bool)
    keep[0, 6:] = False
    keep[1, 4:] = False

    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        our_out, _ = ours(x, mask=keep.unsqueeze(1).unsqueeze(2))
        ref_out, _ = reference(x, x, x, key_padding_mask=~keep, need_weights=False)

    # Compare only the non-padded positions. Outputs at padded query positions
    # are undefined in both implementations and are discarded by pooling.
    for row in range(batch):
        valid = keep[row]
        assert torch.allclose(our_out[row, valid], ref_out[row, valid], atol=TOL)


def test_mha_head_splitting_preserves_information() -> None:
    """``_split_heads`` then ``_merge_heads`` must be an identity round trip."""
    mha = MultiHeadSelfAttention(32, 4, dropout=0.0)
    x = torch.randn(3, 9, 32)
    assert torch.allclose(mha._merge_heads(mha._split_heads(x)), x, atol=TOL)


def test_mha_rejects_indivisible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(d_model=30, num_heads=4)


# --------------------------------------------------------------------------
# Positional encoding
# --------------------------------------------------------------------------


def test_positional_encoding_matches_closed_form() -> None:
    d_model, max_len = 16, 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    pe = pe_module.pe.squeeze(0)

    for pos in (0, 1, 7, 49):
        for i in range(d_model // 2):
            angle = pos / (10000 ** (2 * i / d_model))
            assert pe[pos, 2 * i] == pytest.approx(math.sin(angle), abs=1e-6)
            assert pe[pos, 2 * i + 1] == pytest.approx(math.cos(angle), abs=1e-6)


def test_positional_encoding_is_not_trainable() -> None:
    """The table must be a buffer, not a parameter."""
    pe = SinusoidalPositionalEncoding(16, 50)
    assert "pe" in dict(pe.named_buffers())
    assert list(pe.parameters()) == []


def test_positional_encoding_distinguishes_positions() -> None:
    """Different positions must get different vectors, or word order is invisible."""
    pe = SinusoidalPositionalEncoding(32, 100, dropout=0.0).pe.squeeze(0)
    assert not torch.allclose(pe[0], pe[1], atol=1e-3)
    assert not torch.allclose(pe[5], pe[20], atol=1e-3)


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------


def test_masked_mean_ignores_padding() -> None:
    """The documented bug: unmasked mean is dragged toward the pad vectors."""
    x = torch.zeros(1, 4, 2)
    x[0, 0] = torch.tensor([1.0, 1.0])
    x[0, 1] = torch.tensor([3.0, 3.0])
    x[0, 2] = torch.tensor([99.0, 99.0])  # padding, must be ignored
    x[0, 3] = torch.tensor([99.0, 99.0])  # padding, must be ignored

    keep = torch.tensor([[True, True, False, False]])

    assert torch.allclose(masked_mean(x, keep), torch.tensor([[2.0, 2.0]]))
    # The naive version is wildly off, which is the point of the ablation.
    assert not torch.allclose(x.mean(dim=1), torch.tensor([[2.0, 2.0]]))


def test_masked_mean_handles_fully_padded_row() -> None:
    """A row of pure padding must not divide by zero."""
    x = torch.randn(1, 4, 8)
    keep = torch.zeros(1, 4, dtype=torch.bool)
    assert torch.isfinite(masked_mean(x, keep)).all()


# --------------------------------------------------------------------------
# Whole model
# --------------------------------------------------------------------------


def test_classifier_output_shape_and_finiteness() -> None:
    model = TransformerClassifier(vocab_size=100, d_model=32, num_heads=4, max_len=20)
    tokens = torch.randint(1, 100, (5, 20))
    logits = model(tokens)

    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_classifier_ignores_trailing_padding() -> None:
    """Appending padding to a sequence must not change its prediction.

    This is the end-to-end version of the masked-pooling test and it is the one
    the original notebook would fail.
    """
    model = TransformerClassifier(
        vocab_size=100, d_model=32, num_heads=4, max_len=40, dropout=0.0
    ).eval()

    short = torch.randint(2, 100, (1, 12))
    padded = torch.cat([short, torch.zeros(1, 28, dtype=torch.long)], dim=1)

    with torch.no_grad():
        # Pad the short one to the same length so shapes match, then compare
        # against a version with far more padding.
        a = model(torch.cat([short, torch.zeros(1, 4, dtype=torch.long)], dim=1))
        b = model(padded)

    assert torch.allclose(a, b, atol=1e-4)


def test_classifier_returns_attention_of_expected_shape() -> None:
    model = TransformerClassifier(
        vocab_size=100, d_model=32, num_heads=4, num_layers=3, max_len=20
    )
    tokens = torch.randint(1, 100, (2, 20))
    _, attentions = model(tokens, return_attention=True)

    assert len(attentions) == 3
    for weights in attentions:
        assert weights.shape == (2, 4, 20, 20)


def test_gradients_reach_every_parameter() -> None:
    """A silent way to break a model is to leave a layer disconnected."""
    model = TransformerClassifier(vocab_size=50, d_model=32, num_heads=4, max_len=16)
    tokens = torch.randint(1, 50, (4, 16))

    loss = F.cross_entropy(model(tokens), torch.randint(0, 2, (4,)))
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"
            if "embedding" not in name:  # padding_idx row is legitimately zero
                assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_positional_encoding": False},
        {"use_residual": False},
        {"use_layernorm": False},
        {"use_ffn": False},
        {"num_heads": 1},
        {"num_layers": 1},
        {"norm_first": True},
        {"pooling": "mean"},
        {"pooling": "cls"},
    ],
)
def test_every_ablation_configuration_runs(overrides: dict) -> None:
    """Each ablation the study reports must at least build and do a forward pass."""
    model = TransformerClassifier(vocab_size=50, d_model=32, max_len=16, **overrides)
    logits = model(torch.randint(1, 50, (3, 16)))

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_embeddings_are_scaled_by_sqrt_d_model() -> None:
    """Verify the sqrt(d_model) scaling the original notebook omitted.

    With positional encoding disabled and dropout off, the input to the first
    block should be exactly ``embedding(tokens) * sqrt(d_model)``.
    """
    d_model = 64
    model = TransformerClassifier(
        vocab_size=50,
        d_model=d_model,
        num_heads=4,
        max_len=8,
        dropout=0.0,
        use_positional_encoding=False,
    ).eval()

    tokens = torch.randint(1, 50, (2, 8))
    captured: list[torch.Tensor] = []
    model.blocks[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )

    with torch.no_grad():
        model(tokens)
        expected = model.embedding(tokens) * math.sqrt(d_model)

    assert torch.allclose(captured[0], expected, atol=TOL)


def test_prelayernorm_stack_ends_with_a_final_norm() -> None:
    """Pre-LN needs a final LayerNorm, or the residual stream leaves the stack unnormalised.

    Omitting this made the pre-LN ablation collapse to chance accuracy during
    development. The test exists so the regression cannot come back silently.
    """
    pre_ln = TransformerClassifier(
        vocab_size=50, d_model=32, num_heads=4, max_len=16, norm_first=True
    )
    post_ln = TransformerClassifier(
        vocab_size=50, d_model=32, num_heads=4, max_len=16, norm_first=False
    )

    assert isinstance(pre_ln.final_norm, nn.LayerNorm)
    assert isinstance(post_ln.final_norm, nn.Identity)


def test_prelayernorm_output_scale_is_controlled() -> None:
    """The pooled representation entering the classifier must not blow up."""
    model = TransformerClassifier(
        vocab_size=50,
        d_model=64,
        num_heads=4,
        num_layers=4,
        max_len=16,
        dropout=0.0,
        norm_first=True,
    ).eval()

    with torch.no_grad():
        logits = model(torch.randint(1, 50, (8, 16)))

    assert torch.isfinite(logits).all()
    assert logits.abs().max() < 100, "pre-LN residual stream is growing unchecked"
