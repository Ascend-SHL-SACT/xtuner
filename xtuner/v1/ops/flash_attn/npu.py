import warnings

import torch
import torch_npu


def npu_flash_varlen_attn(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """NPU varlen flash attention over packed sequences (TND layout).

    Contract (signature-compatible with the ``flash_attn_varlen_func`` interface; semantics
    that torch_npu cannot deliver fail loudly instead of silently degrading):

    - Honors ``softmax_scale``; when None it defaults to ``q.shape[-1] ** -0.5``.
    - Returns a single ``Tensor``: ``softmax_lse`` is not produced (callers such as
      ``attn_imp.flash_attention`` and the MLA paths branch on this), and
      ``return_attn_probs`` is accepted for signature compatibility but ignored.
    - Semantics torch_npu cannot deliver and that would change the math (``softcap``,
      ``alibi_slopes``, sliding ``window_size``, paged ``block_table``) fail loudly instead of
      silently degrading to full attention / an ignored scale. ``deterministic=True`` has no
      NPU knob and only weakens run-to-run reproducibility, so it warns and proceeds.

    Args:
        q (torch.Tensor): Query, ``[total_seq, num_heads, head_dim]`` (TND).
        k (torch.Tensor): Key, ``[total_seq_kv, num_heads_kv, head_dim]`` (TND).
        v (torch.Tensor): Value, ``[total_seq_kv, num_heads_kv, v_head_dim]`` (TND).
        cu_seqlens_q (torch.Tensor | list[int]): Cumulative query sequence ends, length ``n_seqs + 1``.
        cu_seqlens_k (torch.Tensor | list[int]): Cumulative key sequence ends, length ``n_seqs + 1``.
        max_seqlen_q (torch.Tensor): Max query sequence length (unused by NFA, kept for signature compatibility).
        max_seqlen_k (torch.Tensor): Max key sequence length (unused by NFA, kept for signature compatibility).
        dropout_p (float): Dropout probability (Default: 0.0).
        softmax_scale (float | None): Softmax scale; ``None`` falls back to ``q.shape[-1] ** -0.5`` (Default: None).
        causal (bool): Whether to apply a causal mask (Default: False).
        window_size (tuple[int, int]): Sliding window; ``(-1, -1)`` disables it (Default: (-1, -1)).
        softcap (float): Logit softcap; ``0.0`` deactivates it (Default: 0.0).
        alibi_slopes (torch.Tensor | None): ALiBi slopes; unsupported on NPU (Default: None).
        deterministic (bool): Deterministic mode; no NPU knob, warns and proceeds (Default: False).
        return_attn_probs (bool): Accepted for signature compatibility; attention probabilities are
            not returned (Default: False).
        block_table (torch.Tensor | None): Paged-attention block table; unsupported on NPU (Default: None).

    Returns:
        torch.Tensor: Attention output, ``[total_seq, num_heads, v_head_dim]``.
    """
    if softcap != 0.0:
        raise NotImplementedError("npu_flash_varlen_attn does not support softcap")
    if alibi_slopes is not None:
        raise NotImplementedError("npu_flash_varlen_attn does not support alibi_slopes")
    if window_size != (-1, -1):
        raise NotImplementedError(f"npu_flash_varlen_attn does not support sliding window {window_size}")
    if block_table is not None:
        raise NotImplementedError("npu_flash_varlen_attn does not support paged attention (block_table)")
    if deterministic:
        warnings.warn(
            "npu_flash_varlen_attn ignores deterministic=True: torch_npu NFA has no deterministic "
            "knob, run-to-run reproducibility is not guaranteed.",
            RuntimeWarning,
            stacklevel=2,
        )

    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    if not causal:
        fa_out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            pse=None,
            atten_mask=None,
            scale=scale,
            keep_prob=1 - dropout_p,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].tolist())
            if isinstance(cu_seqlens_q, torch.Tensor)
            else cu_seqlens_q[1:],
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].tolist())
            if isinstance(cu_seqlens_k, torch.Tensor)
            else cu_seqlens_k[1:],
        )[0]
    else:
        fa_out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[1],
            pse=None,
            atten_mask=torch.triu(torch.ones([2048, 2048], dtype=torch.bool, device=q.device), diagonal=1),
            scale=scale,
            keep_prob=1 - dropout_p,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].tolist())
            if isinstance(cu_seqlens_q, torch.Tensor)
            else cu_seqlens_q[1:],
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].tolist())
            if isinstance(cu_seqlens_k, torch.Tensor)
            else cu_seqlens_k[1:],
            sparse_mode=3,
        )[0]
    return fa_out
