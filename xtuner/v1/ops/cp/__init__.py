# Copyright (c) OpenMMLab. All rights reserved.
"""Context-parallel ring attention for GLM-5.2 (absorbed DSA-MLA).

Gated by ``XTUNER_CP_RING=1`` (default OFF); the implementation is the
genuine ring -- per-chunk kernels with online-softmax merging, hop ``j+1``
posted before chunk ``j``'s kernel. The ring transport (``RingP2P``), the
math (``forward_true``/``backward_true``) and the autograd entry
(``RingAttentionCP``) all live in the single module ``ring_attention.py``.
There is no gather-all variant: the former ``chimera`` relay-then-one-kernel
path (a KV-allgather in ring clothing) was removed. The small indexer key is
all-gathered separately (outside this module) so the DSA top-k stays global.
"""

from xtuner.v1.ops.cp.ring_attention import RingAttentionCP, RingP2P


__all__ = ["RingAttentionCP", "RingP2P"]
