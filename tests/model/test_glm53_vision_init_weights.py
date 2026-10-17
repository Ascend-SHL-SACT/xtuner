"""Regression for Glm53VisionModel.init_weights under recompute-wrapped blocks.

TestGlm53VisionInitWeights
    test_init_weights_with_checkpoint_wrapped_blocks  wrapped blocks must not false-positive

apply_activation_checkpointing (run during fully_shard with vision_recompute_ratio > 0)
wraps every checkpointed block as ``<block>._checkpoint_wrapped_module.<...>``, so
``named_parameters()`` carries that token while ``default_init_weights`` claims the
stripped (``clean_param_name``) names. Comparing raw names against the claimed set made
every wrapped-block parameter look uninitialized under random init:
``RuntimeError: {'blocks.19._checkpoint_wrapped_module.norm1.weight', ...} is not initialized``.
"""

import torch


class _CheckpointWrapper(torch.nn.Module):
    """Mimics the wrapper module torch's checkpointing inserts around blocks."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self._checkpoint_wrapped_module = inner


class TestGlm53VisionInitWeights:
    def test_init_weights_with_checkpoint_wrapped_blocks(self):
        from xtuner.v1.model.compose.glm53.glm53_config import Glm53VisionConfig

        config = Glm53VisionConfig(
            depth=1,
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            attn_impl="eager_attention",
        )
        model = config.build()
        model.blocks[0] = _CheckpointWrapper(model.blocks[0])
        # With the raw-name comparison this raised on every wrapped parameter.
        model.init_weights()
        for name, param in model.named_parameters():
            assert not param.is_meta, name
            assert torch.isfinite(param).all(), name
