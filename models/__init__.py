"""Models package for TE-Q-Transformer research repository.

Exports the proposed TE-Q-Transformer architecture and configuration factories.
Baseline architectures for benchmark comparisons are self-contained under
`notebooks/benchmark/baseline_comparison_nasa.ipynb`.
"""

from models.proposed.te_q_transformer import (
    TEQTransformer,
    TEQTransformerConfig,
    rich_entangler_config,
)

__all__ = [
    "TEQTransformer",
    "TEQTransformerConfig",
    "rich_entangler_config",
]
