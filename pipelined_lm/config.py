"""Four-stage partition for the untied pipeline."""

from dataclasses import dataclass
from .core.layers import Backbone


@dataclass(frozen=True)
class Partition:
    """front | branches x depth | tail, summing to Backbone.layers.

    The research configuration has four stages of six layers.
    """

    front: int = 2
    branches: int = 4
    depth: int = 6
    tail: int = 2

    @property
    def total(self):
        return self.front + self.branches * self.depth + self.tail

    def front_indices(self):
        return list(range(self.front))

    def branch_indices(self, b):
        if not 0 <= b < self.branches:
            raise ValueError("branch out of range")
        start = self.front + b * self.depth
        return list(range(start, start + self.depth))

    def tail_indices(self):
        start = self.front + self.branches * self.depth
        return list(range(start, start + self.tail))


@dataclass(frozen=True)
class PipelineConfig:
    partition: Partition
    branch_attention: str
    layout: str = "folded"
    execution_core: str = "plt"
    cross_part_source: str = "previous_token"
    front_attention: str = "global"
    tail_attention: str = "global"

    def shares_kv(self):
        return self.branch_attention == "gswa"


def tiny_backbone():
    """Small four-stage instance for CPU examples and correctness tests."""
    return Backbone(
        vocab=128,
        dim=32,
        layers=12,
        heads=4,
        kv_heads=2,
        head_dim=8,
        intermediate=64,
        window=4,
    )
