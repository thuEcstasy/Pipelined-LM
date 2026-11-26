"""Packed/ragged math path, with the same PLT backbone and last-loop readout."""

from dataclasses import dataclass
import torch


@dataclass
class GraphOutput:
    hidden: torch.Tensor
    loops: tuple
    inputs: tuple
    kv: dict


def layout(tokens, docs, valid=None):
    if tokens.ndim != 2 or docs.shape != tokens.shape:
        raise ValueError("tokens and docs must be [B,T]")
    if valid is None:
        valid = torch.ones_like(tokens, dtype=torch.bool)
    if valid.shape != tokens.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean [B,T] mask")
    # Baseline has unpadded rows. The new API supports a contiguous valid span
    # (left/right padding and empty rows), and explicitly rejects internal holes.
    if tokens.shape[1] > 1:
        starts = valid[:, 1:] & ~valid[:, :-1]
        if bool((starts.sum(1) + valid[:, 0].long() > 1).any()):
            raise ValueError(
                "Interior padding holes are unsupported; compact valid tokens first"
            )
    starts = torch.ones_like(docs, dtype=torch.bool)
    starts[:, 1:] = (docs[:, 1:] != docs[:, :-1]) | ~valid[:, :-1]
    # Repeated external document IDs still identify separate contiguous segments.
    segments = starts.long().cumsum(1)
    return tokens.masked_fill(~valid, 0), segments.masked_fill(~valid, -1), valid
