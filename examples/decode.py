"""Greedy incremental decode from a portable example checkpoint."""

import argparse
import torch
from pipelined_lm import Backbone, PipelinedLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tokens", nargs="+", type=int, required=True)
    parser.add_argument("--new-tokens", type=int, default=4)
    args = parser.parse_args()
    if args.new_tokens < 1:
        parser.error("new-tokens must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PipelinedLM(Backbone(**ckpt["backbone"]), **ckpt["options"]).eval()
    model.load_state_dict(ckpt["model"])
    prefix = torch.tensor([args.tokens])
    if (prefix < 0).any() or (prefix >= model.c.vocab).any():
        parser.error("token ID outside checkpoint vocabulary")
    out = model.prefill(prefix)
    produced = []
    for i in range(args.new_tokens):
        token = out.logits[:, -1:].argmax(-1)
        produced.append(token.item())
        if i + 1 < args.new_tokens:
            out = model.decode(token, cache=out.cache)
    print("generated IDs:", *produced)


if __name__ == "__main__":
    main()
