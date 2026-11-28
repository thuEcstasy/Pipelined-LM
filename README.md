# PLT and Pipelined LM

Train autoregressive models that overlap work for successive predictions. All three recipes support sequence-parallel training, prefill, incremental decode, held-out validation, and resumable single-device or distributed pretraining.

| Recipe | One prediction trajectory | Attention and hidden-state inputs |
|---|---|---|
| `plt4` | Four passes through one shared 28-layer stack | The first pass supplies global KV; each later pass reads its own 64-cycle window. Each pass takes the current embedding and its own trajectory's previous pass output. |
| `folded-plt4` | Each of the four shared passes splits into four 7-layer stages, for 16 stages total | Direct 2D G-SWA with 7 global banks and 105 windows of 32 cycles. Earlier stages publish the legal same-cycle frontier. Every stage receives the current embedding. Foresight is off. |
| `pipeline` | 2 input layers, four independent 6-layer stages, 2 output layers | Foresight at each stage boundary, direct 2D G-SWA with a shared first-stage global bank per layer offset and 64-cycle local windows. |

[Architecture, other cache policies, and previous experimental results](docs/architecture.md)

## Start here from a clean checkout

Use Linux, Python 3.10 or newer and PyTorch 2.11.x. The CPU path works without any model downloads. For GPU training, install a CUDA-enabled PyTorch 2.11.x build compatible with the host and use an allocated BF16-capable GPU. Run from the repository root. Access to this repository is required to clone it.

```bash
git clone https://github.com/thuEcstasy/Pipelined-LM.git
cd Pipelined-LM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data]'
python -m unittest discover -s tests -v
```

**Instruction to give a Codex agent**

> Follow this README in a clean checkout. Install the package and run the correctness tests. Run all three CPU smoke recipes below, resume one checkpoint, and decode from it. Report the last step, target count and validation NLL for each run. If I specify a corpus, GPU allocation and budget, proceed to the corresponding real-data recipe after the backend check passes.

```bash
for recipe in plt4 folded-plt4 pipeline; do
  python -m pipelined_lm.train --smoke --recipe "$recipe" --steps 2 --output "runs/smoke/$recipe" || exit 1
done
python -m pipelined_lm.train --smoke --recipe folded-plt4 --steps 3 --resume --output runs/smoke/folded-plt4
python examples/decode.py runs/smoke/folded-plt4/latest.pt --tokens 11 23 7 --new-tokens 4
```

The smoke run uses generated IDs. With default settings each update scores 64 next-token targets. Look for JSON events named `ready`, `train`, `validation`, and `checkpoint`. At step 2, each job should have 128 cumulative training targets and a `latest.pt`. The resumed folded run should reach 192.

## First real-data run

The following commands fetch the pinned Qwen tokenizer and stream public FineWeb-Edu and FineMath-3+ text. They write fixed-width rows with 90% web and 10% math over every ten rows. Validation documents are assigned by a deterministic hash before tokenization. The first preparation requires network access. To use the original web-only PLT-4 corpus family, change `--corpus mix` to `--corpus web` and select a different output directory.

```bash
python scripts/prepare_data.py --corpus mix --seq-len 1024 --train-rows 128 --val-rows 20 --output data/pilot-mix
python -m pipelined_lm.train --recipe pipeline --device cuda --data data/pilot-mix \
  --steps 2 --global-batch 64 --micro-batch 1 --eval-rows 20 --output runs/pilot-pipeline
```

For the 16-stage experiment on the same 128 rows, use `--recipe folded-plt4` with the same batch and step arguments and a fresh output directory. For a CPU real-data integration check, use `--size tiny --device cpu` and a small sequence length at data preparation. The tiny real-data model retains the Qwen token vocabulary.

## Research-size training

| Recipe | Data | Default target | Global batch × sequence length | Learning rate |
|---|---|---:|---:|---|
| `plt4` | FineWeb-Edu | 45,777 steps / 6,000,082,944 targets | 128 × 1,024 | 128-update warmup to 3e-4, then constant |
| `folded-plt4` | Select the same corpus as the intended control | Supply `--steps` | 128 × 1,024 by default | Same schedule as PLT-4 |
| `pipeline` | 90% FineWeb-Edu + 10% FineMath | 76,294 steps / 5,000,003,584 targets | 64 × 1,024 | 100-update warmup to 3e-4, with the historical cosine extension at update 22,889 |

The trainer uses AdamW with betas (0.9, 0.95), weight decay 0.1 on backbone matrices, zero decay on scalar/vector parameters and the Pipeline cross-stage aggregation, and gradient clipping at 1.0. CUDA uses BF16 autocast and FP32 optimizer parameters. Smaller microbatches increase gradient accumulation while preserving global batch, LR and row order.

Choose one recipe. Prepare enough unique training rows for the cumulative budget. The launcher below uses four allocated GPUs on one Linux node; `python -m pipelined_lm.train` runs the same recipe on one GPU. With `torchrun`, global batch must be divisible by world size times microbatch.

```bash
# Historical-size Pipeline run. The data files need roughly 20 GB plus preparation/cache space.
python scripts/prepare_data.py --corpus mix --train-rows 4882816 --val-rows 2048 --output data/pipeline-5b
torchrun --standalone --nproc-per-node=4 -m pipelined_lm.train \
  --recipe pipeline --data data/pipeline-5b --steps 76294 --micro-batch 1 --output runs/pipeline-5b
```

```bash
# Shared-weight PLT-4. Reuse this same prepared directory for a matched folded run.
python scripts/prepare_data.py --corpus web --train-rows 5859456 --val-rows 2048 --output data/plt-web
torchrun --standalone --nproc-per-node=4 -m pipelined_lm.train \
  --recipe plt4 --data data/plt-web --steps 45777 --micro-batch 1 --output runs/plt4-6b
```

For folded PLT at that budget, change the last command to `--recipe folded-plt4 --steps 45777 --output runs/folded-plt4-6b`. This is a proposed comparison budget. The previously reported quality table covers the untied Pipeline experiments. The portable public data builder starts a new deterministic row ordering. Replaying an archived experiment exactly also requires its original token rows and initial tensors. An original independent-row directory works directly, and `--initial-weights /path/to/initial.pt` accepts a strictly matching raw state dict on the first launch.

## Checkpoints, validation and handoff

- Every launch verifies the train/val file sizes and checksums, compares training-path logits with incremental decode on the selected device, and checks finite model state. CPU uses relative L2 tolerance 1e-4 and CUDA BF16 uses 0.02. The unit suite also checks KV, causality, packed-document boundaries, gradients and window rollover.
- Validation reports mean next-token NLL on the first `--eval-rows` held-out rows (default 128). Set `--eval-rows 2048` for the whole 2,048-row validation split. The trainer validates and saves at the final step and at `--eval-every` / `--save-every` intervals.
- `latest.pt` is replaced atomically and contains model, AdamW state, per-rank RNG, next data row and cumulative targets. After an interruption, repeat the original command with `--resume`. Keep the same data, model, device type, world size, global batch and microbatch. `--steps` is the cumulative target. Missing training rows produce an error before training starts.
- To use local text with the same data builder, pass `--web-jsonl` and, for a local mixture, `--math-jsonl`. Each file contains one JSON object with a `text` field per line. `--tokenizer-json` accepts a local copy of the pinned tokenizer. The manifest records source identities, per-split counts, widths and hashes.
- Report the Git commit, recipe, data manifest hash, GPU model/count, global and microbatch sizes, last step/targets, validation NLL and checkpoint location. The packaged CPU runs have been tested. The CUDA-specific unit check runs on GPU hosts; GPU throughput and quality of the new folded experiment remain to be measured.

[Training entry point](pipelined_lm/train.py) · [Public data builder](scripts/prepare_data.py) · [Shared PLT and folded PLT](pipelined_lm/folded_plt.py) · [Untied Pipeline](pipelined_lm/model.py)
