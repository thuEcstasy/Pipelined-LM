# Architecture and results

Pipelined LM is an autoregressive Transformer that overlaps the work for successive predictions. Each prediction travels through four independently parameterized stages. While the last stage finishes one prediction, the other three advance newer trajectories. After the pipeline fills, one execution cycle emits one token.

The research model uses a Qwen3-0.6B-shaped backbone with **2 input layers, 4 stages of 6 layers, and 2 output layers**. A trajectory traverses all 28 layers. The ideal per-cycle critical path is 10 layer evaluations when the four middle stages execute concurrently. Actual decode latency depends on attention, memory traffic and the execution backend.

## Follow one prediction

| Cycle | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| 1 | **A1** | | | |
| 2 | B1 | **A2** | | |
| 3 | C1 | B2 | **A3** | |
| 4 | D1 | C2 | B3 | **A4** |

The path for prediction **A** is **A1 → A2 → A3 → A4**. Each stage receives the representation of the token observed in its current cycle. A4 therefore has access to a token that A1 had yet to observe. The table shows the trajectories starting at cycle 1; real prefill also computes boundary states for earlier positions.

**Foresight** lets A4 read the completed states A3, B2 and C1 from cycle 3. A3 carries its main trajectory; B2 and C1 bring intermediate computations from neighboring trajectories. With `foresight=False`, only A3 is used at that boundary.

**Frontier KV** works inside attention. A4 can retrieve keys and values at the same layer offset from earlier stages, including newly published projections for B3, C2 and D1 where the policy permits. These are latent predictions computed from the observed prefix. Training processes a stage across the sequence in parallel. Decode publishes the current same-offset K/V projections before the stages perform attention.

## Folded PLT-4 with a 32-cycle frontier window

`FoldedPLT4` starts from the shared 28-layer PLT-4 backbone. Each of its four passes is split into four cross-token stages of seven layers. A single prediction visits physical layers 1–7, 8–14, 15–21 and 22–28, then repeats that path three more times. The final stage feeds the tied LM head. The implementation retains PLT-4's FFN output RMSNorm and residual scale $0.5/(4\sqrt{28})$.

The first stage receives the current token embedding. Every subsequent stage adds that cycle's token embedding to its own trajectory's preceding stage output from one cycle earlier. This version uses the plain predecessor connection with Foresight disabled. At each layer offset 0–6, stage 1 publishes global KV. Stage $r>1$ mixes global attention with one joint local attention over stages 2 through $r$, using the last 32 execution cycles inclusive. All same-offset projections for the current cycle are published before attention. For a receiver predicting position $t$, an earlier producer $j$ can supply states through $t+r-j$, including the direct frontier.

This gives 7 global banks and 105 local windows. Training evaluates all token positions together within a stage; decode batches the four uses of each physical slice, then processes all 16 global queries together. A trajectory uses 112 layer evaluations, and the ideal backbone critical path per steady-state token is 7 layers. The corresponding **serial four-pass loop** has a 112-layer critical path, giving a theoretical 16× depth ratio. The existing **pipelined PLT-4** has a 28-layer per-cycle path, giving a 4× ratio against that baseline. Both ratios describe depth. Wall-clock speed and training throughput require hardware measurements.

```bash
python examples/train.py --architecture folded-plt4 --checkpoint-layers --steps 2 --output checkpoints/folded-tiny.pt
python examples/decode.py checkpoints/folded-tiny.pt --tokens 11 23 7 --new-tokens 4
python -m unittest tests.test_folded_plt -v
```

For research size, select `--size qwen06`. The training example accepts the legacy PLT-4 data directory via `--data` (memory-mapped rows from `manifest.json` and `train.bin`) and its bare initialization state dict via `--initial-weights`. The example performs single-device AdamW updates. It is a model integration and smoke-training entry point; historical PLT-4 quality comparisons also require its full batch, token order and learning-rate schedule.

## Choosing a cache policy

The six policies below apply to the untied pipeline. In 2D attention, a receiver at stage *r* and prediction position *t* can read a producer at stage *j ≤ r* at the same within-stage layer offset. History ends at *s ≤ t*. The additional legal frontier is *t < s ≤ t + r − j*.

| `policy` | Sources available to a middle-stage query | KV capacity |
|---|---|---|
| `full_1d` | The receiver’s own full causal history | 28 full banks |
| `full_2d` | Its own and earlier stages’ full histories through the legal frontier | 28 full banks |
| `gswa_1d` | Stage 1 global through the current cycle, mixed with the receiver’s own window | 10 full + 18 windows |
| `history` | Stage 1 global and local windows from stages 2 through *r*, restricted to history | 10 full + 18 windows |
| `direct` (default) | The same global/local banks, with frontier indices included directly | 10 full + 18 windows |
| `gated` | The history base plus separate attention over frontier entries, with a learned per-head gate | 10 full + 18 windows |

Counts above include the input and output layers for the 28-layer model. With 64-cycle windows and an 8,192-token context, the windowed variants use **36.2%** of the full-KV tensor capacity. Runtime workspaces, hidden states and measured HBM traffic are separate quantities.

## Matched 5B-token results

Five stage-boundary Foresight runs trained from scratch with the same budget of **5,000,003,584 prediction targets**. NLL is evaluated on an independent set of 2,048 rows × 1,024 targets per domain. The downstream column is the unweighted average of zero-shot ARC-Easy, ARC-Challenge and PIQA accuracy. These are single-seed observations.

| KV policy | FineWeb-Edu NLL ↓ | FineMath NLL ↓ | Downstream avg. (%) ↑ |
|---|---:|---:|---:|
| Full 2D | 2.8412 | 2.5928 | 47.47 |
| G-SWA 1D | 2.8791 | 2.6336 | 46.52 |
| 2D history | 2.8801 | 2.6393 | 47.14 |
| 2D direct frontier | 2.8609 | 2.6185 | 47.62 |
| 2D gated frontier | 2.8647 | 2.6229 | 47.50 |

At the same bounded KV capacity, direct frontier improved NLL over history by 0.0192 on FineWeb-Edu and 0.0208 on FineMath. Full 2D gave the lowest NLL on both domains. Full 1D has a different historical evaluation record and is omitted from this matched table.

## Experimental subblock residuals

For the earlier aligned-Foresight ablation, set `aligned_foresight=True` with `policy="direct"`. It reads the previous cycle’s matching attention/FFN output at each site, without the extra same-trajectory depth reader.

`PipelinedLM(policy="direct", subblock_residuals=True, block_size=4)` enables aligned Foresight before every middle-stage attention and FFN. In the diagram, A4 can also retrieve the completed A1 and A2 endpoints along its own trajectory. Inside a stage, all 12 sublayers have independent depth queries over completed block updates and the live partial update. A block groups four sublayers by default. Setting `block_size=1` exposes each previous sublayer update individually.

The additional write gates start at zero, so the initial function equals the direct-frontier boundary model with the same seed. This variant has CPU checks for the training graph, incremental decode and gradients. The 5B table above reports stage-boundary models.

## Code and verification

- [Model and loss](../pipelined_lm/model.py) · [G-SWA training and streaming](../pipelined_lm/gswa.py) · [Full-KV training and streaming](../pipelined_lm/full_kv.py)
- [Same-trajectory depth reads](../pipelined_lm/depth_read.py) · [Aligned Foresight](../pipelined_lm/subblock.py)
- [Optimized decoder](../pipelined_lm/engine.py) · [CUDA kernels](../pipelined_lm/kernels) · [Independent trajectory checks](../tests/test_pipeline.py)

The CPU suite compares sequence-parallel logits, KV and gradients against scalar token-by-token oracles. It also checks chunked decode, document boundaries, future-token causality and the zero-gate baseline. The optional CUDA decoder uses Triton and a compatible vLLM build exposing the paged FlashAttention backend. This repository packaging was verified on CPU; CUDA Graph parity and throughput need measurement on the target GPU stack.

For the related shared-weight looped architecture and its benchmarks, see [InfuseLoop](https://github.com/thuEcstasy/InfuseLoop).
