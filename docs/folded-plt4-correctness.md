# Verify folded PLT-4 on a new machine

This protocol covers the 16-stage folded PLT-4, with Foresight **on and off**, using direct frontier KV and 2D G-SWA. It uses synthetic token IDs and deterministic weights. A single BF16-capable, allocated GPU is sufficient for the CUDA check. The CPU reference needs only PyTorch. Run from a Git checkout with the package installed.

## Command to hand to an agent

```bash
git clone https://github.com/thuEcstasy/Pipelined-LM.git
cd Pipelined-LM
python -m pip install -e .
report_dir=$(mktemp -d)
python -m tests.check_folded_plt4 --device cuda --report "$report_dir/folded-cuda.json"
```

> Run the command above on the intended training GPU using CUDA-enabled PyTorch 2.11.x. Return the JSON report, exit code, commit, GPU model and the largest relative L2 for each Foresight arm. On failure, include the first failing check and traceback. Use the recorded tolerances. A completed CUDA check has exit code 0, `requested_device: "cuda"` and `status: "passed"`. The command runs the CPU oracle before the CUDA attention and train/serve checks.

For CPU-only verification, use `--device cpu --report "$report_dir/folded-cpu.json"`. Each invocation writes a new report. The CUDA command checks GPU availability explicitly. The first CUDA run compiles the attention kernels. A report marked `running` identifies the last check reached if the process is interrupted.

## What one prediction is allowed to read

Number stages from 1 to 16 and execution cycles from 0. After observing token $x_c$, stage $r$ is working on prediction position $t=c+17-r$. At stage 16, this is the immediate next token $x_{c+1}$. A producer in stage $j$ at cycle $u$ is working on position $s=u+17-j$. A KV source at the **same within-stage layer offset** is cycle-causal when $j\le r$ and $u\le c$, equivalently $s\le t+r-j$. The visible prediction positions ahead of $t$ are latent states computed from the observed prefix.

**Foresight is a stage-input connection.** Follow prediction A from A1 to A16. At A4 with Foresight off, add the current observed embedding to the raw output A3 from the previous cycle. With Foresight on, take A3, B2 and C1, all from that same previous cycle. RMS-normalize each source, score it with the receiver query divided by $\sqrt d$ plus a per-source bias, softmax across sources, and add the scaled weighted sum to the current embedding. In general, stage $r$ reads stages $1,\ldots,r-1$ at $c-1$. At a document start the message is zero. Stage 1 always takes just the embedding. The same rule applies at boundaries within an outer pass and across two passes.

**Frontier KV is an attention read.** At each layer offset, project Q/K/V from all current stage inputs, publish K/V, then attend. Current-cycle projections from shallower stages are ready for use. Current-cycle *stage outputs* are not inputs to Foresight. In training, earlier stages have already processed all columns, so masking by document and publication cycle is essential.

**2D G-SWA uses two softmaxes.** The entries below are for one batch row, one same-document receiver $(r,c)$ and one layer offset. Clip each lower bound at the start of the current document and exclude invalid positions.

| Branch | Producer stages | Visible publication cycles | Normalization |
|---|---|---|---|
| Global | $j=1$ | All $u\le c$, including $c$ | One softmax over the global prefix |
| Local, $r>1$ | $j=2,\ldots,r$, including the receiver | $c-31\le u\le c$, including both endpoints | One **joint softmax** across every eligible stage–cycle pair |

Stage 1 uses only global attention. At later stages the output per query head is $(1-g)A_{global}+gA_{local}$. Compute $g$ by taking a head-specific dot product with the RMS-normalized query before RoPE, adding the head bias, and applying sigmoid. Apply the output projection after mixing. Direct frontier has no separate frontier softmax or frontier gate. The first stage already supplies its frontier through the global branch.

**Worked example.** With one uninterrupted document starting at cycle 0, let stage 4 attend at $c=39$. Its prediction position is $t=52$.

| Producer stage | Branch | Visible cycles | Prediction positions strictly ahead of 52 |
|---|---|---|---|
| 1 | Global | 0–39 | 53, 54, 55 |
| 2 | Local | 8–39 | 53, 54 |
| 3 | Local | 8–39 | 53 |
| 4 | Local | 8–39 | none |

Cycle 7 has expired from every local window and is still valid in the global bank. Stage 5 and cycle 40 are invisible. Across the seven layer offsets, decode retains seven full-history KV pairs and $7\times15=105$ local KV pairs of 32 slots. Publication cycle $u$ occupies slot $u\bmod32$ in its own local bank. Switching Foresight on reuses the 15 previous-cycle stage outputs already retained by the predecessor chain. Both arms use this KV layout.

## Using the reference with another implementation

The scalar reference takes a mapping of raw parameter tensors, a backbone configuration, a one-dimensional CPU token row and an explicit Foresight setting. For an external implementation, map its tensor names to the research checkpoint names and use the same values in both models. This example reconstructs the logits for every column of a single document:

```python
import torch
from tests.folded_reference import readout, scalar_trajectory

weights = dict(cpu_fp32_model.named_parameters())
states, entries, kv = scalar_trajectory(weights, config, ids, foresight=True)
# Final stage has zero-based index 15; column c predicts position c+1.
hidden = torch.cat([states[15, c + 1] for c in range(len(ids))], dim=1)
expected_logits = readout(weights, config, hidden)
```

The Python API uses zero-based stage indices. To localize a mismatch at stage index `r` and column `c`, look up `entries[r, c+16-r]`, then `kv[r, layer_offset, c+16-r]`, then the stage output in `states`. For the CPU gradient check, give the implementation and reference separate copies of the same differentiable weights. Both score the same next-token targets.

## What the runner verifies

| Check | Method and acceptance |
|---|---|
| Independent trajectory and gradient reference | A scalar CPU FP32 implementation consumes raw named weights and token IDs. It explicitly computes tied physical-layer selection, stage inputs, Q/K/V, RoPE, both attentions, residuals, FFN and tied readout. It enumerates sources by prediction position. Compare every stage input/output, every K/V, logits, next-token NLL and every parameter gradient. Foresight and the attention gates use nonuniform weights in this check. The elementwise tolerances are `rtol=2e-4, atol=3e-5`. |
| Frontier legality and G-SWA weights | A diagnostic fixture sets Q and K to zero and tags V by producer stage and publication cycle. Four heads use local gates 0, 1, 0.25 and 0.75. Read the vectors arriving at the actual output projection in both the production training graph and the incremental decoder. Each component is one source's mixing weight. All forbidden weights must be exactly zero. Each allowed weight must match its analytic coefficient (CPU relative tolerance 1e-5, CUDA BF16 0.02, absolute tolerance zero). The fixture checks every receiver and cycle over 35 cycles, including the first two window evictions. |
| Boundaries, training objective and resume | CPU cases cover padding, a repeated document ID after an intervening document, prefix invariance, multiple window wraps, chunked decode, matching backbone initialization, and exact replay of the next AdamW update. The existing untied Pipeline CPU cache-policy suite also runs against its own reference. |
| Backend after an optimizer update | Exercise both arms with all 28 physical layers, 16 stages and seven offsets, using reduced width. Check finite loss and gradients, perform a real optimizer update, and compare training-path logits with incremental decode and prefill at 33 followed by a continuation to 40. Compare the previous-cycle state snapshot, every global K/V and every live local ring slot against the training graph for both routes. The report records per-tensor relative L2. Limits are 1e-4 for CPU and 0.02 for CUDA BF16. The CUDA fixture uses contiguous single-document rows to exercise the production Flash/window/Flex path. |

In the CPU handoff run, the tagged-source probe measured zero weight on every forbidden edge in training and decode. The largest absolute coefficient error was $7.45\times10^{-9}$ in both. Three deliberate faults in isolated model copies were detected: removing the legal local frontier first failed at receiver 3, expiring local KV one cycle early first failed at receiver 2, and forcing the enabled arm to use only its predecessor failed the scalar trajectory check.

The CPU scalar reference checks the dependency graph and gradients; the CUDA comparisons measure native training versus serving. The fixtures use reduced width and synthetic documents. Dataset order, tokenizer provenance, full-width memory use, a specific trained checkpoint, and multi-node execution require their own run-level checks.

[Scalar PyTorch reference](../tests/folded_reference.py) · [Frontier and G-SWA weight probe](../tests/test_folded_visibility.py) · [Native backend parity](../tests/folded_backend.py) · [Agent runner](../tests/check_folded_plt4.py) · [Untied Pipeline reference](../tests/reference.py)
