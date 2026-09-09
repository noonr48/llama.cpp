# Tensor-split mode for qwen4exp (Flash-Next) — status, the GQA fix, and honest numbers

This branch implements **tensor splitting** for the qwen4exp architecture: individual
graph tensors (weights, activations, caches) are split across multiple physical GPUs
behind a "Meta" composite device (see `ggml/src/ggml-backend-meta.cpp`), in addition
to the stock layer-split mode. On prefill-heavy workloads this parallelizes each
layer's matrix work across all devices instead of running layers one-GPU-at-a-time.

**Status (2026-09-09): correct and usable.** A multi-week corruption bug was
root-caused and fixed; exact-recall probes and a quality battery pass. The mode
doubles prefill throughput at 25k tokens on our testbed. One capacity limit applies
(context ≤ 32768 on a 4×24-32GB pool — details below).

## The bug we chased (and killed): GQA head correspondence

**Symptom.** With full-attention (FA) layers on the Meta composite, every prompt —
regardless of content — degenerated into the same short garbage string. Needle-recall
(a code planted mid-document) failed at any depth. Layer-split on the same hardware
was perfect, and a `GGML_META_MIRROR_WQ=1` workaround (mirroring the Q projection so
nothing splits) also restored exact recall.

**Root cause.** The FA handler accepts a **head-split Q with fully mirrored K/V** —
but the CUDA attention kernels derive the grouped-query (GQA) ratio from the *local*
tensor shapes they are handed (`k = head / (local_Q_heads / local_KV_heads)`;
`ggml/src/ggml-cuda/fattn-vec.cuh:106-111`). With, e.g., 24 Q heads / 2 KV heads
split 12+12 across two backends, each backend computes ratio 12/2 = 6 instead of the
true 12 — every backend pairs its queries with the **wrong KV heads**. Caches were
bit-identical the whole time; the *selection* was wrong. Mirroring Q "fixed" it only
by restoring the full head count so the kernel's ordinary math held.

**Evidence chain.** A probe (`GGML_META_GQA_PROBE=1`) at the first
`FLASH_ATTN_EXT` prints per-backend head geometry and flags the mismatch; a
diagnostic offset-scaling dump ruled out the earlier wrapper/offset theories by
arithmetic. Credit: the mechanism was identified by an adversarial review pass over
this branch and confirmed at runtime by the probe within one boot.

**The fix** (`GGML_META_GQA_FIX=1`, in the per-backend wrapper construction,
`ggml-backend-meta.cpp` ~:1365-1416): each backend's FA node receives **read-only
K/V aliases** into the mirrored storage at the correct global head origin —
`first_KV = P_j/G`, `local_KV = H_j/G`, byte offset `first_KV * nb[2]`, strides
unchanged. The kernel then recovers the true ratio with the correct origin, while Q
and the output projection stay split (no redundant compute). Guard: contiguous
splits only (`n_segments == 1 && nr[0] == 1`); group-crossing slices are skipped
loudly rather than mis-aliased.

**Validation** (temperature 0, thinking disabled, deterministic):
- 1.5k reproducer: exact HIT (was identical garbage pre-fix)
- fresh 25k needle: exact HIT, both in the NGL=6 hybrid and full-tensor `-ngl 999`
- 10-item quality suite: **parity with layer mode** (same score 5/10 on the same
  model, same GPU pool — the suite is hard without thinking in both modes)
- Independent spec-compliance review: pass (after a repair wave that added the
  contiguity guard)

## Honest performance (fresh prompts, no prompt-cache reuse)

Testbed: 4-way composite on one RTX 5090 + three RTX 3090 (PCIe Gen3 x4 — see the
caveat in the main README; Gen4/Gen5 systems should do better), fp32 KV.

| Config | 25k-token prefill |
|---|---|
| Layer-split reference | 437-489 tok/s |
| **Full-tensor + GQA fix** | **~875 tok/s (2.0×)** |

**Capacity limit.** The FA/indexer caches grow with context; at `-c 131072` they
need ~4.8 GB per backend and OOM 24 GB cards after their weight share. Practical
ceiling on this pool: **`-c 32768`**. Longer contexts need cache-placement work
(caches off the composite, or per-layer routing) — not a correctness issue; the
working point serves ≤32k prompts exactly.

## How to run it

```bash
GGML_META_GQA_FIX=1 llama-server \
  -m <flash-next-gguf> \
  -sm tensor -ts "1.6,1,1,1" -ngl 999 -fa on -c 32768 \
  -ctk f32 -ctv f32 --no-kv-unified \
  --reasoning-format deepseek --reasoning-preserve
```

- `-sm tensor` selects tensor split; `-ts` weights the composite members; `-ngl 999`
  puts all layers on the composite (a small `-ngl` gives a hybrid: the last N layers
  split, the rest layer-split).
- `GGML_META_GQA_FIX=1` is required for correct FA layers whenever Q is head-split
  with mirrored K/V (the default planning for this architecture).
- Diagnostics: `GGML_META_GQA_PROBE=1` prints the head-correspondence check once
  per process.

## Repo map

- `ggml/src/ggml-backend-meta.cpp` — the Meta composite backend: split-state
  propagation, per-backend wrappers, the gqa-fix/gqa-probe blocks
- `src/llama-model.cpp` — tensor-split planner (per-tensor axis/granularity config)
- `TENSOR_SPLIT_DEV.md` — the full research log (evidence chain, dead ends, every
  measurement; long but complete)
- `validate_true_fa.sh` — the one-command correctness gate (reproducer + fresh needle
  through a self-cycling test server)

License: MIT, inherited from llama.cpp.
