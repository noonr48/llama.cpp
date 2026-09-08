# Flash-Next (qwen4exp) MTP speculative decoding — llama.cpp fork branch

This branch adds **working NextN/MTP speculative decoding for the Qwen3.8-Flash-Next
(qwen4exp) architecture** to `llama-server`, plus several decode/prefill performance
paths and server checkpoint fixes. Upstream `llama.cpp` master already supports the
qwen4exp architecture itself; the speculative-decoding and performance work below is
**not** upstream (as of 2026-09-08).

## What this adds

- **NextN/MTP speculative decoding, end to end** — draft graph and draft context for
  the 1-layer MTP head, verified against the target model (`83718637a`).
- **Flexible draft loading** — MTP draft weights from a sidecar GGUF, in-file draft
  tensors, or draft-head-only GGUFs in the unsloth layout (`290b632df`, `25a450031`).
- **Combinable speculative types** — e.g. `--spec-type draft-mtp,ngram-map-k` runs an
  MTP draft and an n-gram draft together.
- **Gather-based sparse attention for QSA decode** (`296522485`) — avoids the
  scatter path for the compressed full-attention layers during decode.
- **Direct reads for the lazy PLE table** — `--lazy-mode on-direct` (`f77ea055c`).
- **Deferred output-row gather** past the `h_nextn` export (`f86ab0d9c`).
- **l2norm for gated-delta-net q/k**, matching flash-linear-attention (`0e763b171`).
- **Server checkpoint + MTP rollback cluster** — prompt-checkpoint save/restore
  host-side and on-device, speculative recurrent-state checkpoints kept on-device,
  full checkpoints for MTP rollback, and no re-verification of replayed draft tokens
  after a checkpoint restore.

## Measured results

Controlled A/B, fresh 25k-token prompts (no prompt-cache reuse), 600-token decode
means, 7-GPU mixed rig (1× RTX 5090 + 3× RTX 3090 + 3× RTX 5060 Ti, layer split,
262,144-token context, fp32 KV):

| Spec config            | Decode (tok/s) | 25k cold prefill (tok/s) |
|------------------------|----------------|--------------------------|
| MTP draft only         | 54.4           | 424                      |
| MTP + `ngram-map-k`    | **58.8** (+8%) | **437** (+3%)            |

Needle-recall and vision probes pass under speculative decoding on this rig.

## Usage

Build as usual (see the upstream README). Serve with a draft model:

```bash
llama-server \
  -m Qwen3.8-Flash-Next-....gguf \
  --model-draft mtp-draft.gguf \
  --spec-type draft-mtp,ngram-map-k \
  -c 262144 -fa on
```

The draft is the model's own 1-layer NextN/MTP head (extracted to a sidecar GGUF in
our setup; in-file draft tensors and unsloth-layout draft-head-only GGUFs also work).
A 1-deep draft (`--spec-draft-n-max 1`) measured best for us — deeper draft chains
from a 1-layer head decayed acceptance monotonically (n1 54–58, n2 48.7–53.8,
n3 43.5–46.2 tok/s).

## Status

- Branch: `flashnext-mtp` (tip tracks the deployed serving build).
- Not upstream; qwen4exp architecture support **is** already in upstream master, so
  the plain model runs fine on mainline — this branch is for the speculative-decode
  and performance paths.
- License: MIT, inherited from llama.cpp.

## Provenance

Weights tested: an IQ4_NL quantization of an abliterated/uncensored Flash-Next
release (Apache-2.0 source), repacked with the MTP head split out as the draft
sidecar. Quality was verified against the source Q8_0 reference and the base model
in a symmetric balanced evaluation before these performance numbers were taken.
