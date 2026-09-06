# qwen4exp tensor-split development — design (increment 1)

Goal: beat layer-split decode (54.7 t/s no-MTP baseline, 9-GPU lane) with a working
`-sm tensor` mode for qwen4exp on this fork, fixing the measured dispatch tax
(prefill 40x collapse: 9.3 t/s vs 350+ layer-split; MTP composite 2.8-4 t/s).

Branch: `qwen38-tsplit-dev` (from `deea0a591` = enablement prototype + CUDA
graph-cache cherry-picks + draft SPLIT_MODE_LAYER fix, on top of deployed HEAD
`ac82c7278`). Worktree: `/home/benbi/llama.cpp-qwen38-tsplit-dev`.
NEVER rebuild or checkout in the MAIN worktree — `build/bin/llama-server` is the
deployed :8331 binary (qwen38-flash-next.service, raw-source uncensored quant).

## Verified evidence map (all file:line checked on this branch)

- Allowlist gate (the only thing rejecting `-sm tensor` for qwen4exp):
  `src/llama-arch.cpp` `llm_arch_supports_sm_tensor` — `case LLM_ARCH_QWEN4EXP: //
  TODO: fix test-llama-archs` → `return false`. (Also excluded: GROK, MPT, PLAMO2,
  MINIMAX_M3, MISTRAL4, KIMI_LINEAR, BAILINGMOE3, KIMI_K3, QWEN3TTS.)
  Throw site: `src/llama-model.cpp:350-351` in `llama_model_create`.
- Dispatch tax home: `ggml/src/ggml-backend-meta.cpp:1969-1970` in
  `ggml_backend_meta_graph_compute`:
  `needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid)`.
  `backend_ctx->uid` is a single slot (`:1803`), stored after compute (`:2226`).
  Rebuild path (`:1978+`): resets used meta buffers' stc containers (parity flip
  `stc_compute_index ^ 1`, `ggml_reset` per ctx), repopulates `bcj.nodes` /
  subgraphs per backend per node, then MoE AllReduce-delay scan
  (`get_i_delayed_branch` `:2036-2084`).
- uid minting: `ggml/src/ggml-backend.cpp:1085` `graph->uid = ggml_graph_next_uid()`
  inside `ggml_backend_sched_split_graph` (`:1066`), run from
  `ggml_backend_sched_alloc_graph` (`~:2004`). Decode stability relies on the
  copy-rotation trick (`meta:2040-2046` comment); ANY new ubatch shape (prefill
  chunks, MTP draft/verify alternation) re-splits → new uid → full rebuild.
- Op coverage: `MUL_MAT_ID` IS handled (`meta:920-922`, shared with MUL_MAT via
  `handle_mul_mat`); LIGHTNING_INDEXER has a dedicated handler; per-op dispatch
  switch `meta:518-950+`. Gap = handle_mul_mat's supported combos + default case
  + GET_ROWS/abort paths (QWEN4EXP split-state callback conditions at
  `src/llama-model.cpp:595/734` decide which tensors row-split).
- Layer-split hook (for hybrid designs): `get_layer_buft_list` lambda at
  `src/llama-model.cpp:1471`.
- split_state_cache: `meta:430` keyed `(tensor*, assume_sync)` + byte snapshot;
  ANY snapshot mismatch clears the WHOLE cache (`meta:1109-1118`).
- Precedents for shape/uid reuse keying: RPC `ggml-rpc.cpp:1021`
  (`reuse = uid != 0 && last_graph_uid == uid`), CUDA graphs `ggml-cuda.cu:2596-2603`,
  the two cherry-picked commits on this branch (CUDA graph cache keyed by shape).

## Increment 1 options (invasiveness order)

A. **Two-slot uid LRU** (least invasive, targets MTP draft/verify alternation):
   keep `uid_a/uid_b` + per-slot built subgraph state. Requires moving
   `backend_configs[j].nodes` (in-place vector) to per-slot storage AND handling
   the per-buffer `stc_compute` parity (only 2 parity slots exist — matches a
   2-signature cache exactly). Draft/verify steady-state = zero rebuilds.
   Risk: stc containers are shared per meta-buffer across signatures; a rebuild
   of B clobbers containers that A's subgraphs reference → must verify whether
   subgraphs reference the wrappers by pointer (then 2-slot restore is unsafe
   without re-materializing) or by copy.

B. **Shape-signature cache** (general, handles prefill ubatch variety):
   exact signature = walk ops + src tensor shapes of `cgraph->nodes/leafs`
   (O(n_nodes) per call — far cheaper than rebuild; MUST be exact, probabilistic
   hits = wrong subgraphs = correctness bugs). Bounded LRU (8-16 entries), keyed
   on signature; value = per-backend node assignment lists + subgraph tops.
   Same stc-container problem as A but for N entries → needs container
   re-materialization on hit (cheaper than full rebuild if wrapper creation is
   the cheap part) — MEASURE FIRST.

C. **Profile-first** (cheapest, do this before A/B):
   add coarse timers around the three rebuild phases (container reset /
   per-backend node loop / delay scan) and around `graph_compute` total, run
   one prefill + one MTP-verify workload in a GPU window, find where the
   100-200ms actually goes. The fix may be much smaller than A/B (e.g. delay
   scan is O(nodes × backends) and could be cached independently).

## Bench protocol (GPU window)

1. `systemctl --user stop qwen38-flash-next.service` (halts :8331; 9 lane GPUs free).
2. Binary: THIS worktree's `build-tsplit/bin/*` (configure: `cmake -B build-tsplit
   -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;120 -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`).
3. Baselines on same GPUs: layer-split 54.7 t/s decode (no-MTP, temp 1.0) and
   n2-MTP 50.6 t/s from the 2026-09-03 sweep (memory: learn_762ade76).
4. tsplit runs: decode bench (ub 512), prefill bench (pp2048/pp16384), needle
   60k sanity, MTP n2 composite. Targets: decode ≥ 54.7, prefill ≥ 350 t/s,
   needle PASS, no assert.
5. `systemctl --user start qwen38-flash-next.service` + health check when done.
   Residents (VoxCPM GPU-2e4cba95, ComfyUI-spare GPU-ebbf9150, voice-tutor
   GPU-32d5cab0) are NEVER on the lane set — untouched.

## Status

- [x] Upstream scan (nothing grabbable; 3 negative signals; PRs #28100/#28185/#27750)
- [x] Fork recon (this doc's evidence map)
- [x] Branch + worktree + this doc
- [ ] build-tsplit configured + compiles
- [ ] Option C profile instrumentation + one GPU-window measurement
- [ ] Implement chosen fix (A/B per profile)
- [ ] Bench protocol pass, commit on branch, PR-quality summary
