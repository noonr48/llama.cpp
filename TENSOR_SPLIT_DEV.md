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
- [x] build-tsplit configured + compiles (BUILD_RC=0; note: quote `-DCMAKE_CUDA_ARCHITECTURES='86;120'` — the semicolon splits unquoted bash)
- [x] Option C instrumentation in place (phase timers around reset/nodes/delay, GGML_LOG_INFO "meta rebuild: ...") — UNTESTED (blocked by the load crash below)
- [x] First GPU window (2026-09-07 00:45): **tensor mode CRASHES AT LOAD** with the new
      uncensored quant: `GGML_ASSERT(meta_buf_ctx->bufs[i]) failed` at
      `ggml-backend-meta.cpp:1761` in `ggml_backend_meta_alloc_ctx_tensors_from_buft`
      ← `llama_model::load_tensors` (full stack in profile_window.log). Hypotheses:
      (a) PLE lazy-host tensors (>4GiB per_layer_token_embd, host mmap) confusing the
      meta buffer-context allocation; (b) the 24 BF16 indexer projections; (c) eval
      branch was only ever exercised with the CENSORED baseline (iq4nl_pleq8), never
      with this tensor mix. NOTE: the layer-mode comparison run was poisoned (started
      1s after the core dump, failed model load) — rerun cleanly next window.
      llama-bench gotchas learned: no `-c` flag (ctx derives from -p/-n); `-ctk f32`
      rejected (value validation) — defaults f16 are fine for rebuild profiling.
- [ ] Fix the meta:1761 load crash. BISECT TABLE 2026-09-07 01:10 (all on build-tsplit
      = deea0a591 + timing-only instrumentation; lazy-divert stashed as
      "lazy-divert experiment"):
        1. uncensored + new 9-GPU mix, no divert: CRASH meta:1761
        2. uncensored + divert, new mix: CRASH
        3. censored baseline + divert, new mix: CRASH
        4. censored baseline, no divert, new mix (5090+3x3090+5x5060Ti): CRASH
        5. censored baseline, no divert, OLD mix (5090+8x5060Ti): CRASH
      → NOT my edits, NOT the model, NOT the GPU set. Remaining deltas vs the
      successful eval-branch bench (decode 47.3-47.8 measured ~2026-09-03):
      its exact invocation/build. NOTE: REPORT.md's `-sm tensor` example is the
      older 27B DENSE work (official lcpp build) — the MoE tsplit bench artifacts
      from Sep 3-4 live elsewhere (shell history, /tmp logs, tsplit-eval session
      dirs). NEXT-SESSION OPTIONS (pick either):
      (a) recover the original MoE tsplit invocation (grep zsh/bash history for
          `qwen38-tsplit-eval` era `-sm tensor`, /tmp/*tsplit*.log, ~Sep 3-4 mtime)
      (b) instrument the alloc loop: print (i, n_simple_bufts, ctx tensor count,
          bytes requested, backend name, backend free VRAM) before the assert at
          meta:1761 — one load identifies the failing backend + what it tried to fit.
      Also check: the successful bench may have used -ncmoe / --no-host / lazy-mode
      flags that shrink the GPU-resident set (the eval-era fork had them).
- [x] MYSTERY SOLVED 2026-09-07 01:35: commit 22bb4b9a's own message documents the
      66-GiB-single-range-alloc-on-device-0 as a KNOWN 9-GPU limitation ("planner
      needs work"). KNOWN-GOOD tensor config = 12-GPU pool, `-ts 1.3x12`, fp32 KV 262k:
      loads healthy, VRAM 14.2-17.4 GiB/card, decode 44.2 t/s temp-1.0, ~50.7 GiB PLE
      stays CPU-mmap'd. Also documented there: MTP composes only after deea0a591's
      draft-layer-split fix (2.8-4 t/s — the dispatch-tax victim), and the 12-way
      per-token sync/allreduce costs ~4.3 ms/token vs layer-split.
- [x] PROGRESS CHAIN 2026-09-07 01:44 — three distinct failure sites mapped, each
      deeper than the last:
      (1) llama-bench (any GPU count): weights ctx 1222 tensors/66 GiB lands UNSPLIT
          on device 0 (matches 22bb4b9a's documented 9-GPU note).
      (2) llama-server 12-GPU @262k fp32: WEIGHTS SPLIT SUCCEEDS (server load path
          differs from bench) — crash moves to indexer-cache ctx: 48 tensors/9.66 GiB,
          first=cache_idx_k_l3, OOM on CUDA4 (16 GiB card). ROOT CAUSE FOUND in code:
          llama-model.cpp ~line 495 `pattern_idx_cache → SPLIT_AXIS_MIRRORED`
          ("qsa indexer has one key head... cannot be split") + PLE r_cache also
          MIRRORED — every backend needs a full 9.66 GiB copy at 262k; with current
          residents (VoxCPM 6.9G / voice-tutor 4.9G on pool cards) the 16 GiB cards
          can't fit weights-share + mirror + KV + buffers. Original bench likely ran
          resident-free.
      (3) llama-server 12-GPU @65536 (indexer mirror ~2.4 GiB): loads PAST both,
          crashes at NEW site: ggml-backend-meta.cpp:1099 GGML_ASSERT(
          split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
          == sum * tensor->ne[split_state.axis]) — split-state SHAPE CONSISTENCY
          check (~the snapshot/validation region). NEXT DEBUG TARGET.
- [x] CRASH SITE 3 IDENTIFIED 2026-09-07 01:48 — the meta:1111 ratio assert fires on:
      `op=CONCAT name=node_37 axis=1 src1=linear_attn_qkv_mixed-0 (transposed) src_axis=1
      lhs=256*5*10240 rhs=640*10240 backend j=5/12` — factor-of-2 ratio mismatch.
      The LINEAR-ATTENTION (GDN/QSA) qkv_mixed tensor's split segments (from
      get_split_segments' QWEN4EXP branch: ssm_d_state/n_group-based key_dim/value_dim
      segmentation) are inconsistent with what the CONCAT consumer expects on some
      backends. THE CONCRETE NEXT FIX: align the qwen4exp qkv segmentation with the
      concat's ratio requirements in llama-model.cpp get_split_segments (256 vs 640
      = 2*320: likely key_dim vs 2*key_dim+value_dim style arithmetic on the
      transposed view).
- [x] CONFIG MAPPING 2026-09-07 02:00 (from raw config.json text_config):
      linear_key_head_dim=128, linear_num_key_heads=16, linear_value_head_dim=128,
      linear_num_value_heads=48, conv kernel 4. → key_dim = 128*16 = 2048;
      value_dim = 128*48 = 6144; 2*key_dim + value_dim = 10240 == the diagnostic's
      ne[axis] ✓. head_ratio = 48/16 = 3 (INTEGER) → {{2048, 2+3}} = 5×2048 is a
      CORRECT partition of qkv_mixed (2 key + 3 value segments). Therefore the
      factor-2 mismatch (lhs 256*5=1280 vs rhs 640) is NOT in the segmentation table
      but in the TRANSPOSE/CONCAT consumer arithmetic (src1 = "qkv_mixed (transposed)").
      NEXT-SESSION TARGET: read the GGML_OP_TRANSPOSE + GGML_OP_CONCAT split-state
      handlers (ggml-backend-meta.cpp:518-950 dispatch) and the ratio propagation
      through them; the 2× smells like the transpose swapping the axis-0/1 roles so
      the concat double-counts one dim. Graph context: qwen4exp.cpp:717-724 builds
      qkv_mixed (named linear_attn_qkv_mixed), :1260 feeds build_conv_state_at.
- [x] HANDLER-LEVEL ANALYSIS 2026-09-07 02:05 (handle_concat meta:562-575,
      handle_transpose meta:727-745): the ratio check at meta:1069-1105 fires on the
      CONCAT that INHERITED src0's split state (branch 3: `src_ss[0].axis ==
      src_ss[1].axis && != concat_axis → return src_ss[0]` — assumes ratios agree;
      they don't). src0: ne[5]=256 ×nr[0]=5 (=1280/backend-5) vs src1 (transposed
      qkv_mixed) sum=640 — src0 and src1 have DIFFERENT segment layouts on the same
      axis. handle_transpose is simple axis-swap (0↔1, keeps nr) — the transposed
      qkv_mixed carries its 5×2048 segmentation through onto axis 1. NEXT-SESSION
      DERIVATION: identify src0 of node_37 (likely the conv-state piece from
      build_conv_state_at, qwen4exp.cpp:1260 — r_cache configured SPLIT_AXIS_0 with
      ssm_out pairing) and either (a) make its segmentation match qkv_mixed's 5×2048
      (config-side), or (b) teach handle_concat to build a consistent merged state
      instead of inheriting src0 when ratios disagree (handler-side, more general).
      Handler-side (b) is the principled fix for all archs.
- [x] GRANULARITY SECTION FOUND 2026-09-07 02:10 (llama-model.cpp:655-686, the
      else/QWEN4EXP branch of the granularity lambda): qkv/conv1d/attn_gate/ssm_out
      → granularity = lcm(lcm(blck_size,128), ssm_d_state); r_cache → granularity ×
      (ssm_d_conv - 1) (=×3 for qwen4exp, conv kernel 4); s_cache → granularity ×
      ssm_d_state. The diagnostic's 640 = 5×128 aligns with qkv_granularity=128
      (blck 128) at 12 backends; src0's 5×256 layout is the coarser conv-state
      granularity footprint. ALL DERIVATION PIECES NOW MAPPED: segments fn
      (594-640) + granularity fn (655-686) + handle_concat (meta:562) +
      handle_transpose (meta:727) + the ratio check (meta:1069-1105). Next session:
      derive src0's exact identity from qwen4exp.cpp build_conv_state_at (:1260),
      then choose handler-side merged-state fix vs config-side segment/granularity
      alignment for qwen4exp r_cache, implement, test at 64k window.
- [x] SRC0 IDENTIFIED 2026-09-07 02:15 (qwen4exp.cpp:1487 build_conv_state_at
      internals + :1260 call): the failing CONCAT is the GDN conv-window SLIDE —
      concat(conv_states_all[old state, (d_conv-1) tokens × 10240 ch], qkv_mixed[new,
      transposed], axis=time). src0 = the old conv-state views (r_cache-shaped);
      src1 = transposed qkv_mixed. Both split on the FEATURE axis with different
      segment layouts (conv state granularity ×3 per the granularity fn). The fix
      must make the conv state's feature-axis segmentation IDENTICAL to qkv_mixed's
      5×2048 — either by configuring r_cache segments for qwen4exp as
      {{key_dim*(d_conv-1) per segment}}… see granularity fn interplay — or the
      handler-side merged state. NEXT SESSION: implement one of the two fixes,
      rebuild, 64k window. Everything else is mapped.
- [x] EXACT FIX LINES 2026-09-07 02:20 — the two r_cache segment patterns:
      QWEN3NEXT reference (llama-model.cpp:611-613): `{{key_dim*(d_conv-1), 2},
      {value_dim*(d_conv-1), 1}}`; QWEN4EXP target (:627-629):
      `{{key_dim*(d_conv-1), 2 + head_ratio}}` ← CHANGE THIS; granularity co-target
      (:684-686): r_cache → `granularity_qkv * (ssm_d_conv - 1)`.
      REQUIREMENT: conv-state per-backend channel share must equal (d_conv-1) × the
      qkv_mixed per-backend share. Observed: 256×5=1280 vs 640 (=2× off, not 3× —
      nr/ne semantics interplay; read the split_state struct def meta:~430-520 and
      the reshape order in build_conv_state_at :1487-1520 to pick between (a)
      step-major segments {{key_dim, (2+head_ratio)*(d_conv-1)}} + granularity_qkv,
      or (b) channel-major with qkv-aligned per-backend boundaries.
- [x] CONCAT STRUCTURE CONFIRMED 2026-09-07 02:25 (qwen4exp.cpp:1487-1515, exact):
      `state = ggml_reshape_3d(ctx0, rows, state_cols /*=d_conv-1*/, channels, n_seqs)`
      → `conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0)` — the
      concat is AXIS 0 (time), src0 = state [state_cols, channels=10240] whose rows
      come from conv_states_all via get_rows (build_rs, row_total = ne[0]); src1 =
      transposed qkv_mixed. BOTH split on axis 1 (channels) — the fix is purely:
      state's per-backend channel boundaries must equal qkv_mixed's. REMAINING
      AMBIGUITY: conv_states_all's exact axis-0 layout (channels×seqs vs time-major)
      determines whether segments {{key_dim*(d_conv-1), 5}} must become
      {{key_dim, 5} per time step} (i.e. {{key_dim, (2+head_ratio)*(d_conv-1)}}) or
      granularity alone fixes the boundaries. Read split_state struct def (meta:~430-520,
      ne/nr semantics) + conv_states_all allocation (grep build_conv_state_at caller /
      cache init in llama-context.cpp) — then implement, rebuild, 64k window.
- [x] FINAL PIECES 2026-09-07 02:35: (1) get_split_state's own FIXME (meta:~495):
      "preserves/erases the information in n_segments and nr in an inconsistent
      way… can lead to unexpected results" — the nr[0]=5 anomaly matches this
      documented inconsistency. (2) handle_get_rows (meta:747-752): src0 axis-0 +
      src1 mirrored → returns src state UNCHANGED (segments/nr ride through).
      (3) handle_reshape (meta:611+): axis-remapping logic with n_segments==1
      fast paths — the {{6144,5}} 5-SEGMENT state entering reshape takes the
      generic path where nr/ne get re-derived; the 1280-vs-640 (2× not 3×) is
      consistent with the window factor being folded into ne[] (256 = 640×... )
      mid-propagation. NEXT SESSION EXECUTES: instrument ggml_backend_meta_get_split_state
      (log name/axis/ne/nr per tensor for cache_r_l* + linear_attn_* nodes at
      graph-build time — one CPU-only test-backend run may suffice) → pinpoint the
      exact handler that breaks the channel-alignment → fix (likely: normalize
      nr/n_segments in handle_reshape for multi-segment axis-0→axis-1 remaps, or
      config-side {{key_dim, 5} per window}) → rebuild → 64k window → needle/t/s.
- [ ] Memory-placement fix for mirrored caches at 262k (crash site 2)
- [ ] 12-GPU instrumented run for rebuild-phase timings once load completes
- [ ] 9-GPU placement planner fix (single-range alloc on device 0)
- [ ] Implement chosen fix (A/B per profile)
- [ ] Bench protocol pass, commit on branch, PR-quality summary
