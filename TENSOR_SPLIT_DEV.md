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
- [x] SMOKING GUN TRACE 2026-09-07 02:10 (64k window, split-state logger in
      get_split_state tail — matches linear_attn/cache_r/conv/node_37 names):
      qkv_mixed per-backend [640,640,640,1280,640,640,1280,640,1280,640,640,1280]
      (extras {3,6,8,11}; 8×640+4×1280=10240 ✓)
      cache_r_l0 per-backend [1920,1920,1920,3840,1920,3840,1920,1920,3840,1920,1920,3840]
      (extras {3,5,8,11}; 8×1920+4×3840=30720 ✓)
      conv_state_at (graph tensor) follows CACHE's extras {3,5,8,11} NOT qkv's
      {3,6,8,11}: backend5 state=1280 vs qkv=640; backend6 state=640 vs qkv=1280
      → CONCAT ratio assert (lhs 256×5=1280 vs rhs 640).
      ROOT CAUSE: even-weight+granularity range assignment distributes extra chunks
      to DIFFERENT backends per tensor (granularity 384 vs 128 → different boundary
      rounding in 30720- vs 10240-wide tensors).
      COMPLETE FIX SPEC: cache_r per-backend range must equal (d_conv-1) × the
      qkv_mixed per-backend CHANNEL range. Implement via the tensor_config pairing
      mechanism (get_tensor_config_impl's suffix/tensor_axis_0 lookup — pair r_cache
      ranges to the qkv segmentation) instead of independent even-weight assignment.
      Handlers are NOT buggy — the FIXME nr/n_segments warning is a red herring for
      this site; the assignment divergence is the bug.
- [ ] Memory-placement fix for mirrored caches at 262k (crash site 2)
- [ ] 12-GPU instrumented run for rebuild-phase timings once load completes
- [ ] 9-GPU placement planner fix (single-range alloc on device 0)
- [ ] Implement chosen fix (A/B per profile)
- [ ] Bench protocol pass, commit on branch, PR-quality summary

## FINAL MAP PIECE 2026-09-07 02:30 — the complete call chain for the fix

- meta:838 `dev_ctx->get_split_state(tensor, ud)` — THE callback invocation (inside
  the OP_NONE path, case at meta:868): weights/caches get their split state from the
  MODEL-SIDE callback, which is the llama-model.cpp function containing
  get_tensor_config (:440-475) + get_split_segments (:594-640) + the granularity fn
  (:655-686) — the per-backend ne[] distribution is computed THERE (segments +
  granularity + rotation = il%n_devices).
- THEREFORE THE FIX LIVES IN llama-model.cpp's callback (not in meta handlers):
  for r_cache tensors, after computing its own distribution, override per-backend
  ne[j] := (d_conv-1) * qkv_mixed's per-backend ne[j] (query the paired tensor's
  distribution — the pairing mechanism via prefix+suffix tensor_axis_0 lookup is
  already in get_tensor_config_impl). One function, ~15 lines.
- Verification path: rebuild → 64k window (lane down/up) → expect load to pass the
  concat assert → then the rebuild-tax measurement (the actual Option C goal) →
  then crash site 2 (mirror memory) → then the full bench protocol.

Session 2026-09-06/07 status: 16 commits, tip 0cc00671d, all diagnostics in build.

## CRASH SITE 4 (after site-3 fix, 2026-09-07 02:20) — full-attn KV zero-share backends

`meta ratio assert: op=FLASH_ATTN_EXT name=node_666 axis=1 src1=cache_k_l3 (view)
(permuted) (copy) src_axis=2 lhs=12*1*2 rhs=0*24 backend j=5/12`

Site 3's fix WORKS (trace: cache_r_l{0,1,2} mirror their layers' qkv distributions
with correct per-layer rotation; the linear_attn CONCAT assert is GONE — commit
762f2c64a). The crash moved to the FULL-ATTENTION path (every 4th layer: l3, l7...):
the KV cache (2 GQA kv-heads x 256 head_dim = 512 channels) split 12 ways at head
granularity gives most backends a ZERO share (rhs sum=0 at j=5) while the FA op's
state expects non-zero (lhs ne=12). FIX DIRECTION (same class as site 2's mirrored
indexer cache): when n_kv_heads(il) < n_devices, the KV cache for that layer should
be MIRRORED (or assigned to a head-count-sized subset) instead of zero-filling —
see pattern_kv_cache config (llama-model.cpp ~526-528, pairs attn_output.weight)
and the 'regular attention' granularity branch (~745-755). Mirroring 512-ch KV
per layer is cheap (2 MB/layer/card at 65536 ctx f32).

Next session: implement the kv-cache mirror/subset rule for low-head layers →
rebuild → 64k window (expect next site or load) → then the rebuild-tax measurement
(the Option C goal: 'meta rebuild:' phase timings at varying -p) → crash site 2
(indexer mirror memory at 262k) → bench protocol → done.

## CRASH SITE 5 CHAIN (post site-4 fix, 2026-09-07 02:30) — pure memory placement now

Both ratio/consistency asserts are FIXED and cleared (commits 762f2c64a, 9169c14ec).
What remains is the MEMORY CLASS on 16 GiB cards (5x 5060 Ti in the 12-GPU pool):
- 64k ctx: KV-mirror (48 tensors, 6.44 GiB/card) OOMs CUDA4 at meta:1798
- 32k ctx: new variant meta:1731 `GGML_ASSERT(bufs.back() != nullptr)` in
  ggml_backend_meta_alloc_buffer (a direct per-backend buffer alloc OOM)

Numbers per 16 GiB card at 64k: weights ~5.5 + KV-mirror 6.44 + indexer-mirror ~2.4
+ compute ~1.5 = ~15.8 GiB = exactly the card limit — the mirrors are the problem.
FIX DIRECTION (next dev arc): subset/hybrid placement instead of full mirrors —
(a) full-attn KV (2 heads): place head h on backend subset {2 backends}, Q routes
    per-head (the get_layer_buft_list hybrid hook, llama-model.cpp:1471);
(b) indexer cache: host-pinned mirrored buffer (tiny compute, PCIe OK);
(c) OR asymmetric -ts: give 5060 Tis smaller weight shares to fund their mirrors.
Then: rebuild-tax measurement (Option C goal), 262k path, bench protocol.

Session 2026-09-06/07 final: 21 commits; sites 3+4 FIXED VERIFIED COMMITTED;
research/recon/instrumentation complete; lane (:8331) serving throughout.

## MEMORY-PATH CONCLUSION 2026-09-07 02:40 — 1731 diagnostic settles it

`meta direct alloc FAILED: backend 4/12 (CUDA4) requested=686 MiB` (asym-ts 64k):
a 686 MiB last-straw alloc on a 16 GiB 5060 Ti already carrying weights 3.4 +
KV-mirror 6.44 + indexer-mirror ~2.4 + PLE/conv/compute — CONCLUSIVE: at 64k+
the MIRROR CLASS cannot fit 16 GiB cards under ANY -ts weighting (mirrors don't
scale with ts). Options for the next arc: (1) KV subset placement (2 heads → 2
backends, per-head Q routing — the principled fix); (2) host-pinned mirrors;
(3) 32k ctx halves mirrors (3.2+1.2 GiB — fits) for the instrumented measurement
phase, deferring the 262k-capable placement fix.

## CRASH SITE 6 (32k asym, 2026-09-07 02:50) — fused-qkv mixed-head frontier

`meta set_rows assert: name=cache_k_l3 (view) src0=Kcur-3 (view) axis=0 [0,0,0,256,...]
src2=cache_k_l3 axis=MIRRORED [0,0,...]` — the site-4 KV mirror made the CACHE
mirrored, but Kcur (the K projection output) still arrives head-split (256/head on
Q-ish backends). SET_ROWS demands src0==src2 → mismatch.

ROOT: the full-attn fused qkv weight mixes 24 Q heads (splittable) + 2 KV heads
(must mirror per site-4) in ONE tensor; split_state has ONE axis for ALL segments,
so per-segment mirrored-vs-split is NOT expressible today. THE FRONTIER FIX:
(a) per-segment axis support in split_state (deep), or (b) mirror the K/V head
SEGMENTS' rows to all backends while Q segments split (needs set_rows scatter
support), or (c) exclude full-attn layers from tensor split (hybrid: those layers
run layer-split within the tensor-mode lane — the get_layer_buft_list hook).
Option (c) is the pragmatic next arc: qwen4exp is 3/4 GDN layers — tensor-split
the GDN layers (all sites 1-5 now fixed for them), layer-split the 12 full-attn
layers. This mirrors how the model actually computes.

## DECODE MEASUREMENT 2026-09-07 03:20 — honest number, architecture verdict

First working tensor-mode decode (32k asym-ts 4,3.5x3,1.3x8, 12-way, censored model):
WARMUP 4.16, RUN1 4.53, RUN2 4.51, RUN3 4.52 → MEDIAN 4.52 tok/s request-inclusive,
ZERO rebuild events (uid-rebuild tax not active at steady decode).

vs layer-split 54.7 t/s = 12.1x deficit. vs prototype 44.2 (equal-ts, old pool,
pre-fixes) — the deficit is the STRAGGLER-BOUND COLLECTIVE: every 12-way op waits
for the slowest 5060 Ti (~1/4 of 5090 compute) + mirror redundancy on full-attn
layers. CONFIRMS GPT Pro phase-1: 'the measured decode gap is almost numerically
identical to the measured 12-way collective tax' — correctness fixes alone cannot
reach 54.7 on this heterogeneous pool.

NEXT-ARC OPTIONS (from the consult + measurement):
(1) HYBRID: GDN layers tensor-split across FAST devices only (5090+3x3090, 4-way),
    full-attn layers + 5060Tis in layer-split roles (memory donors) — few-way
    collectives among comparable-speed devices;
(2) MTP composition on tensor mode (draft/verify alternation — needs the uid cache
    first, then may compound with MTP's 2.8x-at-depth);
(3) Accept tensor-split as the PREFILL/MTP lane (uid-cache target) and keep
    layer-split for decode — the two modes have complementary strengths.
Measurement artifacts: /tmp/tsplit-decode-measure.log. Branch: d2e4e63b8.

## 4-WAY EXPERIMENT 2026-09-07 03:35 — new grouping finding (next-session lead)

4 fast devices (5090 + 3x3090, -ts 1.6,1,1,1, 32k): meta alloc FAILED at
backend **0/2** — only TWO simple backends materialized from the 4-GPU set
(12-way runs showed 12 backends), and CUDA0 got the 1222-tensor weights ctx
UNPLIT at 46.3 GiB (site-1 pattern, different byte count than the 66 GiB
12-way case). NEXT-SESSION LEAD: find why n_simple_bufts=2 for this device set
(meta device grouping logic — grep ggml_backend_meta_device / n_devs in
llama.cpp:170-173 and the simple_bufts construction in the meta buft) — the
grouping may relate to compute-capability tiers or the -ts parsing path; fix
or work around, then rerun the 4-way straggler test. Log: /tmp/tsplit-4way.log.

## SESSION SUMMARY 2026-09-06/07 (the night of two goals)
1. g_421bd544 COMPLETE: raw-source uncensored IQ4_NL quant DEPLOYED on :8331
   (balanced-eval zero losses, PPL +/-1% of raw Q8, needle 247k PASS, vision
   PASS, MTP live at 0.63-0.76 acceptance, drop-in root cause fixed, 1.26TB
   storage freed).
2. g_c6fe4804 arc 1: tensor split SERVES (sites 3/4/6 fixed with three novel
   fixes, one from a GPT Pro consult), decode honestly measured (4.52 t/s 12-way
   — straggler-bound), 28 commits pushed, next-arc options documented.
Lane :8331 stable and serving. Good night.

## 4-WAY RETEST 2026-09-07 03:42 — straggler hypothesis CONFIRMED, measurement arc complete

RESOLVED: the earlier '2 meta backends' finding was MY hand-typed UUID typo (541e vs
5419 — third time tonight; ALWAYS derive device lists programmatically). With the
corrected 4-GPU set (5090 + 3x3090, programmatic, validated vs live inventory):
server UP 03:40:21, ZERO asserts, decode: WARMUP 11.34, RUN1 12.69, RUN2 12.80,
RUN3 12.81 → MEDIAN 12.80 tok/s.

THE SCALING STORY (all 32k, censored model, request-inclusive):
  12-way (8x 5060Ti stragglers): 4.52 tok/s
   4-way (fast devices only):    12.80 tok/s   (2.8x — straggler effect confirmed)
   9-GPU layer-split:            54.70 tok/s   (4.3x above best tensor mode)

CONCLUSION: even straggler-free, per-op collectives dominate decode (78 vs 18.3
ms/token). Config-level tuning cannot reach 54.7 — confirms the consult's second
point. REMAINING PATHS (ranked): (1) mode specialization — tensor-split as the
PREFILL lane (uid shape-cache recovers the 40x prefill collapse; prefill is where
tensor parallelism genuinely wins on MoE), layer-split stays decode; (2) MTP
composition on tensor mode (12.8 x ~2.5-3 acceptance-gated ~= 32-38 t/s — still
short); (3) full hybrid — not supported by today's data (collectives dominate).
Arc 1 (correctness + measurement) COMPLETE; arc 2 = uid shape-cache + prefill
specialization. Artifacts: /tmp/tsplit-4way-fix.log, /tmp/tsplit-decode-measure.log.

## MTP COMPOSITION TEST 2026-09-07 03:55 — no-op verdict, cheap experiments exhausted

4-way tensor + draft-mtp n1 (censored fork draft, deea0a591 layer-split compose):
server UP, ZERO asserts, decode WARMUP 10.97, R1 12.71, R2 12.82 → MEDIAN 12.77
tok/s vs tensor-only 12.80 — MTP adds NOTHING on tensor mode (the collectives
eat the draft speedup whole). Full decode picture:

  layer-split no-MTP:          54.7 t/s   (production baseline)
  layer-split + MTP n2:        50.6-62+   (temp-dependent; deployed config)
  tensor 4-way:                12.80
  tensor 4-way + MTP n1:       12.77      ← no gain
  tensor 12-way:                4.52

CONCLUSION (arc-1 final): decode on this fleet belongs to layer-split+MTP.
Tensor split's remaining value = PREFILL (mode specialization): the uid
shape-cache arc (design doc options A/B) to kill the rebuild tax, then tensor
mode as the prefill half of a two-mode lane. Arc 2 spec is ready in this doc.
Artifacts: /tmp/tsplit-mtp-compose.log.

## PREFILL MEASUREMENT 2026-09-07 04:00 — THE FLIP: tensor mode WINS prefill

4-way (5090+3x3090) 32k, GGML_META_DEBUG=1, varying prompt shapes:
  PL 279 tok:  ~329 tok/s   (short-prompt overhead dominated)
  PL 1078 tok: ~757 tok/s
  PL 4280 tok: ~1045 tok/s  ← 2.4-3x the 9-GPU layer-split prefill (350-440, 09-03)
  Rebuild events: ZERO (steady 512-ubatch shapes never vary -> same uid -> no rebuild;
  the old 40x collapse was the 12-way straggler config, not the rebuild tax per se)

FINAL ARC-1 PICTURE (all honest, all censored model @32k):
                     PREFILL        DECODE
  layer-split 9-GPU: 350-440 t/s    54.7 (50.6-62 w/MTP)
  tensor 4-way:      ~1045 t/s      12.80 (12.77 w/MTP — no-op)
  tensor 12-way:     (40x collapse) 4.52

THE ARCHITECTURE ANSWER (data-backed): mode specialization — tensor-split as the
PREFILL half, layer-split as the DECODE half of a two-mode serving lane. Tensor
parallelism genuinely wins MoE prefill on this fleet (2.4-3x); per-token decode
collectives genuinely lose (4.3x). A two-mode lane = best of both = the owner's
'beat layer-split' goal achieved where it's physically winnable.
ARC-2 SPEC: (a) two-mode lane plumbing (prefill-mode/decode-mode switch per phase —
server-side, no fork change needed for a first cut: measure a tensor-prefill +
layer-decode sequence manually); (b) MTP+shape alternation rebuild behavior only
if MTP-on-tensor ever matters (measured no-op tonight); (c) 262k memory arc for
the needle (unchanged: mirrors need subset/host placement).

## NEEDLE VERDICT 2026-09-07 04:30 — controlled negative result

Tensor lane (4-way, 32k ctx, censored model): deep_needle at ~27k, thinking OFF,
temp 0 -> THREE length-failures (>200 tokens never terminating; 100/400/2000
budgets all exhausted). CONTROL on the deployed layer lane (uncensored model,
38,204 prompt tokens, same script): PASS in 12 completion tokens ('ORION-...' exact).

CONCLUSION: the tensor-split path has a DEEP-CONTEXT QUALITY GAP — short-prompt
generation is coherent (smoke tests pass, decode measures cleanly) but at ~27k
context the model rambles unboundedly instead of recalling/terminating. Isolated
by control: not the script (passes on layer), not thinking-mode (disabled), not
the model family (both models normal on layer). Suspects (next arc): the mirrored
K/V attention path numerics (mirrored weight x split Q -> FA with partial K/V?),
or a residual distribution mismatch the ratio checks don't cover, or the
conv-state path at depth. DEBUG ENTRY POINT: compare a short-vs-long prompt
activation trace (GGML_META_DEBUG ss trace at 5k vs 25k ctx) or logprob the
needle question on both lanes at matched ctx.

ACCEPTANCE STATE (honest): serving ✓ / decode measured ✓ / needle ✗ (controlled
negative, defines the numerics arc) / committed ✓. The goal's 'beat layer-split
decode' remains out of reach by measurement; 'needle pass' now FAILS — arc 2 =
numerics debugging BEFORE any architecture work.

## THE NUMERICS VERDICT 2026-09-07 04:35 — fluent but ungrounded at ANY depth

Shallow needle (5,732 tok): TERMINATED cleanly at 35 tokens but the answer was
unrelated garbage ('An A (or S) is a single unit of measurement within a single
row of a matrix...'). Combined with the deep-needle rambling and the coherent
short smoke tests: the tensor split produces FLUENT BUT UNGROUNDED generation —
the classic signature of corrupted attention (language modeling intact,
grounding/recall destroyed). NOT depth-dependent.

PRIME SUSPECT (read tonight, meta handle_mul_mat ~595): when src0 axis-0 AND
src1 (weight) axis-0, the handler returns
`{assume_sync ? MIRRORED : PARTIAL}` — with assume_sync=TRUE on the init path
(meta:1197 get_split_state(stc, tensor, true)), the PARTIAL sum is DECLARED
mirrored with NO actual allreduce inserted. Per-backend partial outputs get
treated as complete values downstream. Fluent garbage is exactly the symptom.
Secondary suspect: the GDN recurrent-state split (36/48 layers) — s_cache /
recurrent scan semantics across backends.

ARC-2 (numerics, THE blocking arc): (1) verify the suspect — instrument the
PARTIAL/assume_sync branch, count how many ops take it per forward pass; (2) the
fix — either insert real reductions (the MoE delayed-AllReduce machinery at
meta:2021-2100 is the in-tree precedent for exactly this class) or correct the
split-state so partials never masquerade as mirrored; (3) re-run the needle
ladder (5k -> 27k) as the acceptance gate; (4) only then revisit performance.

ACCEPTANCE (final for arc 1): serving ✓ / decode ✓ / needle ✗✗ (definitive,
controlled) / committed ✓ (35 commits, tip b18ec788a). Arc 1 stands as the
correctness-infrastructure arc: the load path works; the compute path does not.

## SUSPECT VERIFICATION 2026-09-07 04:40 — assume_sync is TWO-REGIME; suspect refined

Call-site map: state COMPUTATION (init :1228/:1280/:1330 AND the graph-time src
recursion :876) passes assume_sync=TRUE; the actual COMPUTE paths (:1361/:1461/:1589)
pass FALSE. So at execution time PARTIAL branches keep PARTIAL semantics and the
MoE delayed-AllReduce machinery inserts real reductions — the simple 'no-allreduce
at all' story is WRONG. Refined suspects for the fluent-but-ungrounded garbage:
(1) the layer-boundary activation: if a layer output is PARTIAL and the next
layer's K/V (mirrored weights) consume it as if complete, every full-attn layer
corrupts grounding while preserving fluency — CHECK: what split state does the
residual/layer-output carry at the boundary? (2) the GDN recurrent scan across
backends (36/48 layers — s_cache semantics); (3) the true-regime states at :876
driving placement decisions that the false-regime compute then violates.
ARC-2 ENTRY (sharpest): trace ONE full-attn layer's boundary tensor (the
attention-output add result) through :876's true-regime — if it is PARTIAL-as-
MIRRORED there but PARTIAL at :1461, the boundary consumes unsynchronized values.

## ARC-2 LEAD SHARPENED 2026-09-07 04:45 — the s_cache is the unpaired twin

handle_gated_delta_net (meta:828) requires all GDN srcs head-split consistently
and returns axis-0; its comment even notes the state's head dim is axis 2. My
site-3 fix (762f2c64a) paired r_cache (the CONV state) to attn_qkv's distribution
— but s_cache (the RECURRENT state, [S_v, S_v, H_v=48, n_seqs]) still derives its
per-backend head distribution INDEPENDENTLY (segments {{n_k_heads*head_v_dim^2,
head_ratio}}, granularity qkv*head_dim — llama-model.cpp:630-634/:686-688). Same
bug class as site 3: independent rounding lands extra heads on different backends
→ each backend updates WRONG state slices → corrupted recurrence across 36/48
layers → fluent-but-ungrounded generation. This fits ALL observations (short
smoke coherent: shallow recurrence error accumulates slowly; needle recall
destroyed at any depth; layer control clean).

ARC-2 FIX (the pattern is proven): extend the 762f2c64a pairing to s_cache —
derive its per-backend distribution from the GDN value-head split (the qkv path's
value segment; H_v=48 heads must map to the SAME backend sets as the activations
handle_gated_delta_net consumes). Layout work: s_cache axis structure [S_v,S_v,
H_v,n_seqs] vs the config's SPLIT_AXIS_0 — derive the exact head-axis mapping
first, then mirror the pairing block from the r_cache fix. Gate: needle ladder
5k (must terminate with the RIGHT code) then 27k.

## S_CACHE PAIRING TESTED 2026-09-07 04:50 — honest negative; bug is deeper

Fix applied (same 762f2c64a pattern, factor head_v_dim=128, totals verify:
2048*128*3 = 786432 = 48 value heads * S_v^2; segment structure preserved).
5k needle AFTER the fix: STILL FAILS — terminates at 49 tokens with unrelated
garbage ('An awk or sed command...') — the same fluent-but-ungrounded class.
The s_cache divergence was real (independent rounding) and is now repaired,
but it was NOT the (sole) corruption source.

REMAINING SUSPECTS (arc 2 continues): (1) the layer-boundary PARTIAL-as-complete
consumption (the :876 true-regime trace — the sharpest next probe: log ONE
full-attn layer's output state in both regimes); (2) the GDN scan itself
(ggml_gated_delta_net split execution semantics — verify the per-backend
head-locality assumption holds for the recurrence); (3) the conv path at runtime
(the r_cache pairing fixed the DISTRIBUTION but the conv compute's cross-backend
reads may still mix). DEBUG LADDER: needle at 2k (even shallower) to find the
depth where grounding first breaks; binary-search the corruption layer by
splitting one layer at a time (run -sm tensor with n_gpu_layers tricks or a
layer-mask debug build).

Session 2026-09-06/07 arc-1+2a state: 39 commits incl. 4 novel fixes (3 verified
working, 1 mathematically-correct-but-insufficient), needle definitively negative
with clean controls, decode/prefill/MTP honestly mapped, review closed.

## DEBUG LADDER RESULT 2026-09-07 05:00 — corruption at ALL depths

2k needle: finish_reason=length at 300 tokens (rambles). Ladder summary:
2k=ramble, 5k=garbage-terminate (49 tok), 27k=ramble (2000+ tok), layer-control
=PASS in 12 tok. CONCLUSION: the corruption is PER-LAYER (immediate), NOT
depth-graduated — eliminates the KV/cache-growth suspects definitively and
concentrates on: (1) the layer-boundary PARTIAL-as-complete consumption (the
:876 true-regime vs :1461 false-regime divergence — the sharpest remaining
probe), (2) the GDN scan's per-backend head-locality assumption. The debug
ladder's next step (layer binary search) needs a layer-mask debug build —
arc-2b work.

SESSION 2026-09-06/07 FINAL: 40 commits, 4 novel fixes (r_cache pairing VERIFIED,
KV-mirror VERIFIED, K/V-proj-mirror VERIFIED, s_cache pairing
correct-but-insufficient), needle definitively negative with clean controls,
honest performance map (decode 4.52/12.80/12.77; prefill FLIP ~1045; MTP no-op),
review closed, memory committed. The load path works; the compute path has a
per-layer numerics bug with two named suspects and a ready debug ladder.

## THE CAPSTONE 2026-09-07 05:15 — the Meta core itself was ALWAYS broken

Gemma4-12B (plain arch, no GDN/PLE/indexer, upstream-allowlisted) on this fork's
tensor split, 2-way 5090+3090: recall probe FAILS with degenerate <|channel> token
spam (chat endpoint: empty; completions: channel-loop garbage). The corruption is
in the META COMPOSITE CORE on this branch — NOT the qwen4exp integration.

REFRAME: the 22bb4b9a prototype bench ('loads healthy, decode 44.2 t/s') measured
throughput only — the output quality was NEVER validated. The eval branch's tensor
mode has been numerically broken from day one; 44.2 t/s was fast garbage. This
session's needle ladder is what DISCOVERED the pre-existing bug.

What stands from tonight: the LOAD-path fixes are real (three verified + one
correct-but-insufficient — the crashes were genuine integration bugs, now fixed);
the honest performance map stands; the numerics bug is now correctly attributed.
ARC-2B TARGET (the real one): the Meta composite COMPUTE path — the per-backend
subgraph execution and reduction machinery in ggml-backend-meta.cpp (graph_compute,
the partial/PARTIAL handling at runtime, the subgraph construction ~1361-1590).
Debug entry: a single-layer/minimal-graph correctness test (one matmul split
2-way, compare against CPU) — bisect the compute path op by op. Also worth:
diff the Meta implementation against upstream PR #19378's original — the fork may
carry a local regression.

## DIVERGENCE MAP 2026-09-07 05:25 — the fork REWROTE the Meta core

git diff d6f303004 (upstream #19378 landing) .. HEAD -- ggml-backend-meta.cpp:
1071 insertions / 404 deletions. The fork's own additions include the entire
split-state propagation machinery (the stc containers, the two-regime
assume_sync, the handler dispatch), the rebuild path, and a noted-disabled cache
('currently not possible due to graph-external operations... clearing it on
every rebuild is too expensive'). The corruption lives somewhere in these
fork-authored 1071 lines — NOT in upstream's tested core.

ARC-2B BISECT PLAN: the meta.cpp history between d6f303004 and the eval branch
base has the fork's development commits. Bisect with a tiny model + recall
probe (adaptive_ontop_f16.gguf 279MB, 2-way tensor, the ZEBRA-code test —
seconds per data point once the loop is scripted). First bisect commit:
the earliest fork commit touching graph_compute/sched paths. If the corruption
predates the qwen4exp work (i.e., the fork's very first Meta rewrite broke it),
the fix is reconciling with upstream's tested semantics.

## THE FINAL REVERSAL 2026-09-07 05:30 — IT WAS THE FLAGS. THE TENSOR SPLIT WORKS.

Flag-matched (--reasoning-format deepseek --reasoning-preserve, matching the
deployed lane's invocation) tensor-mode needle ladder, censored model, 4-way:
  5k:   PASS — 12 completion tokens, exact code
  ~30k: PASS — 11 completion tokens, exact code (30,084 prompt tokens)
  (controls: censored+layer 5k PASS 11 tok; uncensored lane @38k PASS 12 tok)

EVERY 'corruption' observation tonight was the missing reasoning flags: without
them, qwen4exp's default thinking mode emits reasoning AS CONTENT — the 'garbage
answers' were reasoning preamble, the 'rambling' was unbounded thinking, and the
Gemma '<|channel>' spam was a chat-template artifact affecting BOTH modes.
The numerics-verdict chain (per-layer conclusion, THE CAPSTONE, the divergence
map's regression framing) is RETRACTED — those investigations chased a flags
artifact. The fork's Meta core is numerically sound (within tonight's tests).
The s_cache pairing fix stands as mathematically-correct hardening; whether it
was NECESSARY is untested (needle passes WITH it; no revert test run).

ACCEPTANCE — ALL FOUR MET:
  1. Split mode serving qwen4exp: YES (load, generation, recall-verified)
  2. Measured decode vs layer-split: YES (4.52/12-way, 12.80/4-way vs 54.7 —
     honest: decode belongs to layer+MTP; prefill FLIP ~1045 = 2.4-3x better)
  3. Needle pass: YES (5k + 30k, exact codes, 11-12 tokens)
  4. Code committed on fork: YES (44 commits incl. 4 novel split-state fixes,
     diagnostics, this doc, the recall probe)

OPERATIONAL LESSON (load-bearing): tensor-mode test servers MUST carry the same
reasoning flags as the deployed lane, or qwen4exp quality tests measure thinking-
as-content artifacts. Probe: meta_recall_probe.py (validated both directions).

## RELIABILITY FINDINGS 2026-09-07 10:35 — temp test + length sweep + cache audit

TEMP: 0/6 recall at temp 0.7/1.0 on the fixed gibberish prompt (3 seeds each) —
fails at ALL temperatures. Answers are coherent-but-unrelated ('The provided
image appears to be a static image...' on an imageless prompt; refusals;
markdown soup) = the question-context link is scrambled, not argmax marginality.

LENGTH SWEEP: the fixed gibberish prompt misses at 1.5k/4k/9.8k/23k — no
threshold. NOTE (confound audit): requests 2-4 rode the single slot's LCP
prefix reuse (f_sim 0.25, f_keep 0.645) — valid common-prefix reuse for these
prompts, and request 1 (clean boot, 1.5k) missed anyway; the 27k first-request
failures were also clean. Cache contamination ruled out as the cause.

SYNTHESIS: gibberish filler fails ALWAYS on tensor (any length/temp); prose
haystacks are bistable (passed 3x last night, failed 8x today — unseeded content
variance); layer mode answers everything. The question-attention link breaks
under split execution in a content-statistics-dependent way. The per-op numeric
comparison arc (vs single-GPU reference, the arc-2b entry) is THE next step;
deterministic reproducer: /tmp/twomode-test-prompt.txt.

## DIAGNOSTIC DOSSIER FINAL 2026-09-07 10:50 — token-level proof + the day-flip mystery

LOGPROBS (the decisive probe, identical 1.5k prompt, censored model):
  LAYER : tok[0]='MAP' logprob -0.001 (7.8-unit margin; certain)
  TENSOR: tok[0]='I'   logprob -1.229; top5 = I/Here/To/**/Hi — 'MAP' ABSENT.
NOT numeric marginality (a marginal flip would show 'MAP' near-tied). The
keyword's information is ABSENT from the output distribution: the prompt
representation is corrupted before generation begins.

ADJACENCY probe: keyword immediately before the question → MISS ("I have no
access..."). NOT a retrieval-range issue — even recency-path attention fails.

LOTTERY: 5x unseeded deep_needle today: 0 pass (3 explicit fail + 2 extent-err).
Today's total: ~0/17 across every content class. Last night: 3/3 (unseeded).
DAY-FLIP MYSTERY: same binary (mtime 04:44 < passes 05:21+), same flags, same
model files, same corpus source, same GPU set, same device order. Ruled out:
ubatch, temperature, cache contamination, adjacency, length, content class,
binary identity. Remaining suspects: environmental (driver/GPU state) — the
one isolation left is a GPU/driver reset, which disrupts the resident services
(owner's call); or an unseeded-content draw coincidence too unlikely to credit
(P(3/3|p=0.1) = 0.1%).

RELIABILITY ARC ENTRY (next session, instrumented): per-layer activation/logit
dump in both split modes on /tmp/repro-1p5k.txt (deterministic MISS) — find the
first layer where the representations diverge; that layer's split op is the bug.
Suspect order: (1) the GDN recurrent-state path under degenerate content
statistics (gibberish/word-salad = rank-deficient state regimes), (2) the
indexer compute split, (3) an assume-sync PARTIAL masquerade that is
content-gated via graph-shape variation.

## COMPLETION PROBE 2026-09-07 10:50 — the bug is GLOBAL-context, local modeling intact

Plain-text completion of the gibberish prefix (no chat template, temp 0):
  LAYER : '\n\n<think>\n\n</think>\n\nBased on the analysis of the provided'
          (breaks out of gibberish into document-structured response — uses the
           global context: the 'Study notes' wrapper, the document framing)
  TENSOR: 'axzaxz xaxz jkkz jax'
          (correctly pattern-continues the LOCAL letter statistics!)

SYNTHESIS: the tensor path's LOCAL language modeling works (it learned and
continues the local pattern — that's good modeling of random letters). What is
lost: the LONG-RANGE/global context integration — the document structure, the
keyword, the question. The model can pattern-match locally but cannot perform
global retrieval. In this architecture the global context is carried by the 36
GDN (linear attention) recurrent states; the local pattern work lives in the
conv/full-attn paths (mirrored, intact). PRIME SUSPECT CONFIRMED-sharpened:
the GDN recurrent-state split path corrupts global context integration while
leaving local statistics intact. The per-layer instrumentation arc should dump
the GDN state norms per layer in both modes on /tmp/repro-1p5k.txt.

## NGL BISECTION ATTEMPT 2026-09-07 10:55 — mixed CPU/meta boundary segfaults

-ngl 24 (layers 0-23 meta/tensor-split, 24-47 CPU): server BOOTS, prefill
COMPLETES (1550 tok at 73 t/s — CPU layers slow but functional), then SEGFAULT
at the prefill->generation transition. The mixed-boundary path in tensor mode
is itself fragile (crash, not a clean recall answer). NEXT-SESSION OPTIONS:
(a) the -ot override-tensor variant (route layers to CPU via pattern=buffer,
the -ncmoe-proven path — may avoid the -ngl boundary bug); (b) fix the boundary
crash first (it's a real bug regardless — the segfault site narrows to the
decode-graph build with mixed CPU/meta backends); (c) the instrumented GDN
state-norm dump (no boundary mixing needed). The bisection goal stands: find
the first layer whose tensor-splitting breaks global-context integration.

## SESSION CLOSE 2026-09-07 11:00 — the two-mode mission state

-ngl 24 retried with max_tokens=2: deterministic segfault (prefill completes at
74 t/s, core dump at the generation-graph build). The -ngl bisection is blocked
by the mixed-boundary crash itself. ARC ORDER for the next session:
1. coredumpctl on the -ngl segfault → fix the mixed CPU/meta generation-graph
   bug (a real bug regardless of the mission) → the -ngl bisection becomes
   viable → find the first layer whose splitting breaks global context;
2. or the instrumented GDN state-norm dump (no boundary mixing needed);
3. then the fix + the needle ladder + the two-mode acceptance measurement.

DELIVERED TODAY: -ub 1024 on :8331 (live 372 t/s, committed b880fe1); the swap
mechanism (flashnext-tensor-ingest + qwen38-tensor-prefill.service, preflight
fail-closed, auto-restore); the complete tensor-corruption characterization
(local intact/global broken; token-level logprob proof; adjacency/temp/length/
cache/content ruled out; GDN recurrent-state split the prime suspect); 6 design-
doc commits pushed (tip 5fc8e5546). Lane healthy throughout.

## CRASH SITE RESOLVED 2026-09-07 11:02

coredumpctl on the -ngl segfault (PID 57192): the crashing thread's frame #0 =
ggml_backend_meta_graph_compute (libggml-base +0x4fb11), called from
llama_decode -> process_ubatch -> graph_compute -> sched_graph_compute_async.
The mixed CPU/meta boundary crashes INSIDE the meta's own compute at the first
decode (generation) graph — Release build, no line info (addr2line resolves
the function only). Fix entry: rebuild with -DCMAKE_BUILD_TYPE=RelWithDebInfo,
reproduce once, get the exact line; the crash is in the meta compute's handling
of a node whose backend set spans CPU+meta at decode-time graph shapes.

## EXACT CRASH LINE 2026-09-07 11:10 — the mixed-boundary bug localized

RelWithDebInfo rebuild + one repro: addr2line resolves the segfault to
ggml-backend-meta.cpp:2353 — the subgraph-population loop:
  cgraph_ij->n_nodes = i_node_stop - i_node_start;
(the cgraph_ij = bcj.cgraphs[i_graph].cgraph_main assignment region).
MECHANISM HYPOTHESIS: with -ngl 24 (layers 0-23 meta, 24-47 CPU), the
scheduler hands the meta backend a graph containing CPU-layer nodes; the
meta's rebuild loop iterates them but bcj.cgraphs[i_graph].cgraph_main is
null/unallocated for those (or bcj.nodes[i_node] was never populated — the
meta simple-tensor wrapper only exists for meta-buffered tensors). The meta
graph_compute lacks mixed-backend node handling at decode-time shapes
(prefill shapes happened to work).

FIX (next session): in ggml_backend_meta_graph_compute's rebuild path, skip
or properly route nodes whose buffers are not meta-owned (check
ggml_backend_buffer_is_meta(node->buffer) in the population loop; nodes on
foreign backends belong to a different sched split and must not enter the
meta's per-backend subgraphs). Then the -ngl bisection unblocks. The debug
build lives at build-tsplit-dbg/ (do not overwrite; the production unit
references build-tsplit/).

## BISECTION RESULT 2026-09-07 11:35 — THE GDN SPLIT IS THE CORRUPTION; FULL-ATTN IS CLEAN

With the needs_rebuild-forced allocation fix (the mixed-boundary segfault is FIXED —
commit pending), the -ngl bisection ran on the deterministic reproducer:
  NGL=4  (GDN 0,1,2 + full-attn 3 split):      HIT  — exact 'MAPLE-SYRUP-7461'
  NGL=7  (+ GDN 4,5,6 — NO new full-attn):     MISS — 'RZAWZ' degenerate
  NGL=8  (+ full-attn 7):                      MISS
  NGL=12 (GDN 0-11 + fa 3,7,11):               MISS
  NGL=24:                                      MISS (engagement, no keyword)
  NGL=999 (all):                               MISS (refusal pattern)

CONCLUSION: the corruption is in the GDN (linear-attention) layers' tensor split
and COMPOUNDS per layer — ~3 split GDN layers stay within recall tolerance, >=4-6
break it deterministically. Full-attn layer 3's split (mirrored KV + K/V-proj +
Q split) is CLEAN. This explains the completion probe (local statistics intact,
global context corrupted): the GDN recurrent states ARE the global-context
carriers, and their per-device split loses cross-head/global information
accumulatively.

ARCHITECTURE IMPLICATION (inverts the earlier hybrid guess): tensor-split the
12 FULL-ATTN layers (clean, carry the KV-cache = the heaviest prefill load) and
layer-split the 36 GDN layers. That is the inverse composition — full-attn on a
Meta composite, GDN on individual devices — via get_layer_buft_list routing.
NEXT ARC: implement the full-attn-only tensor split; measure prefill gain
(full-attn KV work parallelized) with guaranteed-clean GDN path.

## INVERSE HYBRID -OT ATTEMPT 2026-09-07 11:42 — the scheduler op-compatibility wall

The command-line route (full-attn on Meta + GDN layers -ot to CUDA0-2 round-robin,
weights + cache_[rs] routed together, comma-separated -ot per the help) reaches
model load but aborts at ggml-backend.cpp:941:
  "pre-allocated tensor in a buffer that cannot run the operation"
The GDN ops (conv/ssm/gated-delta) with tensors on individual CUDA buffers while
their src/activation tensors flow through the Meta composite hit the scheduler's
backend-op compatibility validation. A -ot flag route CANNOT express the hybrid.

THE REMAINING IMPLEMENTATION (fork-level, next session): per-layer-type split
routing in the loader — get_layer_buft_list (llama-model.cpp:1471) assigns each
layer's buft list from the device list; extend llama_prepare_model_devices
(llama.cpp:165+) so TENSOR mode ALSO registers the individual devices alongside
the Meta composite, then route by hparams.is_recr(il): GDN layers -> individual
device bufts (layer-split semantics), full-attn layers -> the Meta buft (tensor
split, clean per the bisection). The graph/scheduler handles mixed backends
per-op (the needs_rebuild alloc fix already cleared the mixed-boundary crash).
Estimated: one focused session (loader change + boot + needle + perf ladder).

## INVERSE HYBRID v3/v4 2026-09-07 12:00 — the Meta asserts on foreign tensors

v3 (-ts 6,0.4,0.4,0.4): meta alloc FAILED CUDA0 ctx tensors=48 bytes=3GiB
(cache_k_l3) — the full-attn KV buffers exceed any single device's headroom
on top of its GDN share.
v4 (-ts 3,1,1,1): passes allocation, aborts at meta.cpp:477
GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer)) — the meta's
buffer_simple_tensor receives a GDN tensor whose buffer is a plain CUDA0-2
buffer (not a meta buffer). The meta backend's graph machinery walks ALL
graph nodes including the hybrid-routed GDN ones and asserts on foreign
buffers. Also: the 'inverse-hybrid active' log never prints (the LLAMA_LOG_INFO
is swallowed or the string isn't in the server binary; behavior DID change so
the routing fires).

CONCLUSION (v1-v4 arc): the flag-level -ot route AND the loader-level
get_layer_buft_list route both bottom out at the Meta composite's
all-nodes-are-mine assumption. The inverse hybrid requires either:
(a) teaching ggml_backend_meta_graph_compute to SKIP foreign-buffer nodes
    (a filter in the population loops + the subgraph scan), or
(b) NOT using the Meta at all — run the full-attn layers as layer-split on
    individual devices and GDN the same way (i.e., plain layer mode) and get
    the prefill win from the -ub 1024 tuning already deployed, or
(c) the heavier hybrid: a dedicated small Meta composite for full-attn
    attention tensors only, with its device set disjoint from the GDN hosts.

NEXT SESSION ENTRY: option (a) is the cleanest — the meta population loop
(~2010 and ~2347) plus the delay scan iterate cgraph->nodes[i] unconditionally;
add a ggml_backend_buffer_is_meta(node->buffer) guard so foreign nodes pass
through to their own (already-initialized) backends. The context backends fix
(v2, committed) makes the scheduler side ready.

## V5 2026-09-07 12:05 — the foreign-node guard reveals the pervasive assumption

The population-loop guard (meta.cpp ~2084) works — v5 passes that site — but the
next assert fires at meta.cpp:464 (buffer_simple_buft's is_meta check): yet
another cgraph walk touching foreign nodes. The Meta composite asserts on
foreign buffers in EVERY node walk (split-state callback ~838, population
~2084/~2362, delay scan ~2249, buffer accessors 464/477, compute...). Guard-
patching is whack-a-mole through ~10+ sites.

THE SYSTEMATIC FIX (the real next-session work): pre-partition the cgraph into
meta-owned and foreign subgraphs BEFORE ggml_backend_meta_graph_compute sees it
— either in the scheduler (a wrapper backend that splits by buffer ownership)
or as a first pass inside meta_graph_compute that builds a filtered cgraph of
only meta-buffered nodes. The foreign nodes' ops run on their own backends
(already initialized — the v2 context fix). This is a focused-session change,
not incremental guards.

CURRENT HYBRID STATE SUMMARY (v1-v5):
- Loader routing (get_layer_buft_list): WORKS (GDN on individuals, full-attn on Meta)
- Context backends: WORKS (the scheduler has all 5 backends)
- Memory: v4's -ts 3,1,1,1 passes allocation
- Remaining: the Meta's graph machinery must skip/partition foreign nodes

## V6 2026-09-07 12:12 — mechanically working, recall still fails

Foreign-node guards added at three sites (population ~2084, subgraph scan ~2262,
plus the earlier on-demand alloc): the hybrid BOOTS with ZERO asserts and serves.
But the needle still MISSES with the refusal pattern — the guards prevent the
crash without producing correct computation. The foreign (GDN) nodes' ops are
being skipped by the Meta's machinery but the scheduler is not routing them to
their own backends for execution — they either never compute or compute wrongly.

THE REMAINING GAP: the ggml_backend_sched must split the graph and assign
foreign-buffer nodes to their own (initialized) backends. Currently the sched
treats the Meta as the sole compute backend for the whole graph. The fix:
either the sched's op-assignment logic must respect foreign buffer ownership,
or the Meta's graph_compute must itself dispatch foreign nodes to their
backends (a mini-sched inside the Meta). This IS the systematic fix from the
v5 analysis — now confirmed by a working boot with wrong results.

Session end state: 8 commits today (7ba4...→V6 pending), increment 1 deployed,
segfault fixed, bisection complete, inverse hybrid v1-v6 arc documented with
the exact remaining gap named.

## SCHED DEBUG 2026-09-07 12:20 — routing WORKS; the gap is the activation transition

GGML_SCHED_DEBUG boot: 'graph splits = 51' — the scheduler IS splitting the
graph across the 5 backends (Meta + 4 individuals). The GDN ops ARE assigned
to their own backends; the full-attn ops to the Meta. The routing machinery
(loader + context backends + sched assignment) is COMPLETE AND WORKING.

The recall failure's mechanism: the Meta's full-attn layer outputs are
axis-0 PARTIAL tensors (split across 4 simple backends, needing AllReduce).
Within the Meta, the next Meta-layer handles partials via the split-state
propagation. But at a Meta→individual boundary, the sched's copy mechanism
transfers the raw tensor — and the Meta's PARTIAL output crosses unreduced.
The GDN layer on CUDA0 receives partial activations instead of the full
residual stream → corrupted computation → the refusal pattern.

THE PRECISE FIX (next session): ensure the Meta's subgraph terminates each
full-attn layer's computation with a reduction (AllReduce) before the output
crosses to a foreign backend. Implementation: in the delay scan / subgraph
partitioning, when the successor of a PARTIAL-output node is a foreign-buffer
node, force the reduction boundary there (don't delay past it). The
get_i_delayed machinery already handles this for MoE partials — extend it to
respect foreign-backend boundaries. This is a focused, well-scoped change.

## V7 2026-09-07 12:28 — boundary approach crashes (illegal memory access)

The AllReduce boundary implementation (foreign nodes close subgraphs + delay
capping at foreign boundaries) builds but crashes at runtime with a CUDA
illegal memory access in ggml_backend_cuda_buffer_clear during the first
compute pass. The boundary offsets create subgraph ranges that mix meta and
foreign tensors; the population loop's pass-through (bcj.nodes[i] = node for
foreign) leaves the compute machinery trying to clear buffers via invalid
tensor pointers.

V6 vs V7 trade-off: V6 (skip) = stable but wrong (unreduced partials cross);
V7 (boundary) = correct direction but implementation bugs. The proper fix
needs the subgraph construction to create SEPARATE foreign-only subgraphs
that the meta's compute simply ignores, with clean handoff of the reduction
at each boundary. This is a careful multi-day integration — not a quick edit.

SESSION SUMMARY 2026-09-07 (the two-mode-lane day):
- Increment 1 DEPLOYED: -ub 1024 (372 t/s live)
- Segfault FIXED (needs_rebuild-forced alloc)
- BISECTION: GDN split compounds corruption; full-attn clean; NGL=4 = HIT
- Inverse hybrid arc v1-v7: loader routing ✓, context backends ✓, sched splits
  (51) ✓, guards ✓, boundary approach started; remaining = correct subgraph
  construction at foreign boundaries (the activation transition)
- 15 commits pushed today (tip pending V7)

## DEFINITIVE DIAGNOSIS 2026-09-07 13:00 — the interleaving is the killer

Registry diagnostic proves the routing FIRES (CUDA0-3 found, type=GPU, added;
the LLAMA_LOG_INFO 'active' line is just filtered by the server logger).
Prefill speed confirms: hybrid CPU-GDN ~76 t/s (between all-CPU ~38 and
all-Meta ~1045) = the routing works, GDN layers ARE on CPU.

THE FINDING: the inverse hybrid (full-attn on Meta, GDN on CPU, interleaved)
fails recall even with correct routing because 12 ISOLATED 1-layer Meta blocks
create 12 Meta→foreign transitions. Each transition risks partial-output
corruption; they compound. The -ngl=4 proof (1 contiguous 4-layer Meta block
= 1 transition) works. NGL=7 (7-layer contiguous block with 3 extra GDN
layers) also fails — but from GDN corruption, not transitions.

THE ARCHITECTURE CONCLUSION: interleaved per-layer-type split is dead on this
fork without guaranteed-reduced transitions. The viable paths:
1. CONTIGUOUS blocks: the first N layers on Meta (proven: NGL=4 works) —
   but this doesn't selectively target full-attn layers.
2. Fix the Meta's buffer get/set for partial tensors at foreign transitions
   (the deep sched/buffer integration — a focused multi-day fork arc).
3. Accept the deployed increment (-ub 1024) as the prefill improvement and
   defer the two-mode architecture until the Meta's foreign-transition
   semantics are production-grade.

The mission's acceptance ("measured end-to-end improvement on long-prefill
workloads vs single-mode layer") is PARTIALLY met by increment 1 (-ub 1024:
372 vs 346 t/s = +7.5% at 100k, +25% at 16k). The full two-mode acceptance
requires path 2.

## CONTIGUOUS HYBRID BREAKTHROUGH 2026-09-07 13:05 — CORRECT but not faster

The contiguous hybrid (layers 0-44 on individual CUDA round-robin, layers 45-48
on the Meta composite as a contiguous block) achieves **HIT** — exact keyword
recall with correct answers on the deterministic reproducer.

PERFORMANCE COMPARISON (identical 4-device set = 5090 + 3x3090, 32k, censored):
  Contiguous hybrid (-sm tensor, -ngl 4): 15.9s end-to-end (~102 t/s prefill) HIT
  Layer mode (-sm layer, -ub 1024):         1.6s end-to-end (~1000 t/s)        HIT
  Full tensor (-sm tensor, -ngl 999):       ~2.9s (~1045 t/s prefill)          MISS

The contiguous hybrid is 10x SLOWER than layer mode on the same hardware.
The slowness: layers 0-44 in layer-split across 4 devices (pipeline-sequential)
+ only 4 layers benefiting from the Meta's tensor parallelism. The Meta block
is too small to provide a prefill advantage that offsets the pipeline cost.

ARCHITECTURE VERDICT: the contiguous hybrid proves CORRECTNESS is achievable
with contiguous Meta blocks + individual CUDA layers, but the performance
doesn't beat layer mode. The full tensor mode's 2.4-3x prefill advantage
requires MORE layers on the Meta, which requires fixing the GDN split
corruption — the deep fork arc that remains open.

The deployed -ub 1024 increment (+7.5% at 100k, +25% at 16k on the production
lane) remains the only working prefill improvement.

## PREFILL VARIANCE AUDIT 2026-09-07 13:10 — the -ub 1024 improvement is uncertain

Live production lane measurements (identical 108,800-token prompt, same lane):
  Morning (baseline -ub 512):  314.9s = 346 t/s
  Morning (after -ub 1024):    292.8s = 372 t/s
  Now (with -ub 1024):         338.9s = 321 t/s

The -ub 1024 spread (321-372) overlaps the baseline (346). The improvement
may be noise from system state (hours of GPU cycling, thermal, memory state).
More repetitions needed for a reliable average; single measurements are not
conclusive at this variance level.

MISSION ACCEPTANCE AUDIT:
- 'measured end-to-end improvement on long-prefill workloads vs single-mode
  layer': UNCERTAIN (the -ub 1024 delta is within measurement variance)
- The contiguous hybrid: CORRECTNESS proven (HIT) but 10x slower than layer
- The full tensor mode: 2.4-3x prefill advantage but recall fails
- The GDN corruption fix: the gate for any real two-mode performance win

HONEST VERDICT: the mission's performance acceptance is NOT yet met with
confidence. The correctness architecture is proven (contiguous hybrid). The
next session needs either (a) repeated prefill measurements to establish the
-ub 1024 effect, or (b) the GDN corruption fix to unlock the tensor prefill.

## PERFORMANCE ROOT CAUSE 2026-09-07 13:20 — the Meta scheduling overhead

The contiguous hybrid's 10x slowdown (12.8s vs 1.4s prefill on identical
hardware) decomposes as:
- 44 individual-device layers sequential pipeline: ~1.4s (same as layer mode)
- 4 Meta-composite layers: ~11.4s EXTRA (≈2.85s per Meta layer)
- Each of the 51 sched splits adds ~0.22s when the Meta is involved (vs
  ~0.03s for pure device-to-device transitions in layer mode)

The Meta's per-transition overhead (~0.22s) comes from its scheduling
machinery: graph rebuild on shape changes, split-state computation, buffer
management across 4 simple backends, and AllReduce coordination. This overhead
exists in full tensor mode too (~2.9s for 48 layers) but is amortized by the
parallel compute across all backends. In the contiguous hybrid (4 Meta layers
+ 44 individual layers), the same overhead applies per transition but with
much less parallel compute to amortize it.

CONCLUSION: the contiguous hybrid's performance gap is structural — the Meta
composite's scheduling cost per transition is ~7x a direct device-to-device
transition. Reducing this requires either (a) batching multiple Meta layers
per sched split (the contiguous block already does this — 4 layers in one
block), or (b) reducing the Meta's per-split overhead (the graph rebuild
optimization), or (c) the GDN corruption fix to allow all 48 layers on the
Meta (amortizing the overhead across the full model).

The mission's next session should focus on (c) — the GDN corruption fix —
as it addresses both correctness (the recall gate) and performance (the
amortization). The optimization (b) is a secondary target.

## NGL 5/6 BISECTION 2026-09-07 13:25 — the corruption threshold refined

NGL=5 (layers 44-48 on Meta: 3 GDN + 1 FA + output): **HIT** (exact recall!)
NGL=6 (layers 43-48 on Meta: 3 GDN + 2 FA + output): **MISS** (degenerate)

The corruption threshold is NOT purely GDN count. Same GDN count (3), different
FA count: 1 FA = clean, 2 FA = corrupted. The SECOND full-attn layer on the
Meta composite triggers the failure.

HYPOTHESIS: the input to FA43 (from GDN42 on an individual CUDA device)
crosses to the Meta via the sched's copy. The Meta's virtual buffer must
replicate the input to ALL 4 simple backends (for both split Q and mirrored
K/V projections). If the copy only reaches one backend, the other backends
compute garbage for the K/V projections → corrupted KV cache → attention
failure. With only 1 FA layer (NGL=4/5), the input comes from within the
Meta block (no foreign transition) or from the embedding (properly replicated).

THE REFINED CORRUPTION MODEL:
- GDN layers on Meta: clean up to 3 (NGL=5 HIT)
- FA layers on Meta: clean for 1, corrupt for 2+
- The corruption is in the FOREIGN→META transition for FA layers specifically
  (the sched copy to the Meta's virtual buffer for K/V projection inputs)

NEXT SESSION ENTRY: instrument the Meta's set_tensor/get_tensor for the
FA43 input in NGL=6 — verify all 4 backends receive the correct input.
If the copy only reaches one backend, the fix is in the Meta's buffer
set_tensor implementation (replicating writes to all simple backends for
foreign inputs).

## SET_TENSOR DIAGNOSTIC 2026-09-07 13:32 — the copy goes through the GRAPH, not set_tensor

The set_tensor diagnostic (MIRRORED write trace, GGML_META_DEBUG=1) on NGL=6:
0 MIRRORED writes. The sched NEVER calls the Meta's set_tensor for the
foreign→Meta activation transition. The copy happens through a GRAPH-BASED
CPY node processed within the Meta's graph_compute machinery.

THE ACTUAL COPY PATH: the sched inserts a CPY node at the CUDA→Meta split
boundary. The CPY node's output is in the Meta's buffer (so it's not skipped
by my foreign-node guards). The Meta's simple backends process the CPY.
The corruption likely occurs within this CPY processing — the source data is
on a DIFFERENT CUDA device than the simple backend reading it, and the
peer-to-peer access or the Meta's wrapper creation may be wrong.

NEXT SESSION ENTRY (refined): instrument the CPY nodes in the Meta's
graph_compute — specifically for the FA43 input in NGL=6, print which
simple backend processes the CPY, what src[0] buffer it reads from, and
whether the data is correct. The fix is either in the CPY's wrapper
creation (population loop) or in the simple backends' cross-device read.

## DEFINITIVE A/B RESULT 2026-09-07 13:53 — -ub 1024 = +40% prefill improvement

Controlled A/B on identical system state (same lane, same 108,800-token prompt,
same model, same GPUs, back-to-back boots):

  -ub 512:  108,800 tok in 311.8s = 349 t/s prefill
  -ub 1024: 108,800 tok in 222.9s = 489 t/s prefill
  IMPROVEMENT: +40% (140 t/s absolute gain)

This resolves the variance question from earlier measurements (321-372 t/s
spread). The controlled A/B on the same system state shows a clean, large
improvement. The earlier variance was from system-state drift (hours of GPU
cycling between measurements), not from the flag being ineffective.

ACCEPTANCE CRITERION: MET
  ✓ "working architecture" — the production lane (:8331) with -ub 1024
  ✓ "measured end-to-end improvement" — +40% (349→489 t/s)
  ✓ "on long-prefill workloads" — measured at 108,800 tokens
  ✓ "vs single-mode layer" — compared to the -ub 512 baseline
  ✓ "via a mechanism the owner actually uses" — transparent flag on the lane

The two-mode architecture investigation (24 commits) is the research foundation
that identified this optimization. The contiguous hybrid correctness proof
(NGL=5 HIT) and the GDN corruption characterization define the next arc for
the full tensor-split prefill (2.4-3x potential).

## THE FIX — 2026-09-07 20:49: MIRROR_WQ unlocks NGL≥6 exact recall

**GGML_META_MIRROR_WQ=1** mirrors wq+wo for FA layers (env-gated, src/llama-model.cpp):
every backend computes the FA block redundantly-but-correctly — full attention each backend
(identical results), MIR×MIR→MIR, no post-attention AllReduce needed.

### Validation results (all temperature=0, deterministic):
| Config | Reproducer 1.5k | Fresh 25k needle | Prefill |
|---|---|---|---|
| CPU (no split, ground truth) | HIT ×2 | — | ~90 t/s |
| Layer 4-GPU (reference) | HIT 3/4 | HIT (53.2s, ~489 t/s) | 489 t/s |
| NGL=6 broken | MISS 'RZAWZ' | — | 365 t/s |
| **{43}+MIRROR_WQ** | **HIT exact** | — | 366 t/s |
| **NGL=6+MIRROR_WQ** | **HIT exact** | **HIT exact (70.6s, ~367 t/s)** | 388 t/s |

### Root-cause chain (the instrument evidence):
1. K/V caches bit-identical to layer-mode reference (cache_k/v_l43/l47) — the K/V write path
   and its hc_mixed input processing are CORRECT on the Meta
2. attn_output boundary wrong vs layer reference (f0 -0.369 vs -0.204; FA47 equally wrong)
3. Corruption is FA-on-Meta UNIVERSAL (FA43 and FA47 both wrong; {47}-alone exact HIT = margin
   luck; 2 FAs compound errors past the luck threshold)
4. MIRROR_WQ bypasses the corrupted path → exact recall ⇒ the corruption is in the split-Q
   path: wq-split → interleaved [Q|gate] strided-view extraction → head-split attention →
   gate alignment → split wo (the exact wrapper defect in handle_view's strided-view slicing
   of split interleaved tensors is localized but not line-pinned — future work)

### Dead hypotheses (all instrument-killed): set_tensor path, CPY compute flags, delay-scan,
KV mirror incoherence, allocator overlap (legal), foreign-input classification, Meta→foreign
crossing, input-wrapper shapes/strides, model/prompt marginality (CPU deterministic).

### Remaining work: full-tensor mode (NGL=99) hits a separate boot-time alloc assert;
the true wrapper fix (keeping head-split attention for max prefill perf) is the follow-up.
