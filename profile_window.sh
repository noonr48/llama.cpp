#!/usr/bin/env bash
# [tsplit-dev] Option C profile: phase-timed rebuilds in tensor mode + layer comparison.
set -u
DEVS=$(python3 -c "import json; print(','.join(json.load(open('/home/benbi/models/flashnext_uncensored_gguf/raw_source/balanced_eval_spec.json'))['models']['Q8REF']['devices']))")
M=/home/benbi/models/flashnext_uncensored_gguf/raw_build/qwen38_flash_next_uncensored_raw_iq4nl.gguf
B=/home/benbi/llama.cpp-qwen38-tsplit-dev/build-tsplit/bin/llama-bench
export CUDA_VISIBLE_DEVICES="$DEVS"
echo "=== TENSOR MODE (profiled) $(date +%H:%M:%S) ==="
"$B" -m "$M" -sm tensor -ts 2,1,1,1,1,1,1,1,1 -ngl 999 -fa 1 -p 512,2048,8192 -n 64 -r 1
echo "TENSOR_RC=$?"
echo "=== LAYER MODE (comparison) $(date +%H:%M:%S) ==="
"$B" -m "$M" -sm layer -ts 2,1,1,1,1,1,1,1,1 -ngl 999 -fa 1 -p 512,2048,8192 -n 64 -r 1
echo "LAYER_RC=$?"
echo "=== DONE $(date +%H:%M:%S) ==="
