#!/usr/bin/env bash
# [inverse-hybrid] Full-attn layers (≡3 mod 4) on the Meta composite (4-way tensor split);
# GDN layers round-robined to individual CUDA devices (weights + recurrent caches).
# Tests whether the clean-split component (full-attn) + unsplit GDN = working recall.
set -u
D4=$(python3 -c "
import json
d = json.load(open('/home/benbi/models/flashnext_uncensored_gguf/raw_source/balanced_eval_spec.json'))['models']['Q8REF']['devices']
print(','.join(d[:4]))
")
# GDN layers by residue mod 4: 0→CUDA0, 1→CUDA1, 2→CUDA2 (residue 3 = full-attn, stays Meta)
R0="0|4|8|12|16|20|24|28|32|36|40|44"
R1="1|5|9|13|17|21|25|29|33|37|41|45"
R2="2|6|10|14|18|22|26|30|34|38|42|46"
OT="blk\.($R0)\..*=CUDA0,blk\.($R1)\..*=CUDA1,blk\.($R2)\..*=CUDA2"
# also route the GDN recurrent/conv caches with their layers
OT="$OT,cache_[rs]_l($R0)=CUDA0,cache_[rs]_l($R1)=CUDA1,cache_[rs]_l($R2)=CUDA2"
echo "OT=$OT" | head -c 300; echo

CUDA_VISIBLE_DEVICES="$D4" timeout 700 /home/benbi/llama.cpp-qwen38-tsplit-dev/build-tsplit-dbg/bin/llama-server \
  -m /home/benbi/models/flashnext/qwen38_flash_next_iq4nl_pleq8.gguf \
  -sm tensor -ts "1.6,1,1,1" -ngl 999 -fa on -c 32768 -ctk f32 -ctv f32 -np 1 --no-kv-unified \
  -ot "$OT" \
  --reasoning-format deepseek --reasoning-preserve --host 127.0.0.1 --port 8336 > /tmp/inverse-hybrid.log 2>&1 &
SRV=$!
UP=0
for i in $(seq 1 60); do H=$(curl -s --max-time 3 http://127.0.0.1:8336/health 2>/dev/null | head -c 20); echo "$H" | grep -q '"ok"' && { UP=1; echo "HYBRID UP $(date +%H:%M:%S)"; break; }; kill -0 $SRV 2>/dev/null || { echo DIED; break; }; sleep 9; done
echo "asserts: $(grep -cE 'GGML_ASSERT' /tmp/inverse-hybrid.log)"
if [ "$UP" = "1" ]; then
  python3 - <<'EOF'
import json, urllib.request, time
prompt = open('/tmp/repro-1p5k.txt').read()
payload = {"messages":[{"role":"user","content":prompt}],"max_tokens":40,"temperature":0,
           "chat_template_kwargs":{"enable_thinking":False}}
req = urllib.request.Request("http://127.0.0.1:8336/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=400) as r: d=json.loads(r.read())
    dt=time.perf_counter()-t0
    pt=d["usage"]["prompt_tokens"]
    ans = d["choices"][0]["message"]["content"]
    print(f"INVERSE HYBRID: {pt} tok, {dt:.1f}s (prefill ~{pt/max(dt-d['usage']['completion_tokens']/12.8-0.1,0.001):.0f} t/s) | {'HIT' if 'MAPLE-SYRUP-7461' in ans else 'MISS'} | {ans[:50]!r}")
except Exception as e:
    print(f"request failed: {type(e).__name__}")
EOF
fi
kill $SRV 2>/dev/null; sleep 12
systemctl --user start qwen38-flash-next.service && echo "RESTORE $(date +%H:%M:%S)"
