#!/usr/bin/env bash
# [tsplit-dev] TRUE-FA validation: plain -ngl 6 contiguous hybrid (layers 43-48 on the Meta
# composite), NO MIRROR_WQ, fix ccf27ca20 (planner head-block alignment) in build-tsplit.
# Historical baseline without fix: 1.5k reproducer MISS 'RZAWZ'. Expect exact HITs now.
set -u
LANE=qwen38-flash-next-prefill.service
BIN=/home/benbi/llama.cpp-qwen38-tsplit-dev/build-tsplit/bin/llama-server
MODEL=/home/benbi/models/flashnext/qwen38_flash_next_iq4nl_pleq8.gguf
D4=$(python3 -c "import json; d=json.load(open('/home/benbi/models/flashnext_uncensored_gguf/raw_source/balanced_eval_spec.json'))['models']['Q8REF']['devices']; print(','.join(d[:4]))")

restore_lane() {
  pkill -f "port 8336" 2>/dev/null; sleep 12
  systemctl --user start $LANE
  for i in $(seq 1 40); do curl -s --max-time 3 http://127.0.0.1:8331/health 2>/dev/null | grep -q '"ok"' && { echo "LANE RESTORED $(date +%H:%M:%S)"; return 0; }; sleep 15; done
  echo "LANE RESTORE PENDING (restart-loop quirk; self-resolves ~5-10 min)"
}
trap restore_lane EXIT

echo "== stopping lane, settling VRAM =="
systemctl --user stop $LANE
sleep 150
unset GGML_META_MIRROR_WQ

echo "== booting -ngl 6 contiguous hybrid (NO MIRROR_WQ), fix ccf27ca20 =="
GGML_META_GQA_FIX=1 GGML_META_GQA_PROBE=1 CUDA_VISIBLE_DEVICES="$D4" $BIN -m $MODEL -sm tensor -ts "1.6,1,1,1" -ngl 6 -fa on -c 32768 \
  -ctk f32 -ctv f32 -np 1 --no-kv-unified \
  --reasoning-format deepseek --reasoning-preserve --host 127.0.0.1 --port 8336 > /tmp/true-fa-test.log 2>&1 &
SRV=$!
UP=0
for i in $(seq 1 70); do curl -s --max-time 3 http://127.0.0.1:8336/health 2>/dev/null | grep -q '"ok"' && { UP=1; echo "SERVER UP $(date +%H:%M:%S)"; break; }; kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -8 /tmp/true-fa-test.log; break; }; sleep 9; done
[ "$UP" = "1" ] || { echo "BOOT FAILED"; exit 1; }
echo "asserts: $(grep -cE 'GGML_ASSERT|GGML_ABORT' /tmp/true-fa-test.log)"

echo "== probe 1: deterministic 1.5k reproducer (expect MAPLE-SYRUP-7461) =="
python3 - <<'EOF'
import json, urllib.request, time
prompt = open('/tmp/repro-1p5k.txt').read()
payload = {"messages":[{"role":"user","content":prompt}],"max_tokens":40,"temperature":0,
           "chat_template_kwargs":{"enable_thinking":False}}
req = urllib.request.Request("http://127.0.0.1:8336/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as r: d=json.loads(r.read())
dt=time.perf_counter()-t0
ans = d["choices"][0]["message"]["content"]
print(f"REPRO: {d['usage']['prompt_tokens']} tok, {dt:.1f}s | {'HIT' if 'MAPLE-SYRUP-7461' in ans else 'MISS'} | {ans[:60]!r}")
EOF

echo "== probe 2: fresh ~25k needle (random code) =="
python3 - <<'EOF'
import json, urllib.request, time, random
random.seed(20260909)
words = ["".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(3,9))) for _ in range(7500)]
code = "HAWK-AMETHYST-8842"
mid = len(words)//2 + 700
body = " ".join(words[:mid]) + f" verification code {code} " + " ".join(words[mid:])
prompt = f"Study notes.\n\n{body}\n\nWhat was the verification code mentioned in the notes? Reply with only the code."
payload = {"messages":[{"role":"user","content":prompt}],"max_tokens":40,"temperature":0,
           "chat_template_kwargs":{"enable_thinking":False}}
req = urllib.request.Request("http://127.0.0.1:8336/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.perf_counter()
with urllib.request.urlopen(req, timeout=900) as r: d=json.loads(r.read())
dt=time.perf_counter()-t0
pt = d["usage"]["prompt_tokens"]
ans = d["choices"][0]["message"]["content"]
print(f"NEEDLE: {pt} tok, {dt:.1f}s | {'HIT' if code in ans else 'MISS'} | {ans[:60]!r}")
EOF

echo "== done; restoring lane =="
