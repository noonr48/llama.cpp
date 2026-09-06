#!/usr/bin/env python3
"""[tsplit-dev] Arc-2b bisect probe: tiny-model recall test for the Meta-split corruption.

Boots a small model 2-way tensor-split, plants a code in a short prompt, checks
recall. Exit 0 = recall PASS (Meta core computes correctly at this commit);
exit 1 = recall FAIL (corruption present); exit 2 = server failed to boot.

Usage: python3 meta_recall_probe.py [--model PATH] [--port N] [--bin PATH]
Designed for git-bisect: point BISECT_BIN at a build dir containing llama-server.
"""
import argparse, json, random, string, subprocess, sys, time, urllib.request, os, signal

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/benbi/models/gemma4-12b-abliterated/gemma4-12b-abliterated-Q8_0.gguf")
    p.add_argument("--sm", default="tensor", choices=["tensor", "layer"])
    p.add_argument("--bin", default="/home/benbi/llama.cpp-qwen38-tsplit-dev/build-tsplit/bin/llama-server")
    p.add_argument("--port", type=int, default=8341)
    p.add_argument("--devs", default="GPU-79ec2e41-4c0a-5b7e-da4f-59da82c9648e,GPU-1b94cb96-ee44-7ee5-3fcb-cd92274bcb27")
    p.add_argument("--timeout", type=int, default=240)
    a = p.parse_args()

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.devs)
    log = open("/tmp/meta-probe-server.log", "w")
    srv = subprocess.Popen([a.bin, "-m", a.model, "-sm", a.sm, "-ts", "1.6,1",
        "-ngl", "999", "-fa", "on", "-c", "8192", "-ctk", "f32", "-ctv", "f32",
        "-np", "1", "--no-kv-unified", "--host", "127.0.0.1", "--port", str(a.port)],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        up = False
        deadline = time.monotonic() + a.timeout
        while time.monotonic() < deadline:
            if srv.poll() is not None:
                print(f"PROBE: server died rc={srv.returncode}", flush=True)
                return 2
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=3) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        up = True; break
            except Exception:
                pass
            time.sleep(4)
        if not up:
            print("PROBE: boot timeout", flush=True); return 2

        random.seed(1234)
        filler = " ".join("".join(random.choices(string.ascii_lowercase + " ", k=7)) for _ in range(600))
        code = "ZEBRA-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        prompt = (f"Study notes, part 7.\n\nThe access code for tonight is {code}.\n\n{filler}\n\n"
                  f"End of notes.\n\nQuestion: What is the access code for tonight? The access code is")
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions",
            data=json.dumps({"prompt": prompt, "max_tokens": 40, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        ans = d["choices"][0]["text"].strip()[:150]
        ok = code in ans
        print(f"PROBE[{a.sm}]: planted={code} answer={ans!r} -> {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    finally:
        try: os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
        except Exception: srv.terminate()
        try: srv.wait(timeout=30)
        except Exception: srv.kill()

if __name__ == "__main__":
    sys.exit(main())
