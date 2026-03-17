#!/usr/bin/env python3
"""End-to-end benchmark for Project GhostWeight."""
import time, ctypes, os, sys, warnings, json, platform
warnings.filterwarnings("ignore")
import torch, numpy as np
sys.path.insert(0, ".")

DIV = "=" * 60
def hdr(t): print(f"\n{DIV}\n  {t}\n{DIV}")
def row(label, val, unit="", note=""):
    n = f"  # {note}" if note else ""
    print(f"  {label:<42} {str(val):>10} {unit}{n}")

# ── 1. SYSTEM ────────────────────────────────────────────────
def bench_system():
    hdr("1. SYSTEM")
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                row("CPU", line.split(":")[1].strip()); break
    except: pass
    try:
        for line in open("/proc/meminfo"):
            if "MemTotal"     in line: row("RAM total",     f"{int(line.split()[1])/1e6:.0f}", "GB")
            if "MemAvailable" in line: row("RAM available", f"{int(line.split()[1])/1e6:.0f}", "GB")
    except: pass
    row("Python",  platform.python_version())
    row("PyTorch", torch.__version__)
    row("CPU cap", torch.backends.cpu.get_cpu_capability())

# ── 2. KERNEL BENCHMARKS ─────────────────────────────────────
def bench_kernels():
    hdr("2. KERNEL BENCHMARKS (M=1, isolated layer)")

    tlib = ctypes.CDLL("softchip/ternary_matmul_v3.so")
    glib = ctypes.CDLL("softchip/ghost_matmul.so")

    tlib.pack_weights.restype  = ctypes.c_void_p
    tlib.pack_weights.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_float)]
    tlib.ternary_matmul.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    tlib.ternary_matmul_backward.argtypes = tlib.ternary_matmul.argtypes
    glib.ghost_matmul_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_float]
    glib.ghost_matmul_backward.argtypes = glib.ghost_matmul_forward.argtypes

    ITERS, WARMUP = 20, 3
    shapes = [("q/o_proj",1,2560,2560), ("k/v_proj",1,640,2560),
              ("gate/up", 1,2560,6912), ("down_proj",1,6912,2560)]

    print(f"\n  {'Layer':<12} {'PyTorch':>10} {'Ternary':>10} {'Ghost':>10} {'Spdup(T)':>10} {'Spdup(G)':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    total_pt = total_t = total_g = 0
    for name, M, K, N in shapes:
        w = torch.randn(N, K, dtype=torch.bfloat16)
        x = torch.randn(M, K)

        # PyTorch
        for _ in range(WARMUP): _ = x.to(torch.bfloat16) @ w.T
        t0 = time.perf_counter()
        for _ in range(ITERS): _ = x.to(torch.bfloat16) @ w.T
        pt = (time.perf_counter()-t0)/ITERS*1000

        # Ternary
        wf = w.float(); scale = ctypes.c_float(0.0)
        wp = wf.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        packed = tlib.pack_weights(wp, N, K, ctypes.byref(scale))
        pp = ctypes.cast(packed, ctypes.c_void_p)
        xn = x.numpy().copy(); on = np.zeros((M,N),dtype=np.float32)
        xp = xn.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        op = on.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for _ in range(WARMUP): tlib.ternary_matmul(pp,xp,op,M,N,K,scale)
        t0 = time.perf_counter()
        for _ in range(ITERS): tlib.ternary_matmul(pp,xp,op,M,N,K,scale)
        tt = (time.perf_counter()-t0)/ITERS*1000

        # GhostWeight
        sv = scale.value
        for _ in range(WARMUP): glib.ghost_matmul_forward(xp,op,M,K,N,42,0,ctypes.c_float(sv))
        t0 = time.perf_counter()
        for _ in range(ITERS): glib.ghost_matmul_forward(xp,op,M,K,N,42,0,ctypes.c_float(sv))
        gt = (time.perf_counter()-t0)/ITERS*1000

        total_pt += pt; total_t += tt; total_g += gt
        print(f"  {name:<12} {pt:>9.1f}ms {tt:>9.1f}ms {gt:>9.1f}ms {pt/tt:>9.1f}x {pt/gt:>9.1f}x")

    print(f"  {'TOTAL (210 layers)':>12} {total_pt*210/1000:>9.1f}s  {total_t*210/1000:>9.1f}s  "
          f"{total_g*210/1000:>9.1f}s  {total_pt/total_t:>9.1f}x {total_pt/total_g:>9.1f}x",
          "(30 decoder blocks × 7 layers)")

    # Backward
    print(f"\n  Backward pass (q/o_proj, M=1):")
    M,K,N = 1,2560,2560
    w = torch.randn(N,K,dtype=torch.bfloat16); wf = w.float()
    wp = wf.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    scale = ctypes.c_float(0.0)
    packed = tlib.pack_weights(wp,N,K,ctypes.byref(scale))
    pp = ctypes.cast(packed, ctypes.c_void_p)
    go = np.ones((M,N),dtype=np.float32); gi = np.zeros((M,K),dtype=np.float32)
    gop = go.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gip = gi.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    for _ in range(WARMUP): tlib.ternary_matmul_backward(pp,gop,gip,M,N,K,scale)
    t0 = time.perf_counter()
    for _ in range(ITERS): tlib.ternary_matmul_backward(pp,gop,gip,M,N,K,scale)
    tb = (time.perf_counter()-t0)/ITERS*1000
    gogt = torch.ones(M,N,dtype=torch.bfloat16)
    for _ in range(WARMUP): _ = gogt @ w
    t0 = time.perf_counter()
    for _ in range(ITERS): _ = gogt @ w
    ptb = (time.perf_counter()-t0)/ITERS*1000
    sv = scale.value
    for _ in range(WARMUP): glib.ghost_matmul_backward(gop,gip,M,K,N,42,0,ctypes.c_float(sv))
    t0 = time.perf_counter()
    for _ in range(ITERS): glib.ghost_matmul_backward(gop,gip,M,K,N,42,0,ctypes.c_float(sv))
    gb = (time.perf_counter()-t0)/ITERS*1000
    print(f"  {'q/o_proj bwd':<12} {ptb:>9.1f}ms {tb:>9.1f}ms {gb:>9.1f}ms {ptb/tb:>9.1f}x {ptb/gb:>9.1f}x")

# ── 3. STORAGE AUDIT ─────────────────────────────────────────
def bench_storage():
    hdr("3. STORAGE AUDIT")
    model_dir = "models/bitnet-b1.58-2B-4T-bf16"
    model_bytes = sum(
        os.path.getsize(os.path.join(dp,f))
        for dp,dn,fns in os.walk(model_dir)
        for f in fns if f.endswith((".safetensors",".bin"))
    )
    row("Full model (BF16 weights on disk)", f"{model_bytes/1e9:.2f}", "GB")

    ckpts = sorted(f for f in os.listdir("checkpoints") if f.endswith(".pt"))
    if ckpts:
        cp = os.path.join("checkpoints", ckpts[-1])
        ck = torch.load(cp, weights_only=False)
        sc_bytes = sum(v.element_size()*v.numel() for v in ck.get("adapter_scales",{}).values())
        uv_bytes = sum(v.element_size()*v.numel()
                       for d in ck.get("adapter_state",{}).values() for v in d.values())
        row(f"Checkpoint ({ckpts[-1]})", f"{os.path.getsize(cp)/1e6:.1f}", "MB", "full file")
        row("  └─ adapter scales (learned)",     f"{sc_bytes}", "bytes", "← the '420 bytes'")
        row("  └─ u,v buffers (re-derivable)",   f"{uv_bytes/1e6:.2f}", "MB", "redundant, sha256(name) seeded")
    row("PRNG seed storage",         "8", "bytes", "uint64")
    total = 8 + (sc_bytes if ckpts else 420)
    row("Total irreducible state",   f"{total}", "bytes", f"= {total/1024:.2f} KB learned")
    print()
    print("  NOTE: Runtime requires ~6GB RAM (4.5GB BF16 weights for scale capture")
    print("        + 1.3GB FP32 LM head). '1KB' refers to learned parameters only.")

# ── 4. GRPO TIMING FROM LOG ───────────────────────────────────
def bench_grpo():
    hdr("4. GRPO TRAINING — TIMING FROM LOG")
    entries = []
    for logf in ["grpo_ghost_unified.log","grpo_fast.log","grpo_extended.log"]:
        if os.path.exists(logf):
            try:
                with open(logf.replace(".log",".jsonl").replace("grpo_","grpo_log")) as f:
                    pass
            except: pass
    try:
        with open("grpo_log.jsonl") as f:
            for line in f:
                l = line.strip()
                if l:
                    try: entries.append(json.loads(l))
                    except: pass
    except: pass

    if not entries:
        print("  No grpo_log.jsonl entries yet"); return

    non_skip = [e for e in entries if not e.get("skipped",False)]
    skip     = [e for e in entries if e.get("skipped",False)]

    print()
    row("Total steps logged",          len(entries))
    row("Non-skip (gradient updates)", len(non_skip))
    row("Skipped (zero variance)",     len(skip),  "", f"{100*len(skip)/max(1,len(entries)):.0f}%")

    if non_skip:
        rts = [e["rollout_time"] for e in non_skip if "rollout_time" in e]
        uts = [e.get("update_time",0) for e in non_skip if "update_time" in e and e.get("update_time",0)>0]
        tts = [e["total_time"] for e in non_skip if "total_time" in e]
        print()
        if rts:
            row("Rollout time mean",   f"{sum(rts)/len(rts):.0f}", "s")
            row("Rollout time min/max",f"{min(rts):.0f}/{max(rts):.0f}", "s")
        if uts:
            row("Update time mean",    f"{sum(uts)/len(uts):.0f}", "s")
        if tts:
            sph = 3600/(sum(tts)/len(tts))
            row("Full step mean",      f"{sum(tts)/len(tts):.0f}", "s")
            row("Steps/hour",          f"{sph:.1f}")
            row("Steps to 700",        f"{700/sph:.1f}", "hours")
        recent = non_skip[-20:]
        rews = [r for e in recent for r in e.get("rewards",[])]
        if rews: row("Mean reward (last 20 steps)", f"{sum(rews)/len(rews):.3f}")
        sc = [e["mean_abs_scale"] for e in recent if "mean_abs_scale" in e]
        if sc: row("Mean |scale| (last 20 steps)",  f"{sum(sc)/len(sc):.6f}")

# ── 5. FULL MODEL FORWARD (optional, slow) ───────────────────
def bench_model():
    hdr("5. FULL MODEL FORWARD (real BitNet vs GhostWeight)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from softchip import patch_model, patch_lm_head_fp32
    MP = "models/bitnet-b1.58-2B-4T-bf16"
    prompt = "What is 15 plus 27?"

    results = {}
    for label, ghost in [("Real BitNet", False), ("GhostWeight", True)]:
        print(f"\n  [{label}] loading...")
        tok = AutoTokenizer.from_pretrained(MP)
        m = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.bfloat16,
                device_map="cpu", trust_remote_code=True)
        if ghost: patch_model(m, use_ghost=True, ghost_seed=42)
        else:     patch_model(m, backend="cpu")
        patch_lm_head_fp32(m)
        ids = tok(prompt, return_tensors="pt").input_ids

        with torch.no_grad():
            _ = m(ids)  # warmup
            t0 = time.perf_counter()
            out = m(ids)
            fwd_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            gen = m.generate(ids, max_new_tokens=5, do_sample=False)
            gen_s = time.perf_counter() - t0

        answer = tok.decode(gen[0,ids.shape[1]:], skip_special_tokens=True)
        row(f"  [{label}] forward ({ids.shape[1]} tok)", f"{fwd_s*1000:.0f}", "ms")
        row(f"  [{label}] generate (5 tok)",             f"{gen_s*1000:.0f}", "ms",
            f"{5/(gen_s):.2f} tok/s")
        row(f"  [{label}] output",                       answer.strip()[:40])
        results[label] = {"fwd_s": fwd_s, "gen_s": gen_s, "answer": answer}
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if "Real BitNet" in results and "GhostWeight" in results:
        print()
        row("Forward slowdown Ghost/Real",
            f"{results['GhostWeight']['fwd_s']/results['Real BitNet']['fwd_s']:.2f}","x")
        row("Generation slowdown",
            f"{results['GhostWeight']['gen_s']/results['Real BitNet']['gen_s']:.2f}","x")

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="Skip full model benchmark (fast)")
    args = ap.parse_args()

    print(f"\n{'#'*60}")
    print(f"  PROJECT GHOSTWEIGHT — END-TO-END BENCHMARK")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    bench_system()
    bench_kernels()
    bench_storage()
    bench_grpo()
    if not args.no_model:
        bench_model()

    print(f"\n{'#'*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'#'*60}\n")
