---
name: vllm-xpu-esimd-optimize
description: End-to-end loop for optimizing a vLLM model on Intel XPU (BMG / Arc B-series) by writing/接入 ESIMD kernels in llm-scaler, while keeping accuracy verifiable. Use whenever the user asks to "optimize / speed up / fuse / write esimd kernel for <some vllm model> on XPU" and supplies (1) a container name with editable vllm-xpu installed and (2) a path to a llm-scaler custom-esimd-kernels checkout that can be rebuilt in-place. Covers offline reproducer setup, accuracy gating, decode-ITL profiling, kernel-level profiling, and the kernel-write → bind → wire-up → verify cycle.
---

# vLLM XPU ESIMD optimization loop

This skill is a recipe — **always invoke it at the start of any session whose goal is to make a specific vLLM model faster on XPU by writing or接入 ESIMD kernels.** It encodes hard-won lessons from the gemma-4-26B / 31B work; do not reinvent these steps.

## Required inputs (ask if missing)

Before doing anything, confirm with the user:

1. **`CONTAINER`** — docker container name (host network), e.g. `wj-test-new-021`. vllm-xpu must already be installed editable inside this container. Verify with `docker exec $CONTAINER bash -c 'cd /workspace/llm-scaler-vllm-xpu && git log --oneline -1'` (path may differ — ask).
2. **`VLLM_PATH`** — editable vllm-xpu path inside the container, e.g. `/workspace/llm-scaler-vllm-xpu`. Changes to `vllm/...` files are picked up immediately.
3. **`LLMSCALER_PATH`** — llm-scaler custom-esimd-kernels checkout, e.g. `/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm`. New kernels must be built here and the resulting `.so` files copied into the python package directory.
4. **`MODEL_PATH`** — weights, e.g. `/llm/models/weights/gemma-4-26B-A4B-it`.
5. **`MODEL_FAMILY`** — model class name in vllm (so we know which `vllm/model_executor/models/<name>.py` to wire kernels into), e.g. `gemma4`, `qwen3_next`.
6. **`TP`** — tensor_parallel_size (typically 2). And `ZE_AFFINITY_MASK` for which two devices, e.g. `6,7`.
7. **`BASELINE_CONTAINER`** *(optional but strongly recommended)* — a sibling container with stock upstream vllm-xpu (no optimizations) on the same model. Used as ground truth for accuracy regression. Same model weights, same TP, same `MODEL_PATH`.

If any of these are not supplied, ask 2–3 short clarifying questions before proceeding.

## Standing assumptions

- `host` network for both containers ⇒ they cannot run a server on the same port simultaneously. Always use **offline** flows; only spin up an HTTP server when the user explicitly asks.
- `enforce_eager=True` everywhere (XPU graph compilation is incomplete; this also matches all numbers we ever quote).
- `dtype="float16"` + `quantization="fp8"` is the default benchmarking config.
- `kv_cache_dtype="fp8"` for ITL benchmarking (matches release config); leave default for accuracy gating.
- All env vars listed below are gates the existing optimization stack already understands. Use them rather than ripping code out:
  - `DISABLE_ESIMD_GEMV`, `DISABLE_ESIMD_NORM`, `DISABLE_ESIMD_QKV_FUSED`, `DISABLE_ESIMD_PAGE_ATTN`, `DISABLE_ESIMD_ROUTER_GEMV`, `DISABLE_ESIMD_MOE_GELU`
  - `DISABLE_GEMMA4_FUSED_ROUTER`, `DISABLE_GEMMA4_FUSED_H2`, `DISABLE_GEMMA4_FUSED_ATTN_OUT`, `DISABLE_GEMMA4_XFUSE`, `DISABLE_GEMMA4_GELU_ESIMD`, `DISABLE_BMG_GEMV`
  - `DISABLE_ESIMD_MOE_PREFILL`, `MAX_ESIMD_MOE_TOKENS`
- Standard launch envs: `ZE_AFFINITY_MASK=$ZE_DEVS TORCH_LLM_ALLREDUCE=1 CCL_ZE_IPC_EXCHANGE=pidfd VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MLA_DISABLE=1 VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0`.

## The loop

The phases below are mostly sequential, but you can re-enter Profile → Write Kernel → Verify many times. Always close every kernel change with the **Verify** phase before claiming it works.

### Phase 1 — Smoke test: does the model generate sensible output offline?

Goal: produce a 16-token prompt → 64 token decode reproducer that prints a token id list. Two-purpose: (a) lets you eyeball that the model isn't NaN-ing, (b) the printed `Token ids:` is a stable A/B beacon for later kernel changes (token-by-token equality with baseline ⇒ optimization is functionally equivalent).

Skeleton script (write to `/tmp/run_${MODEL_FAMILY}_chat.py` inside `$CONTAINER`):

```python
import torch
from vllm import LLM, SamplingParams

# 16-token prompt that exercises the chat template / instruction path.
# For gemma4: PROMPT_IDS = [2, 105, 2364, 107, 12553, 54847, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]
# For other models, get this once via apply_chat_template on a real chat message.
PROMPT_IDS = [...]

def main():
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP,
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        dtype='float16',
        quantization='fp8',
        trust_remote_code=True,
        max_num_seqs=8,
    )
    out = llm.generate(prompts=[{'prompt_token_ids': PROMPT_IDS}],
                       sampling_params=SamplingParams(max_tokens=64, temperature=0))
    print('=' * 60)
    print(f'Output: {repr(out[0].outputs[0].text)}')
    print(f'Token ids: {list(out[0].outputs[0].token_ids)}')
    print('=' * 60)

if __name__ == '__main__':
    main()
```

Run with the standard env block. **Save the token-id list** as the per-model functional fingerprint — every later kernel change must reproduce it (or document the deviation).

If output text is empty / `Token ids: [0, 0, 0, ...]` / `finish_reason=length` with `chars=0`: this is the **NaN-logits-→-pad-token failure mode**. Bisect (Phase 5) to find the offending kernel before doing any new perf work.

### Phase 2 — Accuracy gating with offline gsm8k chat replay

Single 16-token reproducer is not enough — long prompts, batched prefill, and chat-template-applied paths exercise different kernel shapes. Use this for actual accuracy regression checking.

1. **Extract the requests once** (do this in `$BASELINE_CONTAINER` or any container that has the gsm8k jsonl already cached at `/tmp/{train,test}.jsonl`):

```python
# /tmp/extract_chat_reqs.py
import json
test = [json.loads(l) for l in open("/tmp/test.jsonl")]
train = [json.loads(l) for l in open("/tmp/train.jsonl")]
NUM_SHOTS, NUM_Q = 5, 5
few = []
for i in range(NUM_SHOTS):
    few += [{"role": "user", "content": train[i]["question"]},
            {"role": "assistant", "content": train[i]["answer"]}]
reqs = []
for i in range(NUM_Q):
    msgs = [{"role": "system", "content":
             "Solve the math problem step by step. End your answer with a line in the form: #### <number>."},
            *few, {"role": "user", "content": test[i]["question"]}]
    reqs.append({"messages": msgs, "model": SERVED_MODEL_NAME,
                 "temperature": 0, "max_tokens": 512, "seed": 42,
                 "_label": test[i]["answer"].split("####")[-1].strip().replace(",", "")})
json.dump(reqs, open("/tmp/chat_reqs.json", "w"), indent=2)
```

`docker cp` the result into `$CONTAINER:/tmp/chat_reqs.json`.

2. **Offline replay**: apply chat template locally, hand prompt token ids to `llm.generate`. Critical — `apply_chat_template` returns a `BatchEncoding` when `tokenize=True`; use the two-step form to get a list[int]:

```python
prompt_str = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
ids = tok(prompt_str, add_special_tokens=False).input_ids
```

The full template for `/tmp/offline_chat_reqs.py` is in `assets/offline_chat_reqs.py` of this skill — adapt `MODEL_PATH` and TP only.

Expected baseline: **5/5 accuracy** on the 5 GSM8K questions for gemma-4-26B/31B-fp8/TP=2. **A failed acc on this script is a hard stop**.

A `pred=-9999999, finish=length, chars=0` triple means NaN logits sampled token id 0 every step — model is broken, not a sampling fluke.

### Phase 3 — Performance baseline: ITL bench

`/tmp/bench_itl.py` template:

```python
import os, time
def main():
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_PATH, tensor_parallel_size=TP, max_model_len=8192,
        enforce_eager=True, gpu_memory_utilization=0.90,
        dtype='float16', quantization='fp8', kv_cache_dtype='fp8',
        trust_remote_code=True, max_num_seqs=8,
    )
    PROMPT_IDS = [...]   # same 16-token prompt as Phase 1
    sp = SamplingParams(max_tokens=128, temperature=0)
    llm.generate(prompts=[{'prompt_token_ids': PROMPT_IDS}], sampling_params=sp)  # warmup
    tpots = []
    for i in range(5):
        t0 = time.perf_counter()
        out = llm.generate(prompts=[{'prompt_token_ids': PROMPT_IDS}], sampling_params=sp)
        dt = time.perf_counter() - t0
        n = len(out[0].outputs[0].token_ids)
        itl = dt * 1000 / n
        print(f"  run {i+1}: {n} tokens in {dt*1000:.1f} ms  → {n/dt:.2f} tok/s  avg_itl={itl:.2f} ms", flush=True)
        tpots.append(itl)
    median = sorted(tpots)[len(tpots)//2]
    print(f"median ITL = {median:.2f} ms ({1000/median:.2f} tok/s)")

if __name__ == "__main__":
    main()
```

Always quote **median of 5** (not best, not mean). Note 5-run spread — if it's > 0.3 ms the result is noisy and you need to re-run. The most stable comparisons are A/B with an env-gate flip (e.g. `DISABLE_FOO=1 vs unset`) on the same binary, in the same minute, on the same EUs.

### Phase 4 — Profile to find the bottleneck

Two complementary lenses:

**Lens A — wall-time of model layer subcomponents** (decide where to spend kernel time). Lightweight: monkeypatch the layer's `forward` with `torch.xpu.synchronize() ; perf_counter` brackets. Stats accumulate in module attrs and dump every N calls. Template lives in `assets/profile_attn.py` (gated by `PROFILE_ATTN=1`, zero overhead at 0).

For attention specifically, also bucket by `(gqa_ratio, is_sliding)` so you can see whether sliding vs full layers behave the same — that's how we discovered gemma4 sliding-window attn is already at the kernel limit.

**Lens B — kernel-level micro-bench** (decide how fast a *replacement* kernel needs to be). Standalone `/tmp/bench_gemv.py` skeleton:

```python
import torch, time
from custom_esimd_kernels_vllm import esimd_<your_op>
device = "xpu:0"; torch.xpu.set_device(0)

def bench(op, *args, n_iter=200):
    for _ in range(20): op(*args)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): op(*args)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter

# loop over (K, N) tuples for every shape the model exercises
```

Always report the **HBM bandwidth utilization** alongside time: `bw_GBps = (bytes_in + bytes_w + bytes_out) / dt_us / 1e3`. BMG ~456 GB/s peak; if a kernel is already > 700 GB/s it's L3-cache-fed and you're at a hard bound.

**Lens C — unitrace whole-run kernel timeline (the bottleneck killer; START HERE when you don't yet know where time goes).** Lens A/B require you to already guess *what* to bracket; unitrace needs no guess — one trace gives you, for the whole forward, (1) **device-busy%** = the launch-bound vs compute-bound verdict, (2) **kernel family self-time breakdown** = which op to even consider writing, and (3) the **gap profile** = whether the wall time is host-side dead air the GPU never sees. **Use it to size the prize before writing any kernel** — §4.2 of the unitrace doc killed a multi-day attention-kernel plan by showing attention was 0.2% of device time.

⚠️ **Profile the ONLINE server, and window to a single in-flight request — not an offline serial loop.** This is the single biggest correctness trap and it bit MiniCPM-V 4.6 hard: the *same* model traced two ways gave opposite verdicts. An **offline** reproducer that fires requests serially, aggregated over a wide tail window, showed **GPU busy ~13%** → "launch-bound, no kernel worth writing." But that 87% idle was just the dead time *between* my hand-fired requests, not the model waiting. Re-running on the **online server** (real overlap, real scheduler) and phase-slicing to the tail **4%** — one request's actual execution — showed **GPU busy 88.5% → genuinely compute-bound**. Rule: offline + wide window *under*-counts busy% and fabricates a launch-bound story; only the online server with the window tightened to a single request's execution reflects the deployed bottleneck. Sweep `tail_frac` (e.g. 0.5 → 0.1 → 0.04) and watch busy% climb as the inter-request idle is squeezed out — the tightest stable window is your answer.

Install once per container (it's prebuilt in the llm-scaler tools checkout — do NOT rebuild):
```bash
B=/llm/models/test/tools/pti-gpu/tools/unitrace/build   # path may differ — `find / -name libunitrace_tool.so`
docker exec $CONTAINER bash -c "cp $B/unitrace /usr/local/bin/ && cp $B/libunitrace_tool.so /usr/local/lib/ && ldconfig"
# the ldconfig "not a symbolic link" spam about oneAPI .so files is harmless
docker exec $CONTAINER unitrace --version    # expect 2.x.x
```
The full field guide (install, every flush gotcha, ready-to-copy aggregator scripts) lives at `/llm/models/test/unitrace.md` inside the container — read it before driving a server-mode trace.

**Which mode: ALWAYS trace the ONLINE server — never offline, including for decode-kernel work.** This reverses the earlier "offline is fine for decode" advice: an offline serial reproducer fires requests one at a time, so its trace is dominated by *inter-request* host idle and the busy% reads falsely low (launch-bound) AND the kernel-family proportions are skewed by the gaps. The fix (tightening `tail_frac` to one request) is fiddly and easy to get wrong; the online server with a real scheduler gives the deployed busy% directly. Decode-only questions are exactly where this bit hardest (the gemma4-31B/12B runs: offline read ~74-77% busy and over-counted GEMV, online tightened to ~92.7% and surfaced allreduce as the real 51% bottleneck — the offline trace had completely hidden it). The offline recipe below is kept ONLY as a last-resort fallback when you genuinely cannot start a server (e.g. headless CI); if you use it, you must sweep `tail_frac` down and state in your conclusion that the busy% is offline-derived and likely under-counts. Online driving recipe is (A) below.

⚠️ **Always `cd` into a clean empty dir before launching (`cd /tmp/srv && python3 -m vllm...`).** vllm is a namespace package; launching from a cwd that contains anything named `vllm` (or stray module-shadowing files) triggers a circular import `cannot import name 'SamplingParams' from 'vllm' (unknown location)` at server start — even though a direct `python3 -c "from vllm import SamplingParams"` from elsewhere works. Same symptom on every container regardless of branch. The fix is just the `cd`; don't debug the import chain. (The unitrace launch recipe already `cd`s into its output dir, which is why it never hit this — a plain server launch must do the same.)

**(A) Online server (preferred for serving/TTFT).** Foreground unitrace + `vllm serve`; flush by SIGINT to the EngineCore. `docker exec -d` detaches on the *host* side only, so the container's bash stays foreground and the spawned EngineCore inherits the default SIGINT handler (the §2.4 SIG_IGN trap is avoided):
```bash
# launch (foreground inside container; NEVER exec/&/nohup reaching `vllm serve`)
docker exec -d $CONTAINER bash -c '
  cd /tmp/utr_online && \
  NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0 \
  ZE_AFFINITY_MASK=$ZE_DEVS VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1 OMP_NUM_THREADS=8 \
  unitrace --chrome-device-logging vllm serve $MODEL_PATH ... --enforce-eager 2>&1 | tee serve.log'
# wait for "Application startup complete"; head -1 serve.log must show NO "ld.so ... libunitrace_tool.so" error
# drive the REAL workload (e.g. the multi-image request you care about) ×N to build a steady window
# flush — SIGINT the EngineCore that THIS server's api-server forked. NEVER `pgrep EngineCore|head -1`.
# Resolve it as the EngineCore child of the api-server that is actually LISTENING on YOUR port:
APIPID=$(docker exec $CONTAINER bash -c "ss -tlnp 2>/dev/null | grep ':$PORT ' | grep -oP 'pid=\K[0-9]+' | head -1")
EP=$(docker exec $CONTAINER bash -c "for p in \$(pgrep -f EngineCore); do [ \"\$(awk '/PPid/{print \$2}' /proc/\$p/status 2>/dev/null)\" = \"$APIPID\" ] && echo \$p; done | head -1")
docker exec $CONTAINER kill -INT $EP
# poll the LARGEST python3.*.json in YOUR cwd until >1MB and stable (TP>1 → one per Worker_TP).
# the trace filename is the *worker* pid, not $EP — don't grep by $EP, take `ls -S *.json|head`.
```
⚠️ **Wrong-EngineCore trap — the costliest version is MULTIPLE LIVE servers, not just zombies.** `pgrep -f EngineCore | head -1` returns *an* EngineCore, with no guarantee it's yours: (a) a prior killed run left a **defunct** one (→ SIGINT hits nothing → 130-byte trace); (b) **another live server is sharing the box** (e.g. someone's production server on :8002 while you trace on :9010) → `head -1` SIGINTs THEIR EngineCore and **kills their server** (I did exactly this and took down a user's running gemma-4-12B server). ALWAYS resolve the EngineCore as the child of the api-server **listening on your own port** (the `ss`→PPid chain above), never positional `head -1`. Verify your json grows past 1MB after the signal; if 130 bytes you signaled the wrong/instr-less pid (or §2.4/§2.6) — re-check, don't re-theorize. (`docker exec -d` + `tee` is the host-side `&`; do not add a container-side `&`.)
⚠️ **TP>1 → the trace file is named by the Worker_TP pid, not the EngineCore pid.** SIGINT the EngineCore (it cascades the flush to its workers), but collect with `ls -S /tmp/utr_xxx/python3.*.json | head` — you get one ~100-200MB json per worker; pick the largest. `stat python3.$EP.json` will say "No such file".

**(B) Offline reproducer — LAST-RESORT FALLBACK ONLY (busy% under-counts; do not trust for the launch-bound-vs-compute-bound verdict).** Use only when you cannot start a server. Single process, exits cleanly so the destructor flushes with no SIGINT:
```bash
docker exec $CONTAINER bash -c '
  rm -rf /tmp/utr && mkdir -p /tmp/utr && cd /tmp/utr
  NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0 \
  ZE_AFFINITY_MASK=$ZE_DEVS TORCH_LLM_ALLREDUCE=1 CCL_ZE_IPC_EXCHANGE=pidfd \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MLA_DISABLE=1 VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0 \
  timeout 600 unitrace --chrome-device-logging python3 /tmp/run_${MODEL_FAMILY}.py > serve.log 2>&1'
# output: python3.<pid>.json per process (TP=2 → two ~10-15MB worker traces)
```
⚠️ Offline fires requests serially, so the wide-window busy% is dominated by *inter-request* idle and reads falsely launch-bound (see the boxed warning above). Run warmup + a burst of N identical steps and **window tight to one step's execution** (sweep `tail_frac` down) before trusting busy%.

Analyze (aggregator skeleton — auto-detects device pid as the pid with the largest total `ph=="X"` dur; full versions in unitrace doc §5.5/§5.6):
```python
import sys, ijson, collections          # /tmp/utr_agg.py <trace.json> [tail_frac]
pd = collections.defaultdict(float)
for ev in ijson.items(open(sys.argv[1],'rb'),'traceEvents.item'):
    if ev.get('ph')=='X': pd[ev['pid']] += float(ev.get('dur',0))
DP = max(pd, key=pd.get)               # device pid (host-API events live on a different pid)
evs = sorted((float(e['ts']),float(e.get('dur',0)),e['name'])
             for e in ijson.items(open(sys.argv[1],'rb'),'traceEvents.item')
             if e.get('ph')=='X' and e['pid']==DP)
t0,t1 = evs[0][0], evs[-1][0]+evs[-1][1]
cut = t1-(t1-t0)*float(sys.argv[2] if len(sys.argv)>2 else 0.06)   # tail window, skips load/warmup
win = [e for e in evs if e[0]>=cut]; wall=(t1-cut)/1e3
busy = sum(d for _,d,_ in win); fam=collections.defaultdict(float)
for _,d,n in win: fam[n.split('<')[0].split('[')[0][:50]] += d
print(f"busy%={100*busy/1e3/wall:.1f}  (low=launch-bound, high=compute-bound)")
for k in sorted(fam,key=fam.get,reverse=True)[:20]: print(f"{fam[k]/1e3:8.2f}ms  {k}")
# gap profile: sort the >1ms inter-op gaps with their before/after kernel — a recurring
# multi-ms gap between a D2H (read token) and the next M2D (next request) == host dead air,
# not a kernel you can fix. (see /tmp/utr_biggap.py pattern in the doc)
```

**Reading the verdict (only on a single-request window — see the boxed warning):** `busy% low (≈15%)` → launch-bound; the lever is **fewer launches** (graph/fusion) or it's host-bound (preprocess/IPC) — a faster single kernel buys ~nothing. `busy% high (≈80%)` → compute/BW-bound; now Lens B roofline tells you if a kernel rewrite is worth it. Before trusting a *low* busy%, confirm the window isn't padded with inter-request idle (tighten `tail_frac`; if busy% climbs toward 80% it was a windowing artifact, not launch-bound). Two anti-artifact rules carry over: unitrace inflates absolute device-time (trust **relative** %, not the seconds), and `--chrome-call-logging` floods tiny-op counts (cross-check a suspicious "94% elementwise" with `torch.profiler(record_shapes=True)`).

⚠️ **Collective-comm (allreduce/allgather) self-time is OVER-counted — its dur absorbs the in-order queue wait for preceding compute.** On TP>1, a unitrace trace can show `oneccl_allreduce` at ~50% of decode device-time (e.g. gemma4-12B TP=2 showed allreduce 51%, ~200us/call). That does NOT mean communication is the bottleneck. Verified on 021-gc with a controlled torchrun micro-bench: back-to-back pure `tensor_model_parallel_all_reduce` of the same [1,5376] tensor = **21us/call**; the SAME allreduce with a 100us matmul queued before it = **~150-168us/call** (and symmetric-load D=149us ≈ asymmetric C=168us, so it's NOT mainly rank-skew waiting). The allreduce event sits in an in-order queue *behind* the GEMV/norm and its dur counts that queue-drain time. So the real split of that "200us" is ~21us actual comms + ~180us "waiting for the GEMV/norm in front of it." **Mis-reading it as "comms-bound, optimize the allreduce" is wrong** — the lever is fewer sync points / pipeline the step (XPU graph packs the whole step incl. allreduce into one replay, removing these per-collective queue/launch gaps), not a faster collective. **Calibration rule (the collective-comm version of §4.4):** before concluding a collective is a bottleneck, micro-bench it back-to-back in isolation (torchrun, same shape, no other GPU work). If isolated ≪ in-model dur (10x here), the in-model number is mostly queue-wait, not comms. This generalizes: unitrace dur is true wall-time the GPU event occupied, but for any event that sits behind others in an in-order queue, "occupied" includes "waiting" — trust busy% and relative family ranking, never a single kernel's absolute us without an isolated micro-bench.

**When unitrace genuinely hangs / won't flush** (it can, see below): fall back to Lens A monkeypatch/source-instrument timing.

⚠️ **BMG (0xe223) real achievable bandwidth is ~520-580 GB/s, NOT the 456 GB/s "spec peak" — measure it, don't assume.** Calibrated on 021-gc with a pure-read (`x.sum()` on 256MB fp16) = 579 GB/s and copy (read+write) = 543 GB/s. Using the wrong 456 number makes decode GEMVs look like they hit 113-130% of "peak" (impossible) and fabricates phantom headroom. Against the real ~580 ceiling, the gemma4 INT4 decode GEMVs are at **89-102%** (qkv 591=102%, o_proj 560=97%, gate_up 546=94%, down 514=89%) — i.e. **already on the memory wall, no ESIMD-kernel headroom left** (reading the same bytes faster is physically impossible). The down_proj (largest K, 89%) has the only sliver (~11%, from K_SPLIT reduction overhead), worth <2% end-to-end. **Always run the 6-line pure-read/copy micro-bench to get the real ceiling before declaring a GEMV "has headroom" or "is bandwidth-bound."** The lever for a wall-bound GEMV is never the kernel — it's reading fewer bytes (lower-bit quant) or reusing read weights (batch / speculative decode amortizes one weight read over many tokens).

### Phase 5 — Write a new ESIMD kernel

Workflow:

1. **Author kernel header** in `${LLMSCALER_PATH}/csrc/xpu/esimd_kernels/<name>.h`. Copy a similar-shaped existing kernel as the template (`fp8_GEMV_v2.h`, `fused_add_rms_norm.h`, `norm_gemv_norm_fp16.h`, `norm_add_norm.h`). Use `simd<float, VL>` accumulators, `block_load<fp16, VL>` / `block_load<uint8_t, VL>` for fp16 / fp8 weight reads; never recreate the fp8 dequant logic — copy `fp8_dequant_*` from a sibling kernel.

2. **Resource budget — register cache trap.** Avoid `simd<float, VL>[MAX_CHUNKS]` register caches once `VL * MAX_CHUNKS * 4 bytes > 12 KB`. BMG single-thread GRF is ~12 KB; spilling silently runs but accumulates Level Zero state and **eventually triggers `UR_RESULT_ERROR_OUT_OF_RESOURCES`** under server load. Stream loads twice instead — L3 caches the second pass at ~zero cost. (See `norm_add_norm.h` final form.)

3. **K-divisibility trap.** If `K % 64 != 0` or `K` not a clean power-of-2 multiple, the existing `select_vl_ks` in `fp8_GEMV_v2.h` falls back to `vl=32 ks=1` and runs at ~300 GB/s. Use the `fp8_GEMV_bmg.h` pattern: pick `(VL_BIG, VL_TAIL)` where `VL_BIG` chunks fit cleanly and one final `VL_TAIL` chunk handles the remainder.

4. **K_SPLIT for large-N matmuls — but watch redundant compute.** When fusing norm + GEMV, every WG re-computes `sum_sq` over the same K elements. With N WGs that's N× redundant memory traffic. The fuse is a **net win only if N is small** (~hundreds). For N=4096 (qkv_proj), shipping the kernel but **not wiring it up** is the right call (this is what `scaled_resadd_norm_gemv_fp8.h` is — kept as future material for a persistent-kernel rewrite).

5. **Bind the op:**
   - `csrc/xpu/esimd_kernel.sycl`: `#include` the header, write a thin `at::Tensor esimd_<name>(...)` wrapper.
   - `csrc/xpu/torch_extension.cc`: `m.def(...)` schema + `m.impl("...", torch::kXPU, &...)`.
   - `include/kernel_ops.h`: forward declaration.
   - `python/custom_esimd_kernels_vllm/ops.py`: thin python wrapper that calls `_ops.esimd_<name>`.
   - `python/custom_esimd_kernels_vllm/__init__.py`: re-export.

6. **Build:**
   ```bash
   docker exec $CONTAINER bash -c 'cd $LLMSCALER_PATH && \
     rm -f build/temp.linux-x86_64-cpython-312/csrc/xpu/esimd_kernel.o && \
     TORCH_XPU_ARCH_LIST=bmg python3 setup_gemv_only.py build_ext --inplace 2>&1 | tail -8'
   ```
   The explicit `rm` of the `.o` is required when a header changed but the `.sycl` file did not — ninja's dep tracking misses header-only edits. Look for `Build succeeded.` four times and `copying ... .so → python/custom_esimd_kernels_vllm`. If you only edit kernels in `csrc/moe_batch/`, also run `setup.py build_ext --inplace` (the moe ops are a separate extension).

7. **Unit-test the kernel against a reference:**
   ```python
   # /tmp/test_<name>.py
   ... call esimd op + compute torch reference + assert max abs diff < threshold
   ```
   Always check both numerical (max abs diff < ~0.01 fp16, < 0.5 for cumulative GEMV) and shape-edge cases (K not divisible by VL, K_SPLIT > 1, etc).

### Phase 5b — MoE decode: the fused-full-op pattern (preferred end state)

For **MoE models**, the decode bottleneck is usually NOT the expert GEMM — it's the **routing segment**: the topk/softmax kernel launch plus the Python-side glue (`.to(fp16)`, `.contiguous()`, scale-fold gather-mul, separate expert dispatch). On gemma-4-26B this segment was 0.133 ms/layer (×30 = 4 ms) — *larger* than the expert GEMM (0.087 ms/layer). **Always seg-profile MoE forward into `routing | expert-kernel | all_reduce` before optimizing — don't assume the GEMM dominates.** (Patch the model's MoE `forward` with `torch.xpu.synchronize()+perf_counter` brackets around each segment, env-gated.)

The proven end state mirrors `qwen3_next.py`'s decode path: **`router_op(x) → one fused full op(x, logits, ...) → all_reduce`** — a single op that does topk + (any scale fold) + up + activation + down + accumulate internally, so the only Python-visible launches are the router and the fused op.

How to get there (gemma-4 worked example, commits in `optimize_gemma4_continue`):

1. **First fix the expert GEMV load width (decode M==1).** The shared MoE kernel uses `lsc_load_2d<uint8_t,16,16,1>` (DPAS GEMM, tuned for prefill M>1). On decode that 16-byte-wide 2D load fills only 1/4 of BMG's 64B cacheline → ~315 GB/s. Expert weights are plain row-major inside each expert (`gate_up [E,2*inter,hidden]`, `down [E,hidden,inter]`, K contiguous), so a **1D `block_load<uint8_t,256>` along K** (the `fp8_GEMV_bmg` pattern) restores ~528 GB/s. Write decode-only `MoeUpDecode*`/`MoeDownDecode*` kernel structs in `csrc/moe_batch/<name>.h`. Gate `x.size(0)==1`; prefill keeps DPAS.

2. **Fuse routing INTO the op — but topk MUST be fp32-internal.** This is the single most important correctness rule. A fp16-internal softmax+topk (e.g. `esimd_moe_topk`) **diverges from the triton routing on near-tie top-k boundaries and silently drops gsm8k 5/5 → 4/5**, even when a single-shot unit test matched bit-for-bit. The production `dispatch_moe_topk_forward`/`moe_topk` op IS fp32-internal — verify it matches the reference routing (200-trial id-set match, weight diff < 2e-4) AND holds 5/5, then reuse it inside your fused op. Never reverse-engineer a new topk.

3. **Fold any model-specific routing scale on-device.** gemma's `per_expert_scale` (learnable, per-expert) can't be expressed by the generic topk's `norm` flag, so a tiny `MoeFoldExpertScale` kernel multiplies it into the topk weights between topk and the down kernel — no round-trip to Python. This per-model scale is exactly why you **cannot** reuse another model's `moe_forward_full` verbatim (qwen's has no scale fold, uses silu not gelu_tanh, and assumes a shared expert).

4. **Assemble the fused op** `moe_forward_full_<act>_decode(x, logits, w13, s13, w2, s2, <scale...>, top_k, n_experts)`: internal `dispatch_moe_topk_forward(norm=true)` → scale-fold kernel → `MoeUpDecode` → `MoeDownDecode` → `moe_accumulate_kernel`. Register + python-wrap as in Phase 5.5; wire into the model's MoE `forward` for decode, return early before the old routing/topk/expert sequence; prefill path unchanged.

5. **Prefill stays fp32-routing + DPAS.** Prefill (M>1) is precision-sensitive to routing-weight dtype (fp16 routing weights there also drop 5/5) and DPAS is the right kernel for M>1. Keep the two paths split by `x.size(0)==1`.

gemma-4-26B result: decode ITL 21.86 → 18.72 ms (−14%), gsm8k 5/5, token fingerprint identical. Per-op env gates (`DISABLE_MOE_DECODE_GEMV`, `DISABLE_MOE_PROD_TOPK`, `DISABLE_MOE_FULL_FUSED`) layer the steps so each is independently A/B-able and revertible.

### Phase 6 — Wire into the model

**Always add an env gate that disables the new path 1:1**, e.g. `DISABLE_<MODEL>_FUSED_<NAME>=1` defaulting to OFF. This is required to make subsequent A/B comparisons trivial and to give the user an emergency disable when a corner case breaks accuracy in production. Pattern:

```python
_fused_path = (
    ESIMD_AVAILABLE
    and not disable_esimd_norm()           # global gate
    and tensor.shape[0] == 1                # decode-only
    and tensor.is_contiguous()
    and tensor.dtype == torch.float16
    and weight.dtype == torch.float16
    and os.environ.get("DISABLE_<NAME>", "0") != "1"
)
if _fused_path:
    if not hasattr(self, "_buf"):
        self._buf = torch.empty_like(tensor)   # cache on the module
    from custom_esimd_kernels_vllm import esimd_<name>
    esimd_<name>(...)
    return self._buf
else:
    # original path, unchanged
    ...
```

### Phase 7 — Verify (mandatory after every kernel change)

Always run **both** before claiming any kernel change works:

1. **Phase 1 reproducer**: token-id list must match the recorded baseline. If it deviates, the optimization is **not** functionally equivalent and you must explain why before continuing.
2. **Phase 2 chat reqs replay**: `Accuracy: 5/5 = 1.000`. Less than that means a different shape (long prompts, prefill chunk M>1) is broken — common failure mode for kernels that tested fine in isolation.
3. **Phase 3 ITL bench**: median of 5 runs improved (or at least did not regress beyond noise). If your change doesn't improve ITL, either disable the wire-up by default and ship the kernel as future material (like `scaled_resadd_norm_gemv_fp8`), or revert.

### Phase 8 — Bisect when something breaks

If accuracy regresses anywhere (Phase 1 token mismatch / Phase 2 acc < 5/5), **stop adding things and bisect**.

The bisection priority list, in order:

1. **First, env-gate disable everything** — set every `DISABLE_*` env var listed in "Standing assumptions". If accuracy is restored, the bug is in our optimization stack; otherwise it's elsewhere (config, attention backend, vllm-xpu base).
2. **Compare against `$BASELINE_CONTAINER`** running the exact same offline script on the same prompt. If baseline is also broken, this isn't your bug.
3. **Diff `vllm/...` files between the two containers.** Frequent suspects: `vllm/v1/attention/backends/flash_attn.py` (`supports_head_size`), `vllm/model_executor/models/config.py` (forced backend selection), `vllm/model_executor/models/<family>.py` itself.
4. **Git bisect within `$VLLM_PATH`** — checkout an older commit, rerun Phase 2. Bisect interval halves each step. We have caught bugs this way at three distinct levels: (a) MoE kernel called with `M>1` prefill chunk produces NaN (`f7c0693e9`), (b) `GeluAndMul` esimd path NaN at large `d` (`28d462ed5`), (c) attention backend selector chose FLASH_ATTN where head_size=512 silently NaN'd (`config.py` XPU exception).
5. When you do find the offending commit, **fix by gating, not reverting** — narrow the activation condition (`d <= 4096`, `M == 1`, etc.) so the optimization keeps working in its safe regime. Always add the new gate as both a hard condition and an env override.

### Phase 8b — Wire-up 阶段的坑清单(kernel 隔离对但接进模型就坏)

kernel 单测过了、接进 vllm 却崩/慢/乱码，几乎都栽在这些**集成路径**问题上(不是 kernel 数学)。逐条 check:

- ⭐**隔离测试 PASS ≠ e2e 正确**——头号陷阱。三种成因:
  - **ref 与 kernel 共享同一套可能错的约定** → 永远吻合。必须对**真实 Triton/官方 golden**;
    TP 场景要拿**两 rank 相加**比 all-reduce 后的 gold(别拿单 rank partial 比，会误判成 cos 0.8"语义错"，正确是 0.9997)。
  - **单测没覆盖真实 regime**:单测用 q_len≤window 让 sliding 窗口没起作用、用内置 topk 而模型用 routed 变体、用小合成权重侥幸避开满量程 fp8(|w|→288)溢出。→ 单测必须用**贴近真机的 shape/量级/真实 dump 的 in-model 张量**。
  - **两个完全不同的 kernel 接进同一路径产生相同失败(0/5+同样慢)** → 铁证 bug 在共享集成路径(routing 布局/gather/accumulate/scale 重量化)，不是 kernel。决定性 debug = 集成路径同时算 Triton golden 逐环节 dump diff。
- **旁路 FusedMoE 的 4 个必踩坑**:① 模型特有 routing scale(如 gemma4 `per_expert_scale` 0.98-1.02)内置 topk 不含 → 需 `_routed` op 接受外部 routing_weights/indices;
  ② **TP 下漏 all-reduce** → 绕过 `FusedMoE.runner` 就得手动补 row-parallel all-reduce，否则输出全 0;
  ③ **thread_local 单 buffer 跨层复用**(所有 MoE 层共享 `sg_final_output`，下层覆盖上层) → narrow+clone 或 kernel 内 ring buffer(`SG_FINAL_RING=4`);
  ④ **M≥16 prefill 累加器精度不足输出 NaN** → 限制 esimd 只走 decode(gate `x.size(0)==1`)。
- **profile_run 全 0 输入会误触 esimd 路径 + 让对拍误判**(out_norm=ref_norm=0 假性通过) → 门控要能识别/排除 profile_run。
- **fp16 累加器溢出**:per-tensor scale 应在 fp16 累加**前**折叠进权重 tile(累加后 K=2816 raw 部分和溢出 65504→inf);gelu_tanh 的 exp clamp inner ±30 防 NaN。单个 NaN 经残差污染整批 → 0/5 空输出。
- **gate 认知更新**:`x.size(0)==1` 注释"M≥16 fp32 累加器溢出 NaN"**可能已过时**(当前 fp8 kernel fp32 DPAS 累加 T=1..256 单测全对);真实拦阻常是**性能**(per-route M=1 DPAS 无权重复用)+ routed batch 集成 bug，不是数值。判断量化类型用 `quant_config.get_name()` 而非 `hasattr(layer,"weight_scale")`(后者在 `process_weights_after_loading` 之前永远 False)。

### Phase 8c — 给 ESIMD attention kernel 加原生 GQA 支持

给 ESIMD `page_attn_decode`(eagle kernel)加原生 GQA=2(去掉 vllm 的 pad-to-4)的通用手法:

- ⭐**先破解 kernel 里的"魔法常数"到底映射什么**——常是多义复用。实战:GQA=4 kernel 里的 "4" **不是 gqaRatio**——
  Phase1 的 4 线程实际是**切 head_dim(256=4×64)做 QK 部分点积**，只是 softmax/reduction 阶段"复用为 4 个 q-head"(数字巧合);Phase2 的 2 线程是**切 token**。
- **零回归模式**:新增独立 device 函数(`sdpaDecodeGqa2Phase{1,2}`)而非改原函数;**保留完全相同的线程几何和 simd 宽度**，只改(1)q-head 循环 4→2、(2)q-head→SLM 映射、(3)多余线程在 `barrier()` **之后** early-return(避免 barrier 分歧)。计算量自然减半(只算真 q-head)。
- host 按 gqaRatio 分派;`gqaGroups=gqaRatio/4` 在 ratio=2 时整除得 0 会崩 → 用 `gqaGroupsLaunch=(gqaRatio>=4)?gqaRatio/4:1` 兜底;`eagle_ops` schema/host 签名一字不变。
- fp8 走 DPAS 时:A 矩阵 RepeatCount=8，GQA=2 只填前 2 行，padding 行算 0 被 C 累加器天然忽略——不用改 DPAS 形状。
- **三级数值门禁**:(1) PyTorch SDPA 绝对 golden、(2) padded-to-4(老 kernel) 、(3) native GQA=2。门槛先 padded-vs-ref(fp16~2e-5/e4m3~6e-4/e5m2~3e-5)，再要求 **native-vs-padded ≈ 0(逐元素，实测 ≤8e-6)**。**必须扫长序列 16k/32k/64k + 跨 phase 分界形状(batch>1、多 kv-head、跨 1024 分界)**，否则漏长上下文 bug。
- **性能做 kernel 级 A/B**(端到端短上下文会被 attn 占比小 + Triton 固定开销误导)。kernel 级实测 native 比 padded 快 1.4-1.85x。commit 分支 `feat/page-attn-gqa2-native`。
- ⚠️ **page_attn_decode 完全不识 sliding window**(只传 `seqused_k` 对全部 kv 做 attention)。gemma4 sliding_window=1024，prompt>1024 时 sliding 层应只看最近 1024，page_attn 却看全程 → 长上下文输出偏离。这是 esimd page_attn 对 sliding 模型的**固有缺陷**(GQA=4 padded 同样没传 window)，不是 GQA 改动引入的。sliding 模型长上下文 decode **走 varlen**(既快 7-10% 又唯一正确)。

### Phase 8d — fp8 / DPAS / VNNI 的硬件坑(写任何 fp8 kernel 通用)

- **DPAS `<8,8>` 必须 fp16 累加器**:float-acc 版是 `#ifdef` 死代码(从没编译过)，XeLPG/mtl-h 不支持该 intrinsic。
- **BMG 上 `dpas RepeatCount=4 + fp16` 产 NaN + 非确定性**(未文档化) → 用 RepeatCount=8 + 前几行 A 填 0 padding。
- **dpas 下 OOB 字节的 NaN 会经 K 维累加扩散到所有 N=token lane** → reshuffle 时显式 zero-merge(scalar MAC 靠最终 merge 兜底、dpas 不行)。
- **`lsc_gather<u32,NElts,16>` 返回 SoA 布局**(先 16 lane 的第 0 个 u32，再第 1 个)，不是 lane 内 AoS;fp16 用 NElts=4 已默认依赖此 SoA，写 fp8 按 AoS 解读会 q-head 重复。
- **`block_load<u8,32>` 在 BMG 上不兼容**(32 bytes < 最小 block-load 尺寸) → 最低 VL=64 + tail overlap-read(从 `K-VL` 起读最后一段、mask 重叠)。
- **VNNI/DPAS B 矩阵 layout 转置**:fork vs 上游权重 layout 可能镜像(`[E,N,K]` vs `[E,K,N]`);只 swap 2D-load 坐标不够——`fp8e4m3_block_to_vnni` 仍按原 major 解读 → B 语义被转置 → 乱码。**ESIMD 的 Transposed block_load 对 fp8/u8 不支持**(只支持 u32/u64) → 硬件转置行不通，得新写 `_nk` 版 vnni pack 用 stride-16 select 在寄存器内逻辑转置。
- **K 对齐坑(数值正确但输出错)**:`select_vl_ks()` 最低到 VL=128 时，K 不是 128 倍数会 `block_load` 越界读垃圾(如 K=1056/704)→ decode 从第 2 个 token 起错;**独立 random-weight 测试测不出**(测试用的 K 恰好对齐)。修:kernel 端支持 VL=64 + tail。1056=2^5·3·11 无法被任何 VL≥64 整除是关键约束。
- **ESIMD 无 fp8→fp16 native cast、dpas 不吃 fp8**;`lsc_load_2d` 换 `lsc_gather` 常因转置 reshuffle 反而慢 20-30pp(代数等价不代表快)。
- **编译坑**:`TORCH_XPU_ARCH_LIST=bmg` 只编目标设备;大 `simd<fp16,2048>` 权重 buffer + 运行时 index → GRF indirect addressing 溢出("spans complete GRF file") → 改逐 16-K `lsc_load_2d` 小 tile;全量 `build_ext` 会撞无关 extension "translation unit too large" → 写 `setup_<feature>_only.py` 单 extension 编;注意 `esimd_gemm_fp8_pert` 在 `esimd_kernel_gemm.sycl`(另一个 extension)，只编 gemv_only 会漏掉 GEMM 改动。

### Phase 8e — fused kernel 的硬约束

- `esimd_fused_add_rms_norm` 硬编"weight 需预调 w+1.0"(Gemma 风格)，但标准 RMSNorm(`x*w`)不加 1 → 上 fused norm 前**必须验证语义**(gemma4 实测直接 `normed*w` 可用不需 +1)。
- **GRF 8KB/12KB 限**:三合一 fuse(如 `resadd_norm_gemv` K=2816 需 ~11KB GRF)寄存器溢出不可用;`simd<float,VL>[MAX_CHUNKS]` 超 12KB 会 silently spill → server 负载下 `UR_RESULT_ERROR_OUT_OF_RESOURCES`。硬编 VL=512 需改成按 K 选(2816%512≠0)。
- **M=8/16 esimd_gemm 反而慢**(-35%~-42%):esimd WS kernel single-thread-per-output-row，大 M 干不过 onednn tiling → decode 阈值压到 M≤4(实践 M==1)。
- **fp8 大 M GEMM 别默认比 fp16 快**:`torch._scaled_mm` 在 BMG 可能走软件 dequant 路径(74 vs fp16 131 TFLOPS)。见 anti-patterns。

## Anti-patterns (don't waste time on these)

- **Don't dismiss unitrace — but scope it right (this reverses the old blanket "never use unitrace").** It is the *fastest* way to get the launch-bound-vs-compute-bound verdict and the kernel-family breakdown (Phase 4 Lens C) — reach for it FIRST when you don't yet know where the time goes. The real failure mode is narrow: a **server-mode** trace with `--chrome-call-logging` on a heavy **ESIMD decode** loop can hang the worker or refuse to flush (it needs SIGINT to the EngineCore, not the api-server — full gotcha list in `/llm/models/test/unitrace.md`). Mitigations that make it reliable: **always trace the online server** (offline busy% under-counts and hides bottlenecks like allreduce — see "Which mode" above), resolve the EngineCore by your own port's api-server (not `pgrep|head -1`), start with `--chrome-device-logging` only (add `--chrome-call-logging` only when you specifically need host submit-gaps), and wrap in `timeout`. If it still won't flush after a couple tries, **don't chase it** — fall back to Lens A manual `torch.xpu.synchronize() + perf_counter` instrumentation.
- **Don't bypass `vllm` linear with hand-rolled `esimd_gemv` + manual `tensor_model_parallel_all_reduce`.** It's what we tried first; it's measurably slower because vllm linear's internal allocator and dispatcher are already cheap. Only intercept inside the linear method (e.g. `XPUFP8ScaledMMLinearKernel.apply_weights`).
- **Don't write fuses where every WG redundantly recomputes a global reduction.** Counter-example: gating `(h+r) → norm → fp8 GEMV` with 4096 work-groups each redoing K=2816 sum_sq is a net loss. If you must fuse, write a persistent kernel with a global atomic counter. Most of the time a simple two-launch sequence with cached buffers is faster.
- **Don't enable `enforce_eager=False`.** The model definition is not torch.compile-clean; you'll spend hours chasing meta-tensor errors. The whole optimization narrative assumes eager-mode dispatch.
- **Don't try to make `M > 1` prefill chunks share the decode kernel.** All MoE-style ESIMD kernels are tuned for `M == 1`. The prefill-chunk fallback to Triton is the right behavior. Set the gate as `x.size(0) != 1` (decode only) rather than `x.size(0) <= 8`.
- **Don't assume `apply_chat_template(..., tokenize=True)` returns a list.** It returns a `BatchEncoding`. Use `tokenize=False` then run the tokenizer directly to get a clean `list[int]`.
- **Don't optimize the MoE expert GEMM before seg-profiling the routing.** The expert kernel is the *obvious* target but usually not the bottleneck — routing + Python glue dominated on gemma-4 (4 ms vs 2.6 ms). Measure `routing | expert | all_reduce` first. (And: a 1D-load expert GEMV that micro-benches 1.68× faster may only buy ~0.15 ms end-to-end — micro-bench overstates kernel weight; always confirm in-model.)
- **Don't size the prize with a no-op (`SKIP_X=1`) substitution — it measures the op's *total* cost, not the *replaceable* cost.** On MiniCPM-V 4.6 an env-gated `SKIP_GELU=1` (gelu → identity) cut TTFT 65 ms, so the gelu looked worth a kernel. But that 65 ms is overwhelmingly the **unavoidable memory traffic** of reading+writing the (M, 4304) ≈ 11 MB×2 activation — *any* gelu kernel pays it. A hand-written ESIMD gelu that micro-benched 22% faster on the **compute** part bought ~2% end-to-end (lost in noise). A no-op deletes the memory traffic too, so it flatters every memory-bound elementwise op. The honest prize for *replacing* a kernel is **current-kernel time vs its roofline floor** (`max(bytes/BW, FLOPs/peak)`), not op-vs-absent. A no-op only reflects real prize when you can genuinely *delete* the op (fuse it into a neighbor's epilogue so the round-trip disappears, or remove it algorithmically) — and even then the win is the saved round-trip, which on a compute-bound layer largely overlaps the GEMM and shrinks again (resadd+LayerNorm fuse: ~0.4 ms/call isolated → ~3% end-to-end, not worth it).
- **Don't assume fp8 beats fp16 for large-M GEMM on XPU — measure `_scaled_mm` first; it can be 2× *slower*.** Expectation: fp8 doubles XMX throughput. Reality on BMG / this torch-xpu stack for a vision-tower-sized GEMM (M=30240, K=1152, N=4304): `torch._scaled_mm` fp8 = **74 TFLOPS vs fp16 `F.linear` = 131 TFLOPS** — fp8 never reaches the XMX fp8 path and runs a software/dequant route. The existing `esimd_gemm_fp8_pert` only covers M≤64, so it doesn't help large-M prefill either. Upshot: for a **compute-bound** large-M layer (e.g. a ViT encoder), fp16 oneDNN GEMM at ~55% of peak is already the best available path; you cannot break the compute bound by dropping to fp8 here. (fp8 *is* the win for **decode** M=1 GEMV, which is BW-bound — different regime.) Always run the 3-line `_scaled_mm`-vs-`linear` TFLOPS micro-bench before committing to an fp8 rewrite.
- **Don't replace a routing/topk with a fp16-internal kernel.** Topk is tie-break-sensitive: fp16-internal softmax+topk diverges on near-ties and drops accuracy (gsm8k 4/5) while passing single-shot unit tests. Only swap in an fp32-internal topk (the production `moe_topk`), and gate it behind the full Phase-2 gsm8k 5/5, never a unit test alone.
- **Don't reuse another model's `moe_forward_full` verbatim.** Per-model routing semantics differ — gemma folds a learnable `per_expert_scale` (qwen doesn't), uses gelu_tanh not silu, and has no shared expert. Write a model-specific fused op; share only the sub-kernels (topk, GEMV, accumulate).

## Final commit & rebase guidance

When the user is happy with a series of optimization commits, **expect to rebase onto the upstream release branch** (`origin/downstream/release/v0.21.0` typically). Two commits in our history overlap with later upstream cherry-picks of the same PR; `git rebase` will:
- Auto-`drop` commits whose patch is already upstream.
- Throw a small conflict on `vllm/model_executor/layers/esimd_utils.py` for the very first ESIMD infrastructure commit; resolve by `git rebase --skip` (upstream's version is functionally equivalent — your later commits will re-add the additional re-exports they need).

After rebase, recheck Phase 1 and Phase 2 (a successful textual rebase doesn't guarantee semantic correctness).

## Reference assets

The following helper scripts in `assets/` are templates — each one needs only `MODEL_PATH`, `TP`, and a per-model PROMPT_IDS update:

- `offline_chat_reqs.py` — Phase 2 accuracy replay
- `bench_itl.py` — Phase 3 ITL benchmark
- `profile_attn.py` — Phase 4 Lens A attention timing harness (PROFILE_ATTN=1)
- `unitrace_agg.py` — Phase 4 Lens C: aggregate a unitrace chrome trace into the device-busy% verdict + kernel-family breakdown + gap profile (auto device-pid, tail phase-slice)

Read them before writing your own. The full unitrace field guide (install, flush gotchas, server-mode driving, more parser variants) is `/llm/models/test/unitrace.md` inside the container.
