---
name: xpu-graph-enable
description: 在 Intel XPU(BMG/Arc) vLLM 上为模型启用 XPU graph(cudagraph capture/replay)消除 decode 的 kernel-launch 开销提性能(实测 decode TPOT 小 batch 最高 2.45x)。适用于：用户说"为什么 XPU graph 没效果/没生效""enable cudagraph/torch.compile 提速""decode launch-bound 想上 graph""多卡 graph 跑不起来/capture 崩/replay hang""cudagraph_mode/capture_sizes 怎么设""raw pybind 与 torch.ops dispatcher 区别"。覆盖：排查顺序(env 没生效是最高频)、多卡(TP≥2)启用三处必改、FULL_DECODE_ONLY vs PIECEWISE/FULL 选择、capture 崩溃(work_group_scratch)与 replay hang(MoE/ESIMD raw pybind→dispatcher)诊断、capture_sizes 约束、graph 下"先跑大 batch/高并发(gsm8k)再跑单条请求就输出 !!!!/感叹号"(ESIMD kernel static buffer 随 n_tokens 重分配致 graph 固化地址失效，eager 正常)的诊断与修复。前提：torch 2.11+xpu、vllm-xpu(release v0.21.0 线)、能改 vllm 源码、ESIMD kernel(custom-esimd-kernels/vllm-xpu-kernels)。
---

# 在 Intel XPU vLLM 上启用 XPU graph 提 decode 性能

decode 在 XPU 上是 **launch-bound**(每步几十~上百个小 kernel launch，host 开销主导)。
XPU graph 把固定 shape 的 decode 整步 capture 成一个 graph，replay 时跳过所有 host 端
launch → 小 batch decode TPOT 提速 1.5-2.5x(batch 越大收益越小，计算占主导)。
**实测(Qwen3-Coder-Next 四卡 fp8, bsz=1, 1k input)：eager 15.93ms/tok → graph 6.50ms/tok = 2.45x**
(153.8 tok/s，4 trial 方差<0.1%)。

源自 gemma-4-26B / Qwen3-Coder-Next 在 vllm-xpu v0.21.0 上 enable XPU graph 的实战。
权威交接：`/llm/models/test/opt_gemma4/HANDOFF_xpu_graph.md`。

## ⭐⭐ 排查顺序(先查最高频的，别上来就猜 kernel)
「设了 graph 没效果」按这个顺序查，90% 卡在前两步：
1. **env 没真正传进 worker 进程**(最高频，最先查)：启动日志若有
   `xpu.py:... XPU Graph is disabled by environment variable` → `VLLM_XPU_ENABLE_XPU_GRAPH=1`
   根本没生效(常见：写在脚本里但位置不对/没 export/没传进 docker exec 的子进程)。先确认这条没出现。
2. **graph 被降级/跳过**：日志有 `Skipping CUDA graph capture` 或 `Overriding to NONE`
   → 见「三种 cudagraph_mode」「多卡三处必改」。
3. **capture 崩 / replay hang / 输出乱码** → 见「两类运行期故障」A/B/C。
4. **capture 成功但 TPOT 没降**：多半是测的 batch 太大(收益被摊平)或 capture_sizes 没覆盖到实际
   batch(走了 eager fallback)。见「cudagraph_capture_sizes 约束」。
- 成功标志：日志 `Capturing CUDA graphs (decode, FULL): N/N` + `Graph capturing finished`。

## 何时用 / 不用
- **用**：decode launch-bound、想上 graph 提速；或 graph「设了开关没效果/capture 崩/replay hang」。
- **不用**：精度/输出不对(先用 `hf-golden-layerwise-diff` 或确认 eager 基线是对的——见下「第 0 步」)；
  纯 kernel 优化(用 `vllm-xpu-esimd-optimize`)。

## ⭐ 第 0 步：先确认 eager 基线是对的(最省时间，别跳)
graph 只是 capture/replay eager 的执行，**不改变数值**。动 graph 前先 `--enforce-eager` 跑同一请求：
- eager 输出就 `!!!!`/乱码 → **是模型/量化/kernel 基线问题，与 graph 无关**，先去修基线(往往是某 kernel
  PR 没 merge / fp8 路径未适配)。实战：Qwen3-Coder-Next fp8 四卡 eager 就 `!!!!`，graph 白做。
- eager 正常、graph 才坏 → 才是 graph 的 capture/mutation 问题(往下走)。
- 用「stash 掉你所有改动跑原始 release eager」来证明坏的是基线不是你的改动。

## 开关与三种 cudagraph_mode
```
VLLM_XPU_ENABLE_XPU_GRAPH=1
--compilation-config='{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```
- `mode:NONE` 必带：BMG 上 dynamo/inductor 会触发 IGC/ocloc 编译崩；只用 CUDAGraphWrapper 本身 capture。
- **FULL_DECODE_ONLY**=(FULL,NONE)：整个 decode step 一个 graph。`requires_piecewise_compilation()`
  返回 **False** → 配 mode=NONE **不被降级**。XPU 上的推荐路径。但整步 capture，attention/MoE 都进 capture 区。
- **PIECEWISE**：按算子切(attention/MoE 在 splitting_ops 里会被切出去)。但 `requires_piecewise_compilation()`
  为 **True** → 配 mode=NONE 会被 `config/vllm.py` 降级成 NONE(graph 不生效)；必须配 mode=VLLM_COMPILE
  (走 inductor，BMG 上又崩)。所以 PIECEWISE 在纯 XPU(mode=NONE)路线基本用不了。
- prefill 不 capture(FULL_DECODE_ONLY 下 prefill 走 eager)，TTFT 不变，收益全在 decode。
  想 capture prefill 提 TTFT 要 FULL/FULL_AND_PIECEWISE，但 prefill 的 varlen 用 work_group_scratch
  → capture 崩(kernel 限制，纯 vllm 无解)。
- 另 `--async-scheduling` 是纯收益：CPU preprocess 与 GPU decode 重叠，再 −10% TPOT(~0.7ms)，建议带上。

## cudagraph_capture_sizes 约束(不能无限大)
- `max_cudagraph_capture_size` 默认 = `min(max_num_seqs*2, 512)`，**硬上限 512**。
- **超过 `max_num_seqs` 的 size 无意义**：dispatcher 运行时把 decode batch cap 在 max_num_seqs，
  更大的 graph 永远用不上。
- 每多一个 size = 多 capture 一个 graph：增启动时间 + 占显存(每 graph 固化一份中间 buffer)。
- 没列到的 batch size → 运行时向上取最近的已 capture size(pad，多算几行)或 fallback eager，不报错。
- **正确做法：按你真实负载的并发分布列**，如常用 1/4/8/16 并发就 `[1,2,4,8,16]`，别盲目堆大。
  graph 收益集中在小 batch(bsz=1 最高 ~2.5x，bsz≥8~12 趋近 eager，launch 开销被计算摊平)。

## 多卡(TP≥2)启用：三处必改(否则多卡根本没 graph)
gc 初版 XPU graph 常只对 TP=1 验证；多卡(gemma4/qwen 唯一跑法)被各种逻辑关掉。改 `vllm/platforms/xpu.py`
+ `vllm/distributed/parallel_state.py`：
1. **去掉 `world_size_across_dp>1 → cudagraph_mode=NONE`** 一刀切禁用(这是「多卡没效果」最常见根因)。
2. **放行 FULL_DECODE_ONLY**：删掉/放宽 `FLASH_ATTN 下 cudagraph_mode not in {NONE,PIECEWISE} → 强制
   PIECEWISE` 那条(它会把你设的 FULL_DECODE_ONLY 改写掉)。
3. **`graph_capture()` 去 CudaCommunicator 硬断言**(`parallel_state.py`)：
   `assert isinstance(self.device_communicator, CudaCommunicator)` 在 XPU(xccl communicator，无 ca_comm)
   必崩。改成软探测 `getattr(self.device_communicator, "ca_comm", None)`，XPU 走 no-op capture context，
   per-step allreduce 在 FULL_DECODE_ONLY 区内一起 replay。
   参考 commit：本仓 `28f3749af`(intel-sandbox vllm-xpu，分支 enable_gemma4_xpu_graph)。

## 两类运行期故障的诊断

### A. capture warmup 崩：`work_group_scratch_memory ... not yet available for use with SYCL command graph`
- 含义：capture 区里有 kernel 用了 SLM scratch(`sycl_ext_oneapi_work_group_scratch_memory`)，
  command graph 不支持。
- 实战根因(gemma4)：decode **attention** 走 `varlen_fwd`(用 scratch)而非 ESIMD `page_attn_decode`
  (不用 scratch)。看栈底是哪个 kernel(`torch.ops._vllm_fa2_C.varlen_fwd` = 用 scratch 的)。
- 出路：让 decode attention 走不用 scratch 的 kernel(ESIMD page_attn)。它有 gate(典型
  `head_size==256 and GQA>=4`)，不满足就 fall through 到 varlen → 崩。
- 实战(gemma4-26B，分两种层)：
  - **sliding 层(25/30，head_dim=256，GQA=2)可救**：release 入口 gate 卡 `GQA>=4` 但内部已有
    `GQA>=2 && %4!=0` 的 pad-to-4 路径(flash_attn.py:1051-1081，死代码被入口挡死)；把入口 gate 放宽到
    `>=2` 即走 ESIMD page_attn(graph-safe，eager 验证数值正确)。⚠改了 decode kernel 路径，commit 前跑 GSM8K。
  - **full 层(5/30，head_dim=512)硬阻塞**：`page_attn` kernel 硬编码 `TORCH_CHECK(headDim==256)`
    (llm-scaler main eagle.sycl)处理不了 512，只能走 varlen → capture 崩。
  - 想「只排除 full 层 attention 出 capture、其余进 graph」也不行：只有 PIECEWISE 能用 splitting_ops 排除
    attention，而 PIECEWISE 在 XPU(mode=NONE)必被降级(cudagraph_dispatcher.py 的 assert)。死结。
  - **唯一出路 = kernel 侧**给 head_dim=512 加 page_attn 式 graph-safe decode kernel，或 varlen 改预分配 scratch。
  - 对照：Qwen3-Coder-Next 全 head_dim=256/GQA=8 → 全层命中 ESIMD page_attn → graph 一把过(已验证 2.45x)。

### B. replay hang：worker 卡死 → EngineCore `TimeoutError`(shm_broadcast get_response 超时)
- 含义：capture 成功，但 replay 时某 kernel 有 host-side alloc/sync 无法重放 → worker 卡住。
- 实战根因(qwen MoE)：MoE kernel(host-side scatter/gather sync)**留在了 capture 区内没被切出去**。
- 修复：让该 kernel 在 fx graph 里是一个**真正的 dispatcher op 节点**，并注册进 `splitting_ops`：
  - vllm 调用点从 **raw pybind 函数**(`from esimd_utils import moe_forward_full; moe_forward_full(...)`)
    改成 **`torch.ops.moe_ops.moe_forward_full(...)`**(实参不变)。raw pybind 调用在 fx graph 里不是节点，
    splitting 切不到。
  - `platforms/xpu.py` 在 PIECEWISE/FULL_DECODE_ONLY 下把 `moe_ops::moe_forward_full` 等加进
    `compilation_config.splitting_ops`(让 MoE 留在 capture 区外跑 eager)。
- ⚠️ **验证 dispatcher op 是否注册别信 `dir(torch.ops.ns)`**(惰性加载列不全，会误判「namespace 空」)。
  用 `torch._C._jit_get_schemas_for_operator("moe_ops::moe_forward_full")`，不抛异常=已注册。

#### 核心概念：raw pybind 调用 vs torch.ops dispatcher 调用(为什么 splitting 切得到/切不到)
同一个 C++ kernel 有两种暴露给 Python 的方式，**执行/数值完全一样，区别只在 PyTorch 认不认识它**：
- **raw pybind**(`PYBIND11_MODULE` 绑的，`_esimd.moe_forward_full`)：对 PyTorch 是**黑盒普通函数**，
  无 schema、不走 dispatcher、**fx graph 里没有它的节点** → splitting_ops 找不到去切、register_fake/
  functionalize 都处理不了 → 被卷进 capture 区。
- **torch.ops dispatcher**(`TORCH_LIBRARY` 注册的，`torch.ops.moe_ops.moe_forward_full`)：是 PyTorch
  一等算子，有名字+schema(类型/设备/mutation)，**fx graph 里作为具名节点出现** → splitting_ops 能精确
  匹配并切走、能 register_fake、能 functionalize。
- 类比：raw pybind = 外包(活能干但花名册没名字，调度不到)；dispatcher = 正式入职有工号(能排班)。
- **前提**：kernel 的 C++ 侧得有 `TORCH_LIBRARY(moe_ops, m){ m.def(...) }` + `TORCH_LIBRARY_IMPL`。
  实战中这套早在 llm-scaler main 里(含 gemma4 的 moe_forward_full_gelu_tanh*)，vllm 侧只需改调用入口，
  **kernel 零改动**。splitting_ops 列名字(platforms/xpu.py) + 调用走 dispatcher(模型文件) 必须配套，缺一不可。

### C. capture 成功但输出 `!!!!`(legacy schema mutation)
- 含义：capture 区里某 ESIMD in-place op 的 C++ schema 是裸 `Tensor`(无 `(a!)` mutation 标注)，
  functionalization 引用了 mutation 前的旧 tensor → 垃圾。
- 但**先排除第 0 步**(eager 是不是也 `!!!!`)：若 eager 也坏，与此无关。
- 真要修：kernel 侧 schema 加 `Tensor(a!)`(需重编 kernel)，或 vllm 侧用 `direct_register_custom_op`
  + `mutates_args=[...]` 包一层正确声明 mutation 的 wrapper(`infer_schema` 生成 `(a!)`)。

### D. capture 成功但输出 `!!!!`(static buffer realloc — graph 固化地址失效)⭐高频
- **决定性诊断信号：必须"先跑大 batch、再回小 batch"才坏；纯小 batch / 服务刚起就跑小 batch 不坏。**
  在线表现：先跑高并发(gsm8k 100 题 asyncio 齐发)，再发单条请求 → `!!!!`；单独发单条请求一直正常。
  这个"被大 batch 污染"的时序特征是 D 类与 C 类(一上来就坏)的关键区分。
- **⚠️ 别被"只有 batch=1 坏"误导——受害面是 `[1, max_size]` 整个区间，不止 batch=1，也不止离散的
  capture_sizes。** 实测(污染后逐一探测)：capture=[1,2,4,8] 时 batch=1/2/4/8 全坏。原因见
  `cudagraph_dispatcher.py`：`num_tokens <= max_size`(=最大 capture size)的请求**都走 graph replay**
  ——正好命中 capture size 的直接 replay，没命中的(如 3/5/6/7)**向上 pad 到最近的 capture size** 后
  replay(仍读固化地址)；只有 `num_tokens > max_size` 才 fallback eager。所以"看似只有 batch=1 坏"只是
  因为你只用单请求(xinge)去测；gsm8k 大 batch(>max_size)走 eager 反而不坏(但它正是触发 realloc 的元凶)。
- **⚠️ 加大 cudagraph_capture_sizes 不能修，反而扩大受害面**：抬高 `max_size` 等于把 `[1,新max_size]`
  更大的区间全拉进"走 graph → 会中招"的范围。病根是 realloc 使固化地址失效，与 capture 了哪些 size 无关。
  唯一正解在"消除 realloc / 让地址稳定"这一层(见下"修复")。
- **根因**：ESIMD kernel 里的 `static thread_local torch::Tensor s_xxx`(跨调用持久 buffer)，其
  `ensure_*` 在 `n_tokens` 变化时 `torch::empty` **重新分配**(经典写法 `if (s_cached_ntokens != n_tokens)`)，
  且函数 `return s_final_output`(直接返回这个持久张量)。capture(batch∈capture_sizes，如≤8)时把
  "MoE 输出在地址 A"**固化进 replay**；大 batch(>capture 上限，走 eager fallback)运行时 `ensure` 把
  buffer realloc 到新地址、**旧地址 A 引用归零被释放**；之后小 batch 走 graph replay 仍指向已失效的 A
  → 读到被复用的垃圾显存 → logits 全乱 → 采样出 `!` → `!!!!`。
- **为什么只有 graph 坏、eager 不坏**：eager 每个 step 都重跑 host 端 `ensure_*`，kernel 总是用最新地址，
  realloc 无所谓；graph replay **不重跑 host 逻辑**，按 capture 固化的地址跑 → 地址必须 capture 后永久稳定。
  即"kernel 假设每次调用都重读地址"(eager 成立)与"graph 要求地址固化后不变"的契约冲突。
- **定位手法**：① 先 env-gate 二分锁定到某 kernel 大类(本例 `DISABLE_ESIMD_MOE=1` 一关就好，而所有
  细分 gate `DISABLE_ESIMD_MOE_GELU/_NORM/_BATCH_GROUPED/MOE_DECODE_GEMV/...` 都关不掉
  → 说明肇事路径只被总开关覆盖)；② 顺着模型 forward 找到该 kernel 的 C++ 源，
  `grep -n "static thread_local\|s_cached_ntokens\|torch::empty\|return s_"`，看到
  `if (cached != n_tokens) { ... torch::empty ... } return s_final_output;` 即坐实。
- **修复 = 让 buffer 地址永久稳定**："只增不重分配"——判定改 `if (s_cached_ntokens < n_tokens)`，
  分配尺寸用 `an = max(s_cached_ntokens, n_tokens)`，`return s_final_output.narrow(0,0,n_tokens)`(零拷贝切片)。
  加 env 开关(如 `MOE_STICKY_BUF=0` 回退原行为)便于 A/B。**这条 esimd MoE 路径通常 gate 在
  `num_tokens<=128`**(更大走 upstream FusedMoE)，所以 buffer 上限就是 128，**额外显存 ~10MB/卡级别(可忽略)，
  且 static 是文件级、全层共用一组不乘层数**；性能微正(省 realloc)。
- **现成旁证**：同 `moe.sycl` 里 gemma4 的 `moe_forward_full_gelu_tanh` 早因 enable graph 踩过同坑，已用
  `if (sg_cached_ntokens >= n_tokens) return;`(只增) + `SG_FINAL_RING=4` 输出 ring 修好(注释明说
  "callers hold the returned tensor... we cannot overwrite it")。**新 enable 的模型若复用 qwen3_next 的
  v1/v2 路径(`moe_forward_full`/`_v2`，用 `s_*`/`s2_*` + `!=` 重分配)就会复现**——对照 gemma4 改法移植即可。
- ⚠️ 写 env 开关的 C++ 字符判定别踩 shell/py 转义坑：经 `python3 -c`/heredoc 注入 `e[0]=='0'` 时单引号
  常被吃成 `e[0]==0`(与 NUL 比，永远 false，开关静默失效)。**用 ASCII 数值 `e[0]==48` 最稳**，或落地后
  `grep` 一眼确认源码里引号还在。开关失效的征兆：`MOE_STICKY_BUF=0` 本该复现却不复现。

### 离线复现器(D 类必备，比 online 快且能一键 A/B)
关键：要让 decode batch **递减穿过 capture_sizes**——不能用等长 prompt(batch 恒定，要么全程 eager
fallback 要么不触发 realloc)。给每条序列**不同 `max_tokens` 让它们陆续结束**，batch 自然从大降到小：
```python
# 阶段1 大 batch：长度各异 -> batch 逐步穿过 8/4/2/1
sps=[SamplingParams(max_tokens=8+i*11, temperature=0, ignore_eos=True) for i in range(BIG)]
llm.generate([{"prompt":P}]*BIG, sps)
# 阶段2 单条 -> 看是否 !!!! (bad = text.count("!")>20)
```
`LLM(..., enforce_eager=False, compilation_config=CompilationConfig(mode=0,
cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1,2,4,8]))` 开 graph；
`EAGER=1` 走 `enforce_eager=True` 做对照。env 带齐 `VLLM_XPU_ENABLE_XPU_GRAPH=1` + 启动那套
(`ZE_AFFINITY_MASK/TORCH_LLM_ALLREDUCE/CCL_ZE_IPC_EXCHANGE/VLLM_WORKER_MULTIPROC_METHOD/VLLM_MLA_DISABLE`)。

## 验证 capture 真生效
日志出现 `Capturing CUDA graphs (decode, FULL): N/N` + `Graph capturing finished` = 成功。
若见 `Skipping CUDA graph capture` 或 `Overriding to NONE` = 被降级/跳过，回到「三种 mode」「多卡三处必改」排查。
确认提速：同模型同请求 graph vs `--enforce-eager` 测 decode TPOT(小 batch 最明显)。

## varlen 的 graph-safe 快捷路径(§A 的另一条出路)
§A 说 prefill/varlen 用 `work_group_scratch` 导致 capture 崩。但对 **decode(query_len=1)** 有捷径:
- varlen(cutlass-sycl，`flash_api.cpp`)的 `work_group_scratch` **大小只依赖 split-K 策略、不依赖 seqlen**;
  decode 通常 `num_splits=1` → scratch=0。
- **Quick-check**:强制 `get_num_splits()` 返回 1，scratch 消失，graph 可能直接过(先验证再投入)。
- **彻底修**:用 `local_accessor` 替代 `work_group_scratch`(改 `paged_decode.hpp`+`chunk_prefill.hpp`，实战 commit `c7a3a06`/`33254e8`/`13642f0`，工作量 2-3 天)。
- ⚠️ **这个修复是 per-容器 per-checkout 的**:某容器改好不会自动同步到另一容器 → "昨天对今天错/A 容器能跑 B 崩"时按 `xpu-vllm-env` 的跨容器 .so 同步清单排查(对齐 .so 时间戳/rpath/源码 grep `work_group_scratch_size`/`git log --all`)。

## GDN / linear-attention × torch.compile 的确定性错误(不是 graph capture 问题)
- **Qwen3.5/GDN(linear-attention)在 torch.compile/cudagraph 下数值确定性错误输出 `!!!!`**：2×2 真值表验证 eager✅ / compile❌，且**与 weight-clone workaround 无关**。
  根因疑在 GDN 自定义 XE2 kernel 未正确注册进 graph、或 cudagraph 捕获了 stale mamba/ssm state。
- 伴随症状:GDN forward 里改 `weight.data` 触发 `Inference tensors do not track version counter`(要移出 forward 到 load 后，但即便移出 compile 仍错)。
- ⚠️ 另一个 GDN 隐患:XE2 `chunk_gated_delta_rule`/`gdn_attention` kernel 有 **OOB 写**(decode 行 chunk_size=1 写满 64 行 tile 越界)，会污染相邻显存/FP8 权重;b8.3 用首次 forward clone 权重 workaround，eager 有时可绕、compile 下无解。
- 结论:遇 GDN/hybrid 模型上 graph/compile 报 `!!!!`，先按第 0 步确认 eager 是对的，这类属 kernel×compile 兼容性问题，非 capture/mutation。

## 踩坑速记
- 杀 vllm 必须连 `spawn_main` 子进程按 PID 杀(否则孤儿 worker 占卡)；重启前清
  `/dev/shm/psm_* /dev/shm/sem.loky-*`(残留致下次 `BrokenPipeError`/`KeyError /psm_*`)。
- 启动用 `docker exec -d ... setsid`；JSON compilation-config 在 bash 里要单引号整体包，否则花括号被截断。
