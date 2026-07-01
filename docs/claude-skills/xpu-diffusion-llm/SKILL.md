---
name: xpu-diffusion-llm
description: 在 vLLM(尤其 Intel XPU vllm-xpu)上 enable/推理/profile/优化扩散式 LLM(diffusion decoding，非自回归，如 DiffusionGemma)。覆盖：扩散范式与自回归的本质区别(复用 spec-decode draft 通路、单 backbone 切 encoder(causal 写 KV)/decoder(双向读 KV, M=canvas) 两 mode、每 denoise step 整个 canvas 并行 forward 按 entropy 逐位置部分 accept+renoise、收敛才整块 emit)；⭐性能测法陷阱(不强制长输出会得出"慢 14x"的伪结论，metric 应看 token/forward 而非纯 tok/s，4k prompt≠4k output)；扩散独有的坑(sampler 巨型 logits 瞬态 OOM、TP=4 撞 Level Zero 资源上限只能 TP=2、为自回归 shape[0]==1 写的 ESIMD 融合对 canvas(M=256) 全不触发)；瓶颈画像(MoE~64%)与优化落点判断(commit:denoise 步数比决定 commit 类优化不值得，减 denoise 步数才是最大杠杆)。适用于用户说："enable/优化 diffusion gemma/扩散模型""扩散 LLM 在 XPU 上慢/怎么测性能""diffusion 输出很短/tok/s 很低"。前提：可编辑 vllm-xpu 容器。enable 通用流程走 xpu-vllm-enable-new-model；写 kernel 走 vllm-xpu-esimd-optimize。
---

# 扩散式 LLM 在 vLLM/XPU 上的 enable 与优化

**扩散式(diffusion decoding)LLM 是非自回归的,推理范式和性能测法与自回归完全不同**——本 skill 专讲这些差异点,
避免用自回归(decode M==1)的思维得出错误结论。提炼自 DiffusionGemma-26B 在 vllm-xpu 上的 enable + int4/fp8 + MoE 优化实战。
enable 通用流程走 `xpu-vllm-enable-new-model`;写/接 kernel 走 `vllm-xpu-esimd-optimize`;环境走 `xpu-vllm-env`。

## 范式:它和自回归到底哪里不同

- **非自回归,复用 vLLM 的 spec-decode draft 通路**(`num_draft_tokens`/`num_speculative_tokens`)。
- **单 backbone 切两个 mode**,靠 `num_draft_tokens` 区分:
  - **encoder mode**(`num_draft_tokens==0`):causal,**写 KV**(处理 prompt)。
  - **decoder / denoise mode**(`>0`):**双向读 KV,M=canvas(如 256)**——一次 forward 整个 canvas 并行。
- **denoise 每步** = 整个 canvas 并行 forward → 按 entropy **逐位置部分 accept(eb_mask,熵低的接受)+ 部分 renoise**(熵高的重来)→ 多步迭代。
  收敛(平均熵低 + argmax 稳,或 step ≥ `max_denoising_steps` 如 48 强制)后,整块 emit `argmax_canvas[:valid_canvas_len]`。
- **无逐位置置信度兜底**:step≥48 强制收敛会 commit 高熵 token(质量固有 trade-off);`max_denoising_steps` + `entropy_bound` 是质量/速度旋钮。

## ⭐⭐ 性能测法陷阱(最容易得出错误结论的地方)

- **必须强制长输出才能测出真实吞吐**。短输出/提前收敛会得出"比自回归慢 14x"的**伪结论**。
  机制:每次 canvas(256)forward 成本恒定(实测 ~542ms/forward);短输出每 forward 只提交 ~1.5 token,长输出提交 ~11.5 token(并行解码摊开)。
  实测:短输出 4.3 tok/s vs **强制 4096 token 达 31.6 tok/s(7.3x)**——博客宣称的加速只在长输出显现。
- 做法:`SamplingParams(ignore_eos=True, max_tokens=大)`。
- **metric 用 token/forward 比**,而非纯 tok/s(tok/s 被输出长度污染)。
- ⭐**4k prompt ≠ 4k output**:并行优势只随 **output 长度**增长,**不随 context 长度**;长 prompt 反而更慢。

## 扩散独有的坑

- **sampler 巨型 logits 瞬态 OOM**:sampler 产 `[num_decode * canvas, vocab]`(如 vocab=262144)的巨型张量,
  `max_num_seqs>1` 时瞬态撞 ~1G 显存 OOM / Level Zero `UR_RESULT_ERROR_OUT_OF_RESOURCES`。
  修法:**sampler 按 decode-请求循环**(每次 `[1*canvas, vocab]`)。踩坑:循环内 `sampled.zero_()`/`num_sampled.zero_()` 会擦掉前次结果 → 必须 hoist 出循环。commit `62df89a36`。
- **TP=4 不可用**:4 进程撞 Level Zero 资源上限(error 40 `OUT_OF_RESOURCES`),与 max_num_seqs/张量尺寸无关(单卡 probe 同算子能跑)。**diffusion 只能 TP=2**。
- **为自回归 `shape[0]==1` 写的 ESIMD 融合全不触发**:gemma4.py 里 norm/router/decode-GEMV 的 ESIMD fast path 全 gate 在 `x.size(0)==1`,
  canvas(M=256)一个都不命中,走慢 fallback。**这是 diffusion 优化落点判断的关键前提**——想优化就得写 M>1 的 batch kernel。
- **float16 下可能无 XPU MoE backend** → 必须 fp8;`gpu_memory_utilization` 0.9 OOM,用 0.75。

## 瓶颈画像与优化落点

- **先做干净的分段计时**(env-gated `DG_FWD`/`DG_SAMPLER` + 层内 attn/moe `torch.xpu.synchronize()+perf_counter`)。
  实测 256-canvas forward 内:**MoE ~64%(Triton fp8 ~208ms) > attention ~30%(~98ms) > mlp ~2%**;sampler ~9%。端到端 MoE ~58%。
- ⭐**commit : denoise 步数比决定 commit 类优化不值得**:实测 commit:denoise ≈ 2:48(commit 仅占 ~4%)。
  → 「传 scalar causal / 跳过 sampler / 优化 commit 步」这类建议整体只省 <3%,别做。denoise 96% 是真双向必须全扫,不能靠改 causal flag 加速。
- **最大杠杆 = 减少 denoise 步数**(`max_denoising_steps` + `entropy_bound`,质量换速度),不是 kernel。
- **regime 判断**:diffusion 的 canvas(M=256)是 **compute-bound**(与自回归 decode 的 launch-bound/BW-bound 不同)——这里优化 kernel 才真能转化收益。
  但要先确认:`grid` 的 work-group 数 vs BMG Xe-core 数(~160)——若 2D kernel 已用 grid=256 work-group 填满,3D kernel 零收益;fp8-KV 反而慢=compute-bound 的判据。

## MoE batch(M>1)kernel 集成的定位法(diffusion 优化主战场)

diffusion MoE 全是 M>1(无 M==1),这与自回归 decode 完全不同。写 batch MoE kernel 时:

- **cutlass vs fused 的交叉点(定量根因)**:M=256 时,fused(per-route 索引)每 route 独立读自己 expert 权重 → 同一 expert 被 ~16 route 重读 16 次 → 读权重 ~2030MB(BW-bound);
  cutlass gather 后每 unique expert 权重只读 1 次做大 GEMM → ~127MB(compute-bound)。**batch=256 cutlass ~487us/层 vs fused ~3830us/层 ≈ 8x**。
  → **M 大就该走 cutlass**(权重只读 1 次);int4 因 4-bit 省带宽 + 大 batch tensor-core 利用率高,cutlass 本就快,不必开 ESIMD batch。
- ⭐**"kernel 隔离对但 wire 进模型就坏"的定位法**:若两个**完全不同**的 kernel 接进同一 M>1 路径产生**相同失败**(0/5 空输出 + 同样慢)→ 铁证 **bug 在共享集成路径(routing 布局/gather/accumulate/scale 重量化),不是 kernel**。
  隔离单测必须用**和真实模型相同的接口**(routed 变体 vs 内置 topk)+ **真实 in-model 权重/scale**(`process_fp8_weight_tensor_strategy_moe` 重量化、TP 分片 stride),否则单测 PASS 会误导。
  决定性 debug = 在集成路径同时算 Triton golden 逐环节 dump diff。用 opt-in env gate(如 `ENABLE_ESIMD_MOE_BATCH_GROUPED`)保留 kernel 资产 + 默认回退保 5/5。
- **kernel fp16 溢出**(隔离 PASS 集成崩的另一类真因):per-tensor scale 应在 fp16 累加**前**折叠进权重 tile(而非累加后,否则 K=2816 raw 部分和溢出 fp16 65504→inf);gelu_tanh 的 exp 要 clamp inner 到 ±30 防溢出。单个 NaN 经残差污染整 canvas → 0/5 空输出。
- **expert 负载长尾**:真实偏斜 routing 下(如一 expert 164 token / 44 空),grid=(E, n_tiles) 让空 expert 闲置、忙 expert 串行且权重每块重载 → 改 **m-tile 并行**(新增微 kernel `moe_build_tile_map` 按固定 8-token 块切分,与 expert 无关)。实战 up GEMM 4.58→1.86ms(-59%)。

## 攻克后瓶颈会转移到 attention(diffusion full 层)

- MoE 优化后瓶颈转 attention:**full 层单层耗时可达 sliding 层的 ~9x**(如 full 28.7ms vs sliding 3.1ms),占 attention 65%。
- diffusion full 层连 query 维度都没有(整 canvas 双向)→ **page_attn 架构上根本不适用**(它 `TORCH_CHECK(max_query_len==1)`,只算 M==1 decode)。
- full-attention kernel 级优化多是死胡同(256query×非因果×head_dim256×GQA2):3D kernel 零收益(2D grid 已填满 Xe-core);KV fp8 反而慢 28%(head_dim256 逐 tile 反量化算力 > 省的带宽,证明 compute-bound)。
- ⭐**唯一高价值 attention kernel 优化:`BLOCK_M` 16→32**(commit `1d41a9e66`)。根因:BLOCK_M=16 是给自回归 decode 调的,GQA=2 下 16 行只 8 个真实 query,`tl.dot(Q,K)` 的 M 维太瘦喂不饱矩阵引擎。
  改 `BLOCK_M = 32 if max_seqlen_q > 1 else 16`(`triton_unified_attention.py`),full 层 -40%、sliding -42%、decode 无回归(BM64/128 反而慢,work-group 占用率不足)。**这是 canvas/prefill 场景通用招**。
- ⚠️ 移植 PR 的 attention kernel 时注意 bug:softmax 后对 V 的块级滑窗裁剪若用 `qpos_lo`(块最低行),对非因果行会误杀块内更高行仍合法的 key(GQA=2→BLOCK_Q=8 触发);逐元素 S-mask 已置零 P,非因果跳过 V 裁剪即可(单测复现 0.2027→9.1e-5)。

## 实战索引

- DiffusionGemma enable + 优化:memory `project_diffusion_gemma_enable` / `project_diffusion_gemma_int4_handoff` 有完整事实链;交接 `opt_gemma4/HANDOFF_diffusion_gemma_optimize.md`。
- 关键 commit:enable `ef246a021`(mixed-causal sliding 对称化)、TRITON per_seq_causal `f950a5d03`、TRITON 默认+OOM `62df89a36`、profiling `44c008ee2`、batch MoE m-tile `ae5b3a9`/`a6fe359`、BLOCK_M `1d41a9e66`。
