---
name: xpu-vllm-static-feasibility
description: 不动(或极少动)硬件，纯静态读代码判断「某模型/架构/upstream PR/feature 能否在 Intel XPU vllm-xpu 上跑或移植」，以及两个版本间的 feature 差异。覆盖：第一步永远先确认「哪份代码真正生效」(namespace package 陷阱、editable 解析、git cat-file 判断当前 checkout vs BKM/upstream 是否同一 commit)、判断某架构 XPU 是否支持的逐环核查路径(transformers 识别→vllm 注册→平台 backend 选择→backend 内部走 triton/编译 op/CUDA-only→.so 是否真实存在→是"没接线"还是"没 kernel")、PR/commit-message 可信度核验(回溯源码数据流、模板元编程字节数用 static_assert 逼编译器吐真值)、两版本 feature-diff 手法(文件大小/行数/env 字符串引用/目录文件存在性)。适用于用户说："vllm-xpu 0.21 支持 X 架构/DSA/MLA 吗""这个 PR 能不能移植到 XPU""0.14 和 0.19/0.21 差了哪些 feature""XPU 能跑 <model> 吗"(不要求先跑起来)。前提：能读到容器/仓库里生效的 vllm-xpu 源码 + vllm-xpu-kernels 源码；结论是"能/不能/缺什么"的判断，不是跑通。
---

# vllm-xpu 静态可行性分析

**不动硬件、逐链读源码,判断某模型/架构/PR 能否在 Intel XPU 上跑或移植、两版本差了什么 feature。**
提炼自:GLM-5.2(DSA+MLA)/DeepSeek-V4 XPU 可行性核查、vllm-xpu-kernels PR#1 commit 核验、gemma4-31b←qwen3.6 迁移面分析、0.14 vs 0.19/0.21 feature diff。
输出是「能/不能/缺什么/缺口在哪一层」的判断,不追求跑通(要跑通走 `xpu-vllm-enable-new-model`)。

## ⭐ 第一步永远是:确认「哪份代码真正生效」(不然结论全错)

这是本方法最高频的翻车点——读了没生效的代码得出错误结论。动脑之前先钉死生效代码:

- **`vllm` 是 namespace package,`vllm.__file__` 常为 `None`**。不能靠它定位。
- 用 `import <具体子模块>; print(mod.__file__)` 看 editable 实际解析到哪个目录。
  同名多目录(如 `/workspace/vllm` vs `/workspace/enable-diffusion-models`)会被合并成一个 `vllm` 包,
  editable 装的是哪个决定你读的是不是生效代码。实战:849dd534 前半段因先读了未生效的旧目录,结论全被推翻后重来。
- `cd /workspace` 下跑 python 会被 `sys.path[0]=''` 劫持到 cwd 里的旧 checkout → 在 `/root` 或非仓库目录跑。
- 确认分支/commit:`git -C <生效目录> branch --show-current` + `git log -1`。
- **判断"当前 checkout 与 BKM/upstream 用的是不是同一份代码"**:`git cat-file -t <BKM 提到的 commit>`。
  若返回 `Not a valid object` → 你的 checkout **根本没有那个 commit**,BKM 结论不能直接套(实战:DeepSeek-V4 Flash BKM 用 upstream@`eebce65`,本容器 downstream 无此 object → 本容器跑不了,但不代表 XPU 跑不了)。

## 判断「某架构 XPU 是否支持」的逐环核查路径

按数据流从上到下逐环查,任一环断了就是缺口。以 GLM-5.2(DSA+MLA)为例:

1. **上游 transformers 是否识别 `model_type`**:如 5.8.0 已识别 `glm_moe_dsa`。不识别→连 config 都加载不了,先 vendoring(见 enable skill)。
2. **vllm 模型类是否注册** + **config 哪个字段自动触发特殊路径**:如 `GlmMoeDsaForCausalLM` 已注册;`is_v32 = hasattr(config, "index_topk")` 自动触发 sparse。
3. **平台层 backend 选择**(`vllm/platforms/xpu.py`):看 `use_sparse`/`use_mla` 等分支把该模型路由到哪个 backend(如 `use_sparse→XPU_MLA_SPARSE`、`use_mla→TRITON_MLA`)。
4. **backend impl 内部调的是 triton 还是编译 op**:读进去看它 import 什么。
   - triton kernel(如 `triton_bf16_mla_sparse_interface`,硬编 `BLOCK_DMODEL=512/BLOCK_DPE=64`、`assert dim_qk==576`)→ 检查维度是否匹配目标 config(GLM-5.2 的 512+64=576 匹配)。
   - 编译 op(`torch.ops._xpu_C.*` / `_vllm_fa2_C.*`)→ 进第 5 步。
   - **CUDA-only 库**(`has_deep_gemm()`、`if current_platform.is_cuda(): import flashmla`、`assert "Only FlashMLA ... supported"`)→ **硬缺口**,XPU 上 `_missing()` 报错。
5. **编译 op 是否真实存在**:查 `.so`(`_xpu_C.abi3.so`/`_vllm_fa2_C`/`libmqa_logits_kernels_xe_2.so`)、`register_ops_once()`/`direct_register_custom_op` 注册、
   以及 **`vllm-xpu-kernels` 的 `tests/` 目录反推它到底提供了哪些原生 kernel**(如 `tests/flash_attn/test_mla_decode.py` 证明有 dense MLA 的 FA2-varlen)。
6. **区分两种缺口**(决定工作量):
   - **kernel 物理存在但没接线**(如 `_xpu_C::fp8_mqa_logits`/`libmqa_logits_kernels_xe_2.so` 已编译,但生效分支统一调了 XPU 没有的 `deep_gemm` → rebase upstream 时丢了 XPU 分支的 regression)→ **补接线即可**(旧分支就是接 `torch.ops.vllm.xpu_fp8_mqa_logits` 的)。
   - **kernel 根本不存在**(如 DSA sparse MLA 主注意力,GLM 路径只有 triton、V4 路径 CUDA-only,FA2-varlen 只支持 dense 不支持 topk 稀疏)→ **要新写 ESIMD/SYCL kernel**,工作量大。

## 常见结论形态

- 架构骨架/注册往往齐全,真缺口集中在「某算子只有 triton / 只有 CUDA / rebase upstream 时丢了 XPU 分支」。
- 同一功能不同模型走不同代码路径:GLM-5.2 复用 `deepseek_v2.py` 走 triton;DeepSeek-V4 有独立 `deepseek_v4.py` attention 硬绑 CUDA FlashMLA → 同样"支持 MLA"但一个能改一个不能。
- **可以用「一处实测钉死判断」但不做全量运行**:如只在空闲卡上单跑一个 triton MLA kernel 验证维度匹配,其余全静态。用完清理 triton 编译子进程/共享内存(见 `xpu-vllm-env`)。

## PR / commit-message 可信度核验

**不信 commit message 的公式/声明,回源码验证数据流链。**

- 实战:PR#1 `db68110` 声称"cap V-tile 到 256 修单卡崩"——回源码验证链:`cap tile → DecodeTileScheduler 的 grid.x=ceil_div(head_size_vo,V_tile) 自动变 2 → 两 WG 写不相交输出切片 → V 是输出特征轴无跨-WG 规约 → 数值精确`。链成立才采信。
- **模板元编程的字节数别手推**:commit message 的 SLM 公式 `16×512×4×4=128KiB` 把 `SGPerWG` 因子重复乘了一遍(epilogue buffer 已含该因子),且 `SharedStorage` 是 `union<mainloop,epilogue>` 取 max。
  → "精确 cap SLM" 必须用 `static_assert(sizeof(SharedStorage)==0, ...)` 逼编译器吐真实字节数再回填,别照搬公式。
- 留意**隐藏副作用**:如 `has_decode = seq_lens_q.eq(1).any().item<bool>()` 的 `.item()` 引入一次 D2H 同步(仅在 prefill 分支),可改用 device 侧规约。
- 区分真伪 bug:`0b56422`(chunk_prefill 滑窗 mask 因 `seq_is_causal=false` 致 `eff_right` 错取左窗宽 → 能看未来 token)是真正确性 bug;有些"修复"其实是接口适配不是 NaN 源。

## 两版本 feature-diff 手法

不逐行 diff,用廉价信号快速圈定:

- **文件大小/行数对比**:被优化大幅扩展的文件即主战场。实战:`qwen3_5.py`(0.14.0 75KB/1800 行 vs 0.21.0 上游 29KB/811 行)、`qwen3_next.py`(113KB vs 30KB)——说明 **0.14 的 esimd 投影优化根本没移植到 0.21**(0.21 grep `esimd_` 仅 1 处命中)。
- **grep 特定 env/op 字符串是否还有真实现**:如 0.19 里 `sym_int4`/`auto_round`/GPTQ-Marlin XPU 路径**只留字符串引用无实现**;`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` 在某分支 0 处引用(设了无效)。
- **目录/文件存在性**:如 `vllm/v1/kv_offload/worker/cpu_xpu.py` 在 0.19 不存在 → KV offload 缺失。
- **git 圈定**:`git log --oneline <base>..<fork-head> -- <file>` 找被哪些 PR 改过,再对 fork 的上游 base 算完整改动集。

## 实战结论沉淀(可当参考事实)

- **GLM-5.2/DSA/MLA XPU**:骨架齐全(transformers 5.8.0 识别 `glm_moe_dsa`、`GlmMoeDsaForCausalLM` 已注册、`XPU_MLA_SPARSE` triton kernel 实测卡6跑通、dim_qk=576 匹配);
  两处缺口——① DSA indexer logits 在生效分支断(走 XPU 无的 DeepGEMM,但 `_xpu_C::fp8_mqa_logits`/`libmqa_logits_kernels_xe_2.so` 已编译存在,补接线即可);② DSA sparse MLA 主注意力仅 triton,无原生高性能 kernel。
  **BKM 事实**:upstream vllm@`eebce65` + vllm-xpu-kernels 已在 B70×8 跑通 DeepSeek-V4-Flash(fp8 KV/TP=8/enforce-eager/过 gsm8k),路径=indexer→topk→gather→FA2 varlen(非 triton);本容器 downstream 分支 V4 仍 assert CUDA-only FlashMLA、跑不了。
- **0.14 vs 0.19/0.21**:0.19 相对 b8.3 缺 Streaming FP8/INT4 加载(不补则多卡大模型 OOM)、KV offload、Cutlass FA 默认、sym_int4/auto_round/GPTQ-Marlin XPU 路径;0.21 的 qwen3_5 esimd 投影优化未从 0.14 移植。
- **异构 prefill/decode 分工判断**(sglang 调研):先算「算力/带宽比」定分工——高算力密度低带宽的 iGPU 适合 prefill(compute-bound),带宽充裕的 dGPU 适合 decode(bandwidth-bound);PD Disaggregation / Speculative(EAGLE draft 放 iGPU)/HiCache 是候选,但都需先验 Intel XPU/SYCL 后端能否跑通(传输后端/EAGLE kernel/FP8 KV)。
