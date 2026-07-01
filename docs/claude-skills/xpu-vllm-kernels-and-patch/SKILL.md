---
name: xpu-vllm-kernels-and-patch
description: Intel XPU vllm-xpu 生态里两套 kernel 仓的区分/构建/集成，以及 llm-scaler vllm.patch 的应用与跨版本迁移(0.14→0.18/0.19→0.21)。覆盖：vllm-xpu-kernels(analytics-zoo 的 pip wheel 仓，提供 _xpu_C/_moe_C，cutlass grouped GEMM/gdn_attention/fp8_gemm) 与 custom-esimd-kernels-vllm(原地 rebuild，提供 esimd_*/q4_0/moe_int4) 是两套不同来源；改 git 工作目录源码不生效必须 pip install -e 或重 build；wheel 升级引入的接口回归(权重 layout [E,K,N] vs [E,N,K]、dtype 断言、KV page_size)排查；.so 的 RUNPATH 优先 build/temp 而非 site-packages 导致 copy 失效；跨容器 .so 同步；diff-of-patch ≠ diff-of-source(PR 改的是 .patch 文件不能直接 apply 到已 patch 的源码仓)；版本迁移矩阵。适用于用户说："vllm-xpu-kernels 报错/升级/装不上""改了 kernel 源码不生效""把 vllm.patch/某 PR 应用到源码仓""从 0.14 迁到 0.21 缺了什么""跨容器 .so 不一致，昨天对今天错"。前提：能进容器操作 vllm-xpu 源码仓 + 两套 kernel 仓。
---

# vllm-xpu 双 kernel 仓 & patch 运维

在 Intel XPU vllm-xpu 生态里,**分清两套 kernel 仓、正确构建/集成、以及 llm-scaler patch 的应用与版本迁移**。
提炼自:vllm-xpu-kernels wheel 升级(019_fix)、MiniCPM-V PR#472 patch 应用、Qwen3.5-35B-A3B fp8 `!!!!` 跨容器/分支排查、跨容器 work_group_scratch .so 不同步。
基础环境规约走 `xpu-vllm-env`。

## ⭐ 三套代码位置(别只记两套)

一个 XPU vllm 开发容器里通常有**三处**互相独立、更新方式各异的代码:

1. **vllm-xpu 源码仓**(editable,如 `/workspace/llm-scaler-vllm-xpu` 或 `/llm/.../llm-scaler-vllm-xpu`):改 `vllm/...` 即时生效。
   注意:它是 **纯 vllm + llm-scaler patch 应用后的产物**,仓里通常**没有** patch 文件本身。
2. **custom-esimd-kernels-vllm**(如 `/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm`):**原地 `setup_*.py build_ext --inplace` 重编**。
   提供 `custom_esimd_kernels_vllm.*`:`esimd_*`(GEMV/norm/page_attn)、`q4_0_quant_ops`、`moe_int4_prefill_ops`。
3. **vllm-xpu-kernels**(analytics-zoo 那套,如 `/llm/vllm-xpu-kernels`):**pip wheel 仓,`pip install` 才生效**。
   提供 `_xpu_C.*`(`gdn_attention`、`fp8_gemm_w8a16`、`cutlass_grouped_gemm_interface`、`fp8_mqa_logits`)、`_moe_C.*`(`remap_hidden_states`、`moe_gather`)。

⭐**最高频坑:改 vllm-xpu-kernels 的 git 工作目录源码不会生效**——运行时用的是 pip 装的 `vllm_xpu_kernels==x.y.z` wheel。
必须 `pip install -e /llm/vllm-xpu-kernels` 或重 build wheel。生效佐证:升级后 libattn 体积变化(如 550MB→1.4GB)。

## vllm-xpu-kernels wheel 升级的接口回归排查

wheel 版本跳变(如 v0.1.4→0.1.8.dev)最常见的是**接口/layout 回归**,不是 NaN:

- **形状断言**(`RuntimeError: ptr_A.size(1) must match ptr_B.size(1)`):去读 wheel 内 `fused_moe_interface.py` 的 `inter_size = w13.shape[-1]//2` 判断它**实际期待**的 layout。
  实测:fp8 cutlass grouped GEMM 期待 w13 为 `[E, K, N]`,而 mainline `Fp8OnlineMoEMethod` 存 `[E, N, K]`=`[E, 2*inter, hidden]` → 崩。
  修法:在 `process_weights_after_loading` 的 XPU 路径把权重转成 `[E,K,N]`(commit `e0ac35670`/`18dde2342`);ESIMD `moe_forward_full` 文档也已统一为 `[E,K,N]`。
  ⚠️ **docstring 可能过期与代码矛盾,以 `inter_size` 的实际算法为准**(fp8/其它=`[E,K,N]`,4bit/mxfp4 可能=`[E,N,K]`)。
- **新 kernel dtype 断言**:如 `_xpu_C.gdn_attention` 要求 `A_log` 为 float32 → 调用前 `A_log = self.A_log.float()`(commit `1fe8a3e9b`)。
- **KV cache page_size 校验**(`page_size_padded >= page_size`):hybrid(linear+full attn)模型 ssm dtype 用 fp16 时 `attn_block_size` 算错未 round 到 2048 → dtype 修正后正确。
- 升级后用 `import` 逐个验证依赖矩阵:`torch.ops._xpu_C.*` / `custom_esimd_kernels_vllm.*` 是否注册(用 `torch._C._jit_get_schemas_for_operator("ns::op")`,别信 `dir(torch.ops.ns)`)。

## ⭐ .so 加载 / RUNPATH / 跨容器同步(copy 失效的元凶)

- **`_vllm_fa2_C` / `_xpu_C` 的 RUNPATH 优先从 `build/temp` 加载,不是 site-packages**。
  → copy/替换 `.so` **必须两处都更新**,否则 build/temp 里的旧版静默优先生效。
  实战翻车:从 gc 的 site-packages copy 了无效版 libattn 覆盖 build/temp,导致 `varlen_fwd` 注册失败。
- **跨容器 copy 依赖前先 md5 对比**:site-packages vs build/temp 可能不一致;先确认目标容器"真正加载"的是哪份;覆盖前带时间戳完整备份(含 build/temp)+ 写 README_RESTORE。
  实战:021-1 与 gc 真正用的 libattn 本是同一份,只有 `_xpu_C.abi3.so` 和 `libgdn_attn_kernels_xe_2.so` 真不同 → 只需同步这两个。
- ⭐**kernel 修复是 per-容器 per-checkout 的**:某容器改好的 kernel(如 `local_accessor` 替 `work_group_scratch`)**不会自动同步到另一容器**。
  "昨天对今天错" / "A 容器能跑 B 容器崩" 的排查清单:对齐「site-packages 实际加载的 .so 时间戳 + rpath 指向 + 源码 grep 关键符号(如 `work_group_scratch_size`) + `git log --all --grep <commit>`」。
  别人可能用无改动源码重编 .so 覆盖了你的修复。跨容器必须显式同步 .so 或在该容器重编。

## ⭐ diff-of-patch ≠ diff-of-source(应用 PR 到已 patch 的源码仓)

llm-scaler 的 `vllm/patches/vllm_for_multi_arc.patch` 是一个大 patch 文件。dockerfile 真实流程:
`git clone -b v0.14.0 vllm → git apply vllm_for_multi_arc.patch`。
而容器里的源码仓(如 `llm-scaler-vllm-xpu` v0.14.0-b8.3 分支)是 **patch 已 apply 后的产物,里面没有 patch 文件**。

关键:一个新 PR(如 MiniCPM-V #472)的 diff 可能是 **"改 patch 文件" 的 diff(diff-of-patch)**,**不能直接 `git apply` 到源码仓**。
等价关系:`纯v0.14.0 + 旧patch = 现源码仓`;`纯v0.14.0 + (旧patch + PR) = 目标`。
所以要 merge 进源码仓的增量 = **「新 patch 应用结果 − 旧 patch 应用结果」**。做法:

1. 把 PR apply 到 patch 文件生成新 patch(`git apply --check` 验证;行数会涨,如 19374→20879)。
2. 临时 clone 纯 v0.14.0,分别 apply 旧/新 patch,`diff` 出**源码层增量**(如 6 文件 +1425 行,含新文件)。
3. 在源码仓建新分支 apply 该源码增量并 commit。
这套流程与 dockerfile 一致、不污染任何东西。

## 版本迁移矩阵(0.14 → 0.18/0.19 → 0.21)

跨版本移植已有优化,同一功能的位置/实现会漂移:

- **config 类**从 transformers fork 搬到 `vllm/transformers_utils/configs/`。
- **`Fp8OnlineLinearMethod`** 上游化后自带 streaming(meta-device 逐层量化);0.14 自写的 CPU offload 路径(`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT`)被重构掉——在新线该 env 可能 0 处引用(设了无效)。
- **`sym_int4`** 被 `ipex_quant.py` 的 GPTQ-Marlin 路径取代;老的 `sym_int4.py`/相关 cherry-pick 在新分支无法落地。
- API 签名变化:`self.positions`/`compute_slot_mapping`/`_make_buffer`/gpu_model_runner 接口变了 → 老 cherry-pick 崩。
- **cherry-pick 陷阱**:跨基线 cherry-pick 会拖入不属于此 fix 的旁支;宜**手工精确 port 语义**(如只加 `.clone().contiguous()`、只把 env 默认 `"0"→"1"`)。
- 迁移前先核对:env var 是否还被读、config import 路径、gpu_model_runner API。

## 差集核对方法(fork A 分支 → fork B 分支)

- 大多数 A 分支的 commit 在 B 上"已存在/被更新代码取代"(如 `moe_forward_full/_v2` 取代旧 `moe_forward_fused`/topk_v2;带 padding 的 page_attn 取代旧版)——**别盲目全 cherry-pick**。
- 真正需手工 port 的往往是少数语义修复:如 RMSNorm cache corruption(8 处加 `.clone().contiguous()`,commit `1a34ded7e`)、ESIMD 默认开(9 个 env `"0"→"1"`,commit `121384e38`)。
- **`!!!!` 类回归常是"丢了 workaround"**:实战 Qwen3.5-35B fp8 `!!!!` 根因两点——(1) 0.19 cherry-pick 时丢了 b8.3 在 `GatedDeltaNet.forward_xpu` 入口的 `_weights_cloned` workaround(`gdn_attention` XE2 chunk kernel **有 OOB 写**会污染 FP8 权重,首次 forward clone `out_proj/in_proj_*` 避开);(2) 用 `hasattr(self.in_proj_qkvz,'weight_scale')` 判 ESIMD 路径,但 `weight_scale` 要 `process_weights_after_loading` 才有,`__init__` 阶段必 False → ESIMD 路径全被禁。诊断:逐层 trace + 与好分支对比。

## 交付规约

- 升级/迁移时**新建分支只带最小必需 commit**(如只带 wheel 升级必需的 3 个:dtype float32 / layout `[E,K,N]` / API 适配),性能优化暂缓,待正确性验证后分批加回。
- commit 在容器内做(容器 git config,不 amend);push 前问用户。诊断文档落 `/llm/models/test/*/debug-*.md`。

## 实战索引

- vllm-xpu-kernels 升级 019_fix:分支 `upgrade-v0.19.0-fix`,最小 commit `f319cf461`/`18dde2342`/`87e2a4022`;诊断 `/llm/models/test/fix_019/debug-35B-A3B-fp8-nan.md`。
- MiniCPM-V 4.6 PR#472 patch 应用:commit `4f331f318`(容器 wj-test-new-b8.3.1)。
- 跨容器 work_group_scratch:`local_accessor` 修复 commit `c7a3a06`/`33254e8`/`13642f0`(只在 021-gc 做过,未同步 0630)。
