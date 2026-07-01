---
name: xpu-vllm-quant-enable-debug
description: 在 Intel XPU(BMG/Arc) vllm-xpu 上 enable 量化(fp8 e4m3/e5m2、sym_int4、streaming load、fp8 KV cache)并排查量化特有的数值 bug。覆盖：fp8 读写端 dtype 必须一致(XPU 默认 e5m2 但 vLLM 写 e4m3fn 的隐蔽错配)、sym_int4 packing 格式(两补码/group-wise scale)、streaming(OFFLOAD=0) vs 非 streaming、MoE TopK 硬编码(E,top_k) 需补分支、以及量化最典型的数值 bug 指纹——"prefill 在某 token 数(128/129/256)处崩/输出错"、"streaming 崩非 streaming 不崩"(内存邻接越界)、"合成 oracle 通过但 e2e 乱码"(必须 gsm8k 兜底)。适用于用户说："给 <model> enable fp8/int4/fp8-kv-cache""量化后输出乱码/崩/DEVICE_LOST""int4 prefill 到某长度就 error""e5m2 支持吗"。前提：可编辑 vllm-xpu 容器 + 可原地重编的 kernel checkout。写/改 kernel 的通用循环走 vllm-xpu-esimd-optimize;本 skill 专注量化本身的 enable 与数值正确性。
---

# vllm-xpu 量化 enable + 数值 debug

在 Intel XPU 上 enable 量化(fp8 / sym_int4 / streaming / fp8 KV)并排查其**特有数值 bug** 的方法论。
提炼自:Qwen3.5-35B-A3B int4 `DEVICE_LOST`、fp8 KV cache ESIMD enable、Qwen3-30B-A3B int4 MoE TopK E=128、DiffusionGemma int4。
环境/编译/占卡规约走 `xpu-vllm-env`;写 kernel 循环走 `vllm-xpu-esimd-optimize`。

## 量化路径速查(先搞清走哪条)

- **fp8(权重)**:vllm 内置在线量化(加载 bf16/fp16 时量化 Linear/MoE experts,`lm_head`/`embed` 保持高精度)。0.21 线默认走 online streaming,**不看 `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT`**。
- **fp8 KV cache**:`--kv_cache_dtype fp8`。bare `"fp8"` 在 vLLM 主线 = **e4m3fn**(`torch_utils.py` 的 `"fp8":"fp8_e4m3"`)。要传 `k_scale`/`v_scale`。
- **sym_int4**:release 线需 `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`(legacy)或走 streaming(=0)。packing 见下。
- **streaming load**:`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0`——meta-device 逐层量化再释放 bf16,省 CPU 内存(122B 4 卡不 OOM 的关键)。在 0.19+ 上游 `Fp8OnlineLinearMethod` 已实现,老 fork 的自写 offload 路径已被重构掉(部分分支该 env 0 处引用=设了无效,见 `xpu-vllm-kernels-and-patch` 版本迁移)。

## fp8 dtype 一致性(最隐蔽的坑,fp8 KV cache 必看)

⭐ **写入端和读取端的 fp8 dtype 必须一致,否则字节被按错格式解码,全错**:
- XPU 平台 `current_platform.fp8_dtype()` 默认返回 **e5m2**。
- 但 vLLM 写入端(`reshape_and_cache_flash` / attention layer `assert kv_cache_dtype in {"fp8","fp8_e4m3"}`)对 bare `"fp8"` 用 **e4m3fn**。
- 若读取端误用 `current_platform.fp8_dtype()`(→e5m2)去解 e4m3 字节 → 全错。**正确**:读取端用 `get_fp8_dtype_for_flashattn`(→e4m3fn)。
- **诊断会被彻底骗过**:若 trace 的 ref 和 dump 也用同一个(错的)view,ESIMD/ref/fp16-replay 三者「一致地错」→ diff≈0、0 BAD,但模型输出乱码。
  证据在启动日志:`Using KV cache scaling factor 1.0 for fp8_e4m3` 说明写入是 e4m3。
- 格式事实:**e5m2 是 fp16 高 8 位的无损子集**(反量化 `byte<<8`);**e4m3fn 需位运算 + subnormal 处理**(subnormal 当 0)。

## sym_int4 packing 格式(写 torch reference / kernel 必看)

- `implement_zp` 写出的看似 sign-magnitude s4,**实际等价 int4 两补码**(`(nibble-8)*scale`,zero-point=8)。写 torch reference 时按 sign-magnitude 解会错,按两补码 abs-diff-mean≈0。
- byte 低 nibble = K 位置 2i、高 nibble = 2i+1(little-endian)。
- GGML Q4_0 布局(diffusion int4):K-major packed `[E, N, K/8]` int32(每 int32 含 8 nibble);group-scale `[E, N, K/group_size]` fp16。
- **走 ESIMD 时别调 cutlass 专用的 `implement_zp`**;ESIMD 解 nibble 用 even/odd deinterleave(参考 `int4_GEMV.h`),解出后走和 fp8 相同的 VNNI fp16 → DPAS。
- ⭐**group_size 别在 kernel 里写死 128,要从 scale tensor shape 推导**(`group_num = scale.shape[-1]`)。否则用户 pull 改 group_size(128→32)后 scale 读错位。
  实战:DiffusionGemma int4 group=128 只 2/5;group32 达 4/5 = fp8 基线,不需混合精度(GSM8K 67%→90%)。

## MoE TopK 硬编码(E, top_k) 需补分支

ESIMD `moe_topk` / `moe_topk_softmax` 常硬编码只支持某个 `(E=256, top_k=8)`,新模型 config 不同(如 Qwen3-30B-A3B-2507 `num_experts=128`)会落 `else → TORCH_CHECK(false)` 崩。
- 底层模板(如 `MoE_TopK_V2_Kernel<128,8>`)通常本就合法(`NE%64==0 && ≤512 && TK≤16`),只需在 dispatch 处补 `(128,8)` 等分支。
- 补全部实例化点:prefill 的 `moe_prefill_int4.sycl` + decode 的 `moe_batch/moe_int4.sycl`(decode 原走慢 heap fallback,补 V2 快路径提速 2.1-3.0×)。实战 commit `30ef249`。

## ⭐ 量化特有数值 bug 的排查方法论

### 指纹一:"prefill 到某 token 数(128/129/256)就崩/错"
- 这是**强指纹**。二分到 1-token 精度(如 128 OK / 129 崩;M=256/1024 也崩 → **排除"必须 128 倍数"的对齐假设**)。
- 先怀疑 **kernel 内部对 M/tile/pitch 的硬编码假设**,或**特定 token 路由触发了某个危险 expert**(M=128 时路由凑巧没分给危险 expert,M≥129 才分到)。

### 指纹二:"streaming(OFFLOAD=0) 崩,非 streaming(=1) 不崩"
- 几乎必是**内存邻接/对齐/越界**问题(kernel 本身逻辑不背锅)。越界读命中的内容因 shard 顺序/allocator 而异:
  - streaming 下 scale tensor 紧邻其他 live tensor(常是 int4 weight 字节),被当 fp16 读出 inf/巨值 → 累加器爆炸 → 写非法地址 → `UR_RESULT_ERROR_DEVICE_LOST`。
  - OFFLOAD=1 相邻是 allocator padding,bug 潜伏不发作。
- **同 K 的另一模型不崩** 也可能只是 shard 顺序不同导致邻居 dtype 不同(Qwen3.6 vs 3.5,K 都 1408)。别被"另一个能跑"误导成"我的模型特殊"。

### 指纹三(根因样板):W2(down_proj) 比 W13 更易出问题
- 实战根因:cutlass INT4 grouped GEMM 的 scale 用 `block_2d_prefetch` 加载,row pitch = `group_num × sizeof(fp16)`。
  Xe2/BMG 的 `block_2d_load/prefetch` **硬件要求 pitch ≥ 64 字节且 16 字节对齐**。
- W2 的 `K=intermediate_size` 常非 2 的幂(如 1408 → `group_num=1408/128=11` → pitch=22 字节,违反契约);
  W13 的 `K` 常是 2 的幂(4096 → group_num=32 → pitch=64B,满足)。**所以只有 W2 崩**。
- FP8 路径完全不受影响(FP8 用 1D scale `[E]`,无 row pitch)。
- 修复:kernel 加运行时守卫 `can_prefetch_scales = (group_num*sizeof(ElementS))>=64 && %16==0`,不满足就退回普通 indexing(正确性不变,窄 pitch 只丢 prefetch hint,~1-3% 性能)。
  实战 commit:vllm-xpu-kernels `2004ffe`、llm-scaler patch `856374d`(分支 `fix_gemm_kernel`)。

### env-gate 注入实验法(按成本排序,不重编优先)
锁定量化数值 bug 时,用可逆 env 开关做减法实验,别急着改代码重编:
- `VLLM_DEBUG_GEMM_TORCH=none/w13/w2/both`:用纯 torch reference 替换 cutlass GEMM,定位是哪个权重的 GEMM 出问题。
- `VLLM_DEBUG_W2_SKIP`(zeros 占位):判 hang/崩是否归属该 GEMM。
- `VLLM_DEBUG_W2_PRESYNC`:排除 stream race。
- `VLLM_DEBUG_W2_CLONE_SCALES/WEIGHT`:clone 改 base 地址,验证"对齐/邻接"假设。
- `DG_INT4_SKIP_MOE` / `SKIP_LINEAR`:二分是 MoE 路径还是 Linear 路径(diffusion int4 乱码定位到 cutlass `XpuFusedMoe.apply()` 首次调用破坏性置空 q 权重)。
- 常见被证伪的假设:stream race、atomic buffer race、"base 地址一次性 clone 就对齐"。别在这些上耗太久,优先做 torch-reference 替换定位到具体权重。

## ⭐ 量化验证判据(最重要的教训)

- **fp16 内核/torch 当 oracle**:同一份量化字节,host 端 dequant 成 fp16 喂 fp16 内核 vs 量化内核。判据:
  - **e5m2 必须逐 bit 相同(max|d| = 0.00000)**(e5m2 是 fp16 高 8 位无损子集)。
  - **e4m3fn ≤ ~6e-4**(量化舍入)。
  - **int4 abs-diff-mean ≈ 0**(隔离 cos 可达 0.99999)。
- ⭐**合成 randn oracle 通过 ≠ e2e 正确**。这是本类 bug 的头号陷阱:
  - 越界读/dtype 错配会让「ESIMD 与 ref 一致地错」→ diff≈0 却输出乱码。
  - dpas Q stride 用 byte-index 当 element-index → 越界读邻居寄存器,但邻居也是 randn-like → 合成 oracle 抓不到,只有真实数据 e2e(gsm8k acc=0)才暴露。
  - `lsc_gather` 返回 **SoA 布局**(先 16 lane 的第 0 个 u32,再第 1 个),不是 lane 内 AoS;按 AoS 解读→ q-head 输出重复,合成 oracle 若也按 AoS 就一致地错。
  - **单测/隔离测试的权重必须贴近真机**:用小合成权重会侥幸避开满量程 fp8(|w|→288)累加溢出、避开 K 恰好对齐的边界。
- 结论:**任何量化 kernel 改动,必须 e2e gsm8k(5/5 或 100 题)兜底**,不能只信隔离单测。

## 量化编译规约(重复但关键)

- **必须 `TORCH_XPU_ARCH_LIST=bmg`**:默认 `torch.xpu.get_arch_list()` 含 XeLPG 集显(arl-h/mtl-h/lnl-m/ptl-*),int4/dpas2 GEMM 在 XeLPG 报 "dpas2 not supported"/"BF type not allowed"。
- 运行时加载的是已装 wheel(`dist-packages/`)或 site-packages 的 `.so`,不是源码目录;改后须重编覆盖 `.so`(见 `xpu-vllm-kernels-and-patch` 的 .so 加载/RUNPATH 章)。
- 别删 `.ninja_log`(会强制重编 oneDNN ~30min);`libgrouped_gemm_xe_2.so` 是独立子库,可单独 `ninja` 再手工 relink。
- ⚠️ 部分分支的 ESIMD kernel 源文件**未被 git 跟踪**;改动/清理前确认 tracked 状态,别 `git clean`/`filter-branch` 清掉 untracked 源文件(实战出过事故,靠 host 副本恢复)。

## 实战索引

- int4 W2 scale prefetch OOB → DEVICE_LOST:分支 `fix_gemm_kernel`,commit `2004ffe`(kernel)/`856374d`(patch);诊断文档见容器内交接。
- fp8 KV cache ESIMD:分支 `enable_fp8_kv_cache_dpas_debug`(llm-scaler)/`enable_fp8_kv_cache`(vllm-xpu),`README.fp8_kv.md` 记 6 个坑;关键 commit `e756ef8`/`40dab68`。性能 fp8/fp16:短 seq 100-102%、中段 105-117%、64k+ HBM-bound 时 fp8 反超(52-62% 相对时间)。
- MoE int4 TopK E=128:commit `30ef249`(分支 `fix-moe-int4-topk-e128`)。
- DiffusionGemma int4:group32 达 4/5,commit `d076f69`/`3923a7fca`/`2480a8c`(group_size 从 shape 推导)/`0efd068`(gelu_tanh)。
