---
name: xpu-vllm-env
description: Intel XPU(BMG/Arc) 上 vllm-xpu + llm-scaler(ESIMD kernel) 的开发环境约定与操作规约。当用户在某个容器里做 XPU vllm 相关的活(enable 模型/量化/写 ESIMD kernel/XPU graph/profile/跑 benchmark/精度排查)且指明了容器名时，先加载本 skill 确认环境(容器名 + vllm 代码位置 + kernel 代码位置)，再按任务类型加载对应专项 skill。适用于用户说："在容器 X 里…""用 wj-test-new-*/gc-* 容器跑…""帮我在这个 XPU 容器上 enable/优化/benchmark <model>"。前提：一个装了可编辑 vllm-xpu 的 docker 容器 + 一份可原地重编的 llm-scaler custom-esimd-kernels checkout。
---

# Intel XPU vllm-xpu + llm-scaler 开发环境约定

这是在 Intel XPU(BMG/Arc B-series) 上做 vllm-xpu 开发的**环境与操作规约**(不是某个具体任务的方法论)。
目的：用户只要说「在容器 X 里干 XPU vllm 的活」，agent 据此**先确认环境、再按规约操作**，避免重复踩坑。

## 第 0 步：确认环境(每次必做，别假设)

向用户确认 / 自行探测这三样，写下来再动手：
1. **容器名**(如 `wj-test-new-021-gc`)。所有命令走 `docker exec <容器> bash -c '...'`，所有路径是容器内路径。
2. **vllm 代码位置**：可编辑的 vllm-xpu 源码目录(常见 `/workspace/<name>` 或 `/llm/.../vllm`)。
   ⚠️ **`vllm` 是 namespace package，`vllm.__file__` 常为 `None`**——别用它定位。改用
   `docker exec <c> bash -c "python3 -c \"import vllm.model_executor as m,os;print(os.path.dirname(m.__file__))\""`(具体子模块的 `__file__`)
   及 `cd <生效目录> && git branch --show-current`。确认是「源码可改并即时生效」还是「装在 site-packages 只读」。
   ⚠️ **同名多目录陷阱**：`/workspace/vllm` vs `/workspace/enable-diffusion-models` 会被合并成一个 `vllm` 包，editable 装的是哪个决定你读/改的是不是生效代码；
   且 `cd /workspace` 下跑 python 会被 `sys.path[0]=''` 劫持到 cwd 里的旧 checkout → **在 `/root` 或非仓库目录跑**。判断「当前 checkout 与 BKM/upstream 是否同一份」用 `git cat-file -t <commit>`。
3. **kernel 代码位置**：注意有**三套**互相独立的代码(更新方式各异)：
   - (a) **vllm-xpu 源码仓**(editable，改 `vllm/...` 即时生效)——它是「纯 vllm + llm-scaler patch 应用后」的产物，仓里通常没有 patch 文件本身。
   - (b) **custom-esimd-kernels**(常见 `/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm`，host 可能挂在 `/home/.../llm-scaler`)：**原地 `setup_*.py build_ext --inplace` 重编**，提供 `esimd_*`/`q4_0`/`moe_int4`。
   - (c) **vllm-xpu-kernels**(analytics-zoo 那套，常见 `/llm/vllm-xpu-kernels`)：**pip wheel 仓，`pip install -e` 或重 build wheel 才生效，改 git 工作目录源码不生效**；提供 `_xpu_C.*`/`_moe_C.*`(cutlass grouped GEMM/gdn_attention/fp8_gemm)。
   三套的详细区分/构建/patch 迁移见专项 skill `xpu-vllm-kernels-and-patch`。
- 顺带：`docker exec <c> python3 -c "import torch;print(torch.__version__, torch.xpu.get_device_name(0))"`
  记录 torch 版本 + GPU(BMG `0xe223` 等)。

## GPU / 卡占用
- **先查占用再用卡**：`pgrep -af api_server`；看进程的 `ZE_AFFINITY_MASK`(`tr '\0' '\n' </proc/<pid>/environ | grep ZE_AFFINITY`)。
  **别动用户正在跑的服务/端口**(常见 8002)，用空闲卡。
- 起服务用 `ZE_AFFINITY_MASK=<卡号,逗号分隔>` 选卡；TP=N 要给 N 张卡。
- 多卡 fp8/int4 是常态(BMG 单卡显存有限)；TP=2/4 用 `-tp=N`。

## 进程管理(踩过的坑，必守)
- **杀 vllm 必须连 spawn 子进程一起按 PID 杀**：`pkill -f` 常杀不净，`spawn_main`/`resource_tracker`/
  `multiprocessing-fork`/`Worker_TP`/`EngineCore` 会变孤儿(PPID=1)继续 99% 占卡占 CPU。
  做法：`for p in $(pgrep -f "<port>|spawn_main|Worker_TP|EngineCore|api_server"); do kill -9 $p; done`，再 `pgrep` 确认。
- **重启服务前清共享内存残留**：`rm -f /dev/shm/psm_* /dev/shm/sem.loky-*`。
  否则下个进程报 `BrokenPipeError` / `KeyError: '/psm_*'`。
- 起服务用 `docker exec -d <c> bash -c 'setsid bash <脚本> >log 2>&1'`(真后台，脱离 exec 会话)。
  **别**把启动写在带 `pkill ... <port>` 的同一条命令里(pkill 可能杀掉正在写脚本的自己)。
- `--compilation-config` 的 JSON 在 bash 里要**单引号整体包**，否则花括号被截断报 `Invalid JSON`。

## git 规约
- **提交在容器内做**(用容器的 git config，常是 hzjane)，**不要 amend**。
- 在默认/release 分支上动手前先开新分支。commit message 用 `-F <文件>` 避开 shell 转义。
- 测试脚本精简后只提交必要的；实验产物(报告/profile)放资产目录、不入版本控制。
- **push 是外发动作，必须先问用户**。host 上 `.git/refs` 可能 permission denied(之前 commit 都在容器内 root 做的)→ git 操作优先在容器内。

## kernel build(ESIMD / custom-esimd-kernels)
- 单模块快编：`cd <kernel仓> && TORCH_XPU_ARCH_LIST=bmg python3 setup_<feature>_only.py build_ext --inplace`
  (有 setup_eagle_only / setup_moe_only / setup_moe_int4_only / setup_gemv_only / setup_q4_0_only 等)。
- ⭐**编译必须 `TORCH_XPU_ARCH_LIST=bmg`**(或 `bmg-g21`)：默认 `torch.xpu.get_arch_list()` 含 XeLPG 集显(arl-h/mtl-h/lnl-m/ptl-*)，
  dpas2/BF16 在 XeLPG 不支持 → 报 "dpas2 not supported"/"BF type not allowed"。只编 BMG 绕开。
- **改 .h/.hpp(header)后必须先删对应 .o**(ninja 不追踪 header 变更)，否则改动不生效。
- 编出的 .so 要 cp 到 `site-packages/custom_esimd_kernels_vllm/` 才被 vllm 加载(除非仓本身在 path 上)。
- ⚠️ **`_vllm_fa2_C`/`_xpu_C` 的 RUNPATH 优先从 `build/temp` 加载，不是 site-packages**——替换/copy 这类 .so 必须**两处都更新**，否则 build/temp 旧版静默优先生效(实战翻过车)。
- ⚠️ **别删 `.ninja_log`**(会强制重编 oneDNN ~30min)；`libgrouped_gemm_xe_2.so` 是独立子库可单独 `ninja` 再手工 relink。
- ⚠️ **kernel 修复是 per-容器 per-checkout 的**：某容器改好的 kernel 不会自动同步到另一容器。「A 容器能跑 B 容器崩/昨天对今天错」时对齐「site-packages 实际加载的 .so 时间戳 + rpath + 源码 grep 关键符号 + `git log --all --grep`」。
- ⚠️ 部分分支的 ESIMD kernel 源文件**未被 git 跟踪**；`git clean`/`filter-branch` 前确认 tracked 状态，别清掉 untracked 源文件。
- 验证 op 是否注册成 dispatcher：**别信 `dir(torch.ops.<ns>)`**(惰性加载列不全)，用
  `torch._C._jit_get_schemas_for_operator("<ns>::<op>")`，不抛异常=已注册。

## 任务类型 → 该加载的专项 skill
确认环境后，按用户要干的事加载：
- **写/接 ESIMD kernel 优化性能** → `vllm-xpu-esimd-optimize`(reproducer→精度门禁→profile→kernel 循环)。
- **开 XPU graph / cudagraph 提速 / graph 没生效/崩/hang** → `xpu-graph-enable`。
- **模型输出乱码/精度和 HF 对不上** → `hf-golden-layerwise-diff`(HF golden 逐层 diff)。
- **从零 enable 一个新模型/新架构(移植 upstream PR、config vendoring、权重映射、多模态)** → `xpu-vllm-enable-new-model`。
- **enable 量化 / 量化后崩/乱码/DEVICE_LOST / fp8-int4 数值 bug** → `xpu-vllm-quant-enable-debug`。
- **纯静态判断某架构/PR 能否在 XPU 跑或移植 / 版本 feature diff(不要求先跑起来)** → `xpu-vllm-static-feasibility`。
- **vllm-xpu-kernels wheel 报错/升级 / 改 kernel 源码不生效 / vllm.patch 应用 / 版本迁移 / 跨容器 .so 不一致** → `xpu-vllm-kernels-and-patch`。
- **扩散式 LLM(diffusion decoding，如 DiffusionGemma)enable/测性能/优化** → `xpu-diffusion-llm`。
- benchmark / 其它：按本 skill 的规约 + 现有资产脚本(常在 `<host>/test/opt_gemma4/` 或 `/llm/models/test/`)操作。

## 常用资产位置(确认后按实际为准)
- 启动脚本：`/llm/models/test/*.sh`(gemma.sh / bench.sh / xy.sh 等)。
- 优化资产/报告/skill 备份：host `~/test/opt_gemma4/`(= 容器 `/llm/models/test/opt_gemma4/`)。
- 权重：`/llm/models/weights/<model>`。
- 量化要点：sym_int4 在 release 线需 `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`；fp8 从 fp16 权重在线量化时
  默认走 online streaming(0.21 fp8 不看 OFFLOAD 开关)。

## 记忆联动
项目级事实(模型架构/已 enable 状态/各模型 kernel 阻塞点/历史结论)多已记在用户 auto-memory
(`project_*` 系列，如 gemma4-12b-enable / xpu-graph-enable / int4-streaming-load 等)，开工前可参考。
