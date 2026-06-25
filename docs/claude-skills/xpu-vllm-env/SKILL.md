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
   探测：`docker exec <c> bash -c "python3 -c \"import vllm,os;print(os.path.dirname(vllm.__file__))\""`
   及 `cd <vllm> && git branch --show-current`。确认是「源码可改并即时生效」还是「装在 site-packages 只读」。
3. **kernel 代码位置**：llm-scaler 的 custom-esimd-kernels checkout(常见
   `/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm`，host 可能同时挂在 `/home/.../llm-scaler`)。
   确认能否原地 `setup_*.py build_ext --inplace` 重编。也确认 `vllm_xpu_kernels`(另一套 kernel)装在哪。
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
- 单模块快编：`cd <kernel仓> && TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_<feature>_only.py build_ext --inplace`
  (有 setup_eagle_only / setup_moe_only / setup_moe_int4_only / setup_gemv_only / setup_q4_0_only 等)。
- **改 .h/.hpp(header)后必须先删对应 .o**(ninja 不追踪 header 变更)，否则改动不生效。
- 编出的 .so 要 cp 到 `site-packages/custom_esimd_kernels_vllm/` 才被 vllm 加载(除非仓本身在 path 上)。
- 验证 op 是否注册成 dispatcher：**别信 `dir(torch.ops.<ns>)`**(惰性加载列不全)，用
  `torch._C._jit_get_schemas_for_operator("<ns>::<op>")`，不抛异常=已注册。

## 任务类型 → 该加载的专项 skill
确认环境后，按用户要干的事加载：
- **写/接 ESIMD kernel 优化性能** → `vllm-xpu-esimd-optimize`(reproducer→精度门禁→profile→kernel 循环)。
- **开 XPU graph / cudagraph 提速 / graph 没生效/崩/hang** → `xpu-graph-enable`。
- **模型输出乱码/精度和 HF 对不上** → `hf-golden-layerwise-diff`(HF golden 逐层 diff)。
- enable 新模型 / 量化(fp8/sym_int4) / streaming load / benchmark：无专项 skill，按本 skill 的规约 +
  现有资产脚本(常在 `<host>/test/opt_gemma4/` 或 `/llm/models/test/`)操作。

## 常用资产位置(确认后按实际为准)
- 启动脚本：`/llm/models/test/*.sh`(gemma.sh / bench.sh / xy.sh 等)。
- 优化资产/报告/skill 备份：host `~/test/opt_gemma4/`(= 容器 `/llm/models/test/opt_gemma4/`)。
- 权重：`/llm/models/weights/<model>`。
- 量化要点：sym_int4 在 release 线需 `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`；fp8 从 fp16 权重在线量化时
  默认走 online streaming(0.21 fp8 不看 OFFLOAD 开关)。

## 记忆联动
项目级事实(模型架构/已 enable 状态/各模型 kernel 阻塞点/历史结论)多已记在用户 auto-memory
(`project_*` 系列，如 gemma4-12b-enable / xpu-graph-enable / int4-streaming-load 等)，开工前可参考。
