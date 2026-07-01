---
name: xpu-vllm-enable-new-model
description: 在 Intel XPU(BMG/Arc) 的 vllm-xpu(downstream fork, release/v0.21.0 线) 上从零 enable 一个新模型/新架构(尤其 upstream 刚合入的 PR)。覆盖：移植方式决策(cherry-pick 为何必失败→改用 git apply GitHub .patch)、config/processor 手工 vendoring(绕开容器 transformers 版本缺失)、模型 registry 注册、load_weights 权重名映射、"先验证成败命门(如某 attention 能力)再动手"的最小 reproducer 数值验证、跑通顺序(import→构造→权重→warmup→首个 forward→生成)的各阶段专属崩点、多模态 vision tower 处理、以及一份"upstream vs 容器分叉"高频适配 bug 清单。适用于用户说："在 vllm-xpu 上 enable <某 upstream 模型/PR>""把 <model> 移植到 XPU 跑起来""这个新架构 XPU 能不能加上"。前提：一个装了可编辑 vllm-xpu 的容器 + 目标模型权重 + 该模型的 upstream 实现(HF transformers 或 vllm PR)可参考。先能跑对再谈性能(性能走 vllm-xpu-esimd-optimize)。
---

# 在 vllm-xpu 上 enable 一个新模型/新架构

这是**从零把一个 upstream 模型移植到 Intel XPU vllm-xpu 上跑通(先对、后快)** 的方法论，
提炼自 DiffusionGemma-26B(PR #45163) 和 gemma-4-12B(PR #44429) 的实战移植。
目标是「先跑对」——性能优化是之后的事(用 `vllm-xpu-esimd-optimize`)。

环境规约(容器/vllm 位置/kernel 位置/占卡/杀进程/git)统一走 `xpu-vllm-env`，本 skill 不重复。

## ⭐ 总原则：先验证「成败命门」，再投入工程移植

enable 一个新模型最大的浪费是：花几天做完整移植，最后发现某个**单点能力 XPU 根本不支持**。
所以第一步不是移植，是**识别单点最大风险 + 写最小 reproducer 数值验证**：

- 先问「这个模型有没有一个 XPU 可能不支持的核心算子/能力？」典型命门：
  - **双向注意力**(diffusion/encoder 模型 causal=False) —— XPU 的 FLASH_ATTN 是否支持？
  - 特殊 attention(MLA / sparse / sliding-window / 大 head_dim=512)
  - 特殊量化 MoE、特殊 RoPE、特殊激活。
- 命门确定后，**写一个脱离 vllm 的最小 reproducer**，直接调那条 kernel 链，对 `torch.scaled_dot_product_attention`(或 HF eager) 参考，验证数值(典型阈值 1e-3~4e-3 PASS)。
- **命门通过再动手**。实战:DiffusionGemma 先追出调用链 `gemma4 attn → FLASH_ATTN backend → xpu_ops.flash_attn_varlen_func → vllm_xpu_kernels(_vllm_fa2_C.varlen_fwd)`，
  最小测 `causal=False + paged + GQA(8:2)` 全 PASS → 判定 kernel 双向可用，才开始移植。
- ⚠️ **XPU 后端与 CUDA 同名文件行为常常不同**：`triton_unified_attention.py` 里的 `assert causal` 只约束 CUDA triton 路径；
  XPU 的 FLASH 走独立 SYCL kernel，不受该 assert 约束。别看到 CUDA 路径的限制就下结论 XPU 也不行——去看 XPU 实际走的那条链。

## 移植方式决策：为什么 cherry-pick 通常必败，改用 git apply .patch

容器里的 vllm-xpu 通常是 **downstream fork**：只有 `origin` remote、分支形如 `downstream/release/v0.21.0`，
**没有 upstream remote，与 upstream main 无共同祖先**。因此：

- `git cherry-pick <upstream PR commit>` **必失败**(跨 repo + 版本线不同，找不到 base)。
- 正解:**下载 PR 的全量合并 diff 用 `git apply`**。GitHub 的 `.patch` 端点给你全量 diff:
  `https://github.com/<org>/<repo>/pull/<N>.patch` 或 commit 的 `.patch`。
- **只 apply 必需文件子集**:跳过 NV-only / test / benchmark 噪声；apply 不上的 hunk(reject) **手工适配**。
- 用 `git apply --check` 先验证、`--reject` 收集 `.rej` 再逐个手工合。

（若目标是「改 llm-scaler 的 vllm_for_multi_arc.patch 大 patch」或跨版本迁移已有优化，见 `xpu-vllm-kernels-and-patch` 的 diff-of-patch 章节。）

## config / processor 手工 vendoring(容器 transformers 版本缺失时)

新模型的 `config`/`processor` 类常在**比容器更新的 transformers** 里才有(如容器 transformers 5.8.0 不识别
`gemma4_unified` / DiffusionGemma / `glm_moe_dsa`)。此时:

- **手工 vendoring**:从上游 transformers 拷 config/processor 类进 `vllm/transformers_utils/configs/`(或模型文件内联)，绕开 pip 版本。
- registry 注册点照搬同类已 enable 模型的模板(如照 diffusion_gemma 模板改 7 处)。
- PR 太大无法 cherry-pick(如 #44429 分叉 1754 文件)时，vendoring 是比 git apply 更干净的路子(手工只搬 config + processor + 模型文件 3 类)。

## 跑通顺序(每阶段各有专属崩点，别跳)

按此硬顺序推进，每过一关再看下一关:

1. **import 冒烟**:能 `from vllm.model_executor.models.<name> import <Cls>`。崩点:死引用别名、缺失 helper 未连同 import 一起移植。
2. **模型构造**:`LLM(model=...)` 能建起来。崩点:class `__init__` 签名与容器基类不匹配(多/少 `quant_config`/`prefix` 参数)。
3. **权重加载**:见下「权重加载映射」。崩点:权重名映射错位、stacked 合并、tied weight。
4. **warmup / profile_run**:`gpu_memory_utilization` 太高 OOM(diffusion 实测 0.9 OOM,降到 0.75);profile_run 喂全 0 张量会误触某些 fast path。
5. **首个 forward(attention)**:命门若没提前验证,这里暴露(输出全 0 / NaN / OUT_OF_RESOURCES)。
6. **实际生成**:输出通顺 → 再上精度 gate(gsm8k)。

## 权重加载映射(高频坑)

- `load_weights` / `hf_to_vllm_mapper` 的**前缀替换** 与 **delegate 给基类的二次重映射会打架**:
  例 `.experts.` → `.moe.experts.` vs 基类期望 `.experts.gate_up_proj`。先 dump 出实际 ckpt key 再定映射。
- **多模态模型**的 ckpt key 常是 `model.language_model.*`,而 `XXXForCausalLM` 期望 `model.*` → 需 rename 前缀。
- `nn.Linear`(非 vLLM quant Linear)子模块的权重要走特殊映射。
- **tied weights**(`lm_head.weight = embed_tokens.weight`):`device_map` 下必须先 `model.tie_weights()`,否则报 "on meta device"。
- ⭐**验证权重真加载了**:加载后 dump 某层 weight 的 norm/absmax,对 checkpoint ground truth 比;别只看"没报错"。

## 多模态(vision tower)

- 先确认权重是否**真带 vision tower**(带则 vision 别名/子模块不是包袱,得正确接);对齐容器里简化版 embedder 的构造签名。
- **vision embedder fp16 溢出**是高频真 bug:大 patch_dense(如 6912→3840 累加)fp16 溢出 inf→NaN。修法=内部 fp32 计算 + TP-correct 手动 all-gather
  (`ColumnParallelLinear` 权重被 TP 分片,直接 `.weight.float()` 会破坏 gather)。
- 量化下 vision:int4/fp8 会把 `patch_dense.weight` 挪到 `qweight` 置 None → 崩;patch-embed 层保持 fp16 不量化(传 `quant_config=None`)。
- 起 server:`--limit-mm-per-prompt image=N` 默认 1,**超限的图被静默丢弃**(表现为"看不清",不报错),超 limit 才 HTTP 400。必须显式设够。

## "upstream vs 容器分叉" 高频适配 bug 清单(照着排)

移植时反复出现的、因 fork 与 upstream 接口漂移导致的适配点:

- class `__init__` 签名多/少参数(`quant_config` / `prefix`)。
- `ModelConfig` 字段缺失(如 `use_fp64_gumbel`)→ 删该 kwarg 或补字段。
- 接口新增形参(如 `get_mm_embeddings` 多了 `req_states`)、字段搬家(某状态迁到 `req_states`)。
- **硬编码默认值需改成可被覆盖**:如 `build_attn_metadata` 硬编 `causal=True`,双向模型需加 `causal` 形参并让 model-specific 覆盖。
- 缺失 helper 函数(如 `recursive_replace_linear`)需连同其依赖 import(`maybe_prefix` 等)一起移植。
- MoE forward 可能已被本地 ESIMD 优化重写 → 只做最小适配(如加 `router_uses_prenormed_input`),别整块替换。

## 关键环境坑(移植期最常踩)

- ⭐**`cd /workspace` 下跑 python 被 `sys.path[0]=''`(cwd) 劫持到旁边的旧 checkout**(如 `/workspace/vllm`,HEAD 停在旧 PR),
  导致结构核查/import 全错位。**在 `/root` 或非仓库目录跑**,或用 `import <子模块>; print(mod.__file__)` 确认 editable 真实解析到哪个目录。
  (`vllm` 是 namespace package,`vllm.__file__` 常为 None,同名多目录会被合并成一个包——详见 `xpu-vllm-static-feasibility`。)
- **fp8 是 vllm 内置在线量化**:加载 bf16 权重时量化 Linear/MoE experts,`lm_head`/`embed` 保持高精度;冷加载大模型 ~8min(49G),热 ~40s。
  某些新模型 float16 下无 XPU MoE backend → **必须 fp8** 才能跑(DiffusionGemma 实测)。
- 编译任何 ESIMD/SYCL kernel 必须 `TORCH_XPU_ARCH_LIST=bmg`(否则默认含 XeLPG 集显报 dpas2/BF16 not supported)。

## enable-only 阶段的性能免责(别拿这些数字当优化结论)

- `enforce_eager=True` + `max_num_seqs=1` 下的 tok/s **不是优化数字**,只是"能跑"。
- 有些模型输出 token 数天生少(如提前收敛)是模型特性不是 bug。
- 扩散式模型「输出越长越快」等反直觉性能特征见 `xpu-diffusion-llm`。
- 跑对之后再进 `vllm-xpu-esimd-optimize` 做 profile → kernel。

## 精度 gate 与"乱码"归因(先排 prompt,再怀疑实现)

跑通后用 gsm8k few-shot(5/5 或 100 题)做精度 gate。遇到**输出乱码/重复循环**,按此顺序排,别一上来打逐层 trace:

1. **prompt 是否走了 chat template**:裸 `llm.generate(['...'])` 喂 **-it 模型**不 apply chat_template → 退化循环。
   用 `llm.chat` / `/v1/chat/completions` + messages / 直接喂 chat-template token_ids 就对。`/v1/completions`+prompt 不 apply → 退化。
   这是最易误判成"kernel/量化 bug"的大坑(gemma-4-31B/26B 都栽过)。
2. **chat 模板参数**:gemma 系默认 `enable_thinking=False` 预填空 `<|channel>thought` 通道致重复循环;
   传 `chat_template_kwargs={"enable_thinking": true}` + 够大的 `max_tokens` 修复。**HF golden 同样循环 → 证明非后端 bug**。
3. 排除以上再怀疑实现,用 `hf-golden-layerwise-diff` 逐层定位(HF 装不了时用无 golden 排除法,见该 skill)。

## 分支拓扑坑(合并回 release 时)

- 前序优化可能已被**squash-merge** 进 release,你的 feature 分支里仍是原始多 commit → 整条 rebase 会大冲突(git 认不出 squash 等价)。
- 正解:**只 cherry-pick 你这次新增的少数 commit** 到 release 之上(新增文件为主,仅注册文件小冲突)。
- 移植 commit 在容器内做(容器 git config,不 amend);push 前问用户(见 `xpu-vllm-env`)。

## 实战索引(可翻回原始交接文档)

- DiffusionGemma enable:调用链穿透 + 双向 attn 命门验证 + 5+3 个适配 bug + mixed-causal split;交接 `opt_gemma4/HANDOFF_diffusion_gemma_optimize.md`。
- gemma-4-12B(dense 多模态)enable:PR #44429 手工 vendoring + vision fp16 溢出修复 + 精度根因=`enable_thinking` 模板(非实现 bug,hf-golden 逐层证实 vLLM==HF);分支 `enable_gemma4_12b`,commit `9a84e23ba`/`8c7cb88b1`/`e50e7db89`。
