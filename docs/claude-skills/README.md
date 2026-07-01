# Intel XPU vllm-xpu / llm-scaler Claude skills

这一组 skill 沉淀了在 **Intel XPU(BMG / Arc B-series)** 上做 vllm-xpu + llm-scaler(ESIMD kernel)开发的可复用方法论，
来自 gemma-4-26B/31B/12B、DiffusionGemma、Qwen3.5/3.6-35B-A3B、MiniCPM-V 4.6、GLM-5.2/DeepSeek-V4 等实战。
每个 skill 的 `SKILL.md` frontmatter 里有 `description`，用于按任务匹配。

## 入口:先读 `xpu-vllm-env`
任何「在容器 X 里干 XPU vllm 的活」先加载 **`xpu-vllm-env`** 确认环境(容器名 + 三套代码位置 + 占卡 + 进程/git/编译规约)，
再按任务类型加载下面对应的专项 skill。

## 专项 skill 一览

| skill | 什么时候用 |
|---|---|
| **xpu-vllm-env** | 环境与操作规约(容器/三套代码位置/占卡/杀进程/编译/git)。任务入口，先读它。 |
| **xpu-vllm-enable-new-model** | 从零 enable 一个新模型/新架构:移植 upstream PR(git apply .patch)、config vendoring、权重映射、多模态、"先验命门再动手"、跑通顺序。 |
| **xpu-vllm-quant-enable-debug** | enable 量化(fp8 e4m3/e5m2、sym_int4、streaming、fp8 KV)+ 量化特有数值 bug(prefill 分界崩、streaming 越界、合成 oracle 骗人)。 |
| **vllm-xpu-esimd-optimize** | 写/接 ESIMD kernel 优化性能的 8 阶段循环:reproducer→gsm8k 门禁→ITL→unitrace profile→写 kernel→wire→verify→bisect。含 wire-up 坑清单、GQA attention kernel、fp8/DPAS 硬件坑。 |
| **xpu-graph-enable** | 启用 XPU graph(cudagraph)消 decode launch 开销:排查顺序、多卡三处必改、capture 崩/replay hang/`!!!!` 诊断、varlen graph-safe、GDN×compile 确定性错误。 |
| **hf-golden-layerwise-diff** | 模型输出乱码/精度和参考对不上:HF golden 逐层 diff 定位首处发散;含无 golden 排除法、chat 模板循环归因。 |
| **xpu-vllm-static-feasibility** | 不动硬件、纯静态读代码判断某架构/PR 能否在 XPU 跑或移植、两版本 feature diff。 |
| **xpu-vllm-kernels-and-patch** | 两套 kernel 仓区分/构建/集成、vllm.patch 应用(diff-of-patch)、版本迁移(0.14→0.19→0.21)、.so RUNPATH / 跨容器同步。 |
| **xpu-diffusion-llm** | 扩散式 LLM(diffusion decoding，如 DiffusionGemma)的 enable/推理/测性能/优化:范式差异、性能测法陷阱、batch MoE 集成。 |

## 配套 assets
- `hf-golden-layerwise-diff/assets/hf_reference.py` — HF golden-reference 模板。
- `vllm-xpu-esimd-optimize/assets/` — `offline_chat_reqs.py`(gsm8k 精度门禁)、`bench_itl.py`(ITL)、`profile_attn.py`(层计时)、`unitrace_agg.py`(unitrace 聚合)。

## 与 auto-memory 的关系
各模型的**具体事实**(架构参数/已 enable 状态/各 kernel 阻塞点/历史结论/commit)记在用户 auto-memory 的 `project_*` 系列;
这里的 skill 是**方法论**(怎么做、有哪些坑)。开工前两者都值得参考。
