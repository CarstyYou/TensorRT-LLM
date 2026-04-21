# CLAUDE.md

## 核心原则**必须谨记**
1. 每次回复前都用`xiy`称呼我
2. 在分析问题时，任合推论需要数据/代码证据支撑，没有缺乏数据的推论需要询问我
3. 遇到不确定的设计决策时，先和我讨论，不要自作主张
4. 你是严谨的大模型推理服务的专家，需要清晰了解 `TTFT`,`TPOT`,`TPS`等吞吐，延迟的概念和原理，协助我完成**benchmark**和**breakdown**等工作




## 项目关心的核心问题

### **原始实验配置**

- **硬件配置**： nvidia sm120 RTX6000D 84G 单卡/双卡

- **场景**： Qwen3-32B-FP8模型，目前模型文件在/raid/models/Qwen3-32B-FP8目录下，配置 isl=1800, osl=32，对如下配置做 serving benchmark，{(num_prompt=100 cc=1),(num_prompts=200, cc=4),(num_prompts=200, cc=8),(num_prompts=200, cc=16),(num_prompts=1000, cc=32),(num_prompts=1000, cc=64)}; 分别测试对TP=1和TP=2进行benchmark.

- sglang-client:
    1. isl=180,osl=32
    2. 不固定ISL（`--random-range-ratio` 0.5）
    3. 不固定OSL (`--extra-request-body` '{"ignore_eos": false}')
- trtllm-sever: 
    1. `max_batch_size`=128
    2. `max_num_tokens`=16384
    3. `enable_block_reuse`=true(默认)
    4. 同一 server 不重启依次跑 cc=4→8→16→32
- sglang-server:
    1. `--disable-radix-cache`（禁用 KV cache prefix 复用）
    2. `--fp8-gemm-backend cutlass`（标准 FP8 MMA + 软件 blockwise scaling，F-019）
    3. `chunked_prefill_size=8192`（默认，F-024）
    4. prefill/decode 分离调度（默认，非 in-flight batching，F-024）
    5. `--mem-fraction-static 0.9`



### 核心问题
1. **TP=2 cc=4→8 TTFT/TPOT 暴跌** ✅ 已解决
    - **结论：骤降不存在，是 `enable_block_reuse=true` 导致的 KV cache 数据污染（F-047）。** 原始数据在同一 server 上依次跑 cc=4→cc=8，cc=8 复用了 cc=4 的 KV cache block，TTFT 被人为压低。禁用 reuse 或 per cc 重启 server 后，cc=4→8 TTFT 从 300→411ms（上升），TPOT 从 39→64ms（上升），无骤降。
    - 数据支撑：[no_restart_bench.md](.cursor/memory/no_restart_bench.md) disable reuse 对照组, [fixclean_bench.md](.cursor/memory/fixclean_bench.md) 隔离数据
2. **TP=2 cc=16→32 TTFT/TPOT 暴涨** ✅ 已解决
    - 原始现象：TPOT 30→207ms（6.9x）
    - **已解决：6.9x 暴涨是两个实验缺陷叠加的假象**：
        1. **KV cache reuse（F-047/F-051）**：cc=16 TPOT=30ms 被 reuse 压低（干净值 55ms）
        2. **np 不一致**：cc≤16 用 np=200（reuse 效果强），cc≥32 用 np=1000（冲刷 reuse 回到真实值）
    - 干净数据（fixclean，统一 np=1000，per cc restart）：cc=4→8→16→32 TPOT = 24→26→49→118ms
    - **后续控制实验已确认：干净数据中剩余的 cc=8→16 和 cc=16→32 TPOT 增长，主因是 Mixed iteration 占比提升（F-050/F-052）**。`exp22` 中 cc=16 的 Mixed 占比从 5.7%→2.9%→0.0% 时，TPOT 从 56.3→29.7→22.5ms；cc=32 的 Mixed 占比从 13.9%→5.6% 时，TPOT 从 140.7→91.9ms。`exp24` 补测进一步验证 cc=32 在 Mixed=0 且 `bw=20ms` 时 TPOT=26.9ms，说明去除 Mixed 后 TPOT 可收敛到纯 decode 下限 ~20-27ms。
    - 数据支撑：[fixclean_bench.md](.cursor/memory/fixclean_bench.md), [exp22_bench.md](.cursor/memory/exp22_bench.md), [exp23_bench.md](.cursor/memory/exp23_bench.md), [batch_wait.md](.cursor/memory/exp25_nsys/batch_wait.md)
3. **cc=64 时 trtllm-serve 相较于 sglang-serve TTFT 为何那么高** ✅ 已解决
    - **结论：差距来自两个可对齐的配置差异，对齐后两者性能一致（<3%）**：
        1. **max_num_tokens**：trtllm 默认 16384 vs sglang chunked_prefill_size=8192。mnt=16384 时一次 prefill 更多请求（ctx=8/9），排队更长 → TTFT 高。对齐 mnt=8192 后 trtllm TTFT 从 10134→5966ms，接近 sglang 6265ms。
        2. **调度模式**：sglang 默认 prefill/decode 分离（未开启 enable_mixed_chunk），trtllm 只支持 ifb。分离调度 prefill 优先 → TTFT 极高（15963ms）但 TPOT 低（486ms）；ifb 模式 TTFT 更低但 TPOT 更高。sglang 开启 enable_mixed_chunk 后 TTFT 从 15963→6265ms，TPOT 从 486→800ms。
    - **结论**：统一配置（mnt=8192 + enable_mixed_chunk）下：
        - 标准 FP8 kernel：trtllm TTFT=6210 / TPOT=794 vs sglang TTFT=6265 / TPOT=800，trtllm 略优（<1%）
        - MXFP8 kernel：trtllm TTFT=5966 / TPOT=759，比标准 FP8（TTFT=6210 / TPOT=794）更好（TTFT -4%, TPOT -4%）
        - 调度行为完全一致（iters/Mixed/ctx 分布相同），性能差异纯粹来自 kernel 执行效率
    - 数据支撑：[sglang_bench.md](.cursor/memory/sglang_bench.md)
4. **固定 ISL=1800/OSL=32 NP=1000 在什么配置下能满足客户SLA：TTFT=1s,TPOT=30ms,TPS越大越好** ⚠️待解决
    - **结论：**：

---

## 执行步骤

每一次对话开始前，需要读取项目下的 [findings.md](.cursor/memory/findings.md), [progress.md](.cursor/memory/progress.md), 如果没有，创建这两份份文件。

如果项目目录中已经有了，`findings.md`，`progress.md`，读取三份文件的内容，报告当前的进度，接着往下执行后面的计划。 

### 任务工作流 Skill 使用导航

按操作类型选择对应 Skill：

| 操作类型 | Skill | 路径 | 何时使用 |
|---------|-------|------|---------|
| **tmux 操作** | 6kd-bench-tmux | `.cursor/skills/6kd-bench-tmux/SKILL.md` | 在 R6KD-CX8aaS-GPU-02 tmux 中发命令、读输出 |
| **跑 benchmark** | 6kd-benchmark | `.cursor/skills/6kd-benchmark/SKILL.md` | 启动 server/client、固定 ISL/OSL、保存 log、自动化多组实验 |
| **提取数据** | 6kd-data-extraction | `.cursor/skills/6kd-data-extraction/SKILL.md` | 从 client log 提取 TTFT/TPOT/TPS、从 server log 分析调度、nsys kernel 统计 |
| **深入分析** | 6kd-breakdown-analysis | `.cursor/skills/6kd-breakdown-analysis/SKILL.md` | per-iteration GEMM 分布、prefill/decode 分类、GPU idle gap |

**使用规则**：
- 在 tmux 中操作前必须先读 `6kd-bench-tmux/SKILL.md`（window 分工、等待策略）
- 跑 benchmark 前必须先读 `6kd-benchmark/SKILL.md`（固定 ISL/OSL 约定、log 保存规范）


## 工作流约定

每次工作遵循：明确需求目标 -> 采集实验数据 -> 提取分析数据 -> 汇总结论 


### Phase 1: 明确需求目标

- 在 `.cursor/memory/` 下新建  `exp<编号>_<简述>` 目录
  - 例如：`.cur/memory/exp26_bw25/`
- **必须和用户讨论** 确认任务目标，明确需要实验采集性能数据，生成`plan.md`

### Phase 2: 采集实验数据
- 在 `.cursor/memory/exp_XXX/` 实验目录下
  - 新建 `client_logs` 保存 sglang-client生成的 `.out` 和 `.log` 文件
  - 新建 `sever_logs` 保存 trtllm-server生成的 `.out` 和 `.log` 文件
  - 生成 测试 `bash` 脚本让我确认后执行

### Phase 3: 提取性能数据
- 查看当前  **提取数据** skill 是否满足实验 采集/统计 数据
- 若没有 则跟生成新分析脚本并测试验证正确后，保存到 `.cursor/skills/6kd-data-extraction/scripts`中， 并更新到 SKILL.md


### Phase 4: 核验数据，形成结论
- 整理数据到 `result.md`
- 仔细核验实验数据是否有异常，如果有异常需要重新review 测试脚本 和 分析脚本，有问题重新与客户讨论后进入 Phase 2
- 数据没有问题则沉淀和经验到[findings.md](.cursor/memory/findings.md), [progress.md](.cursor/memory/progress.md)



---


