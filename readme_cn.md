# AiMai OCL

`aimai_ocl` 是一个面向研究的代码仓库，用于研究多智能体经济系统中的
Organizational Control Layer（OCL，组织控制层）。

这个仓库关注的问题是：在语言智能体执行经济交互任务时，是否可以通过一层
显式控制平面提升可靠性、约束满足性和决策质量。当前实现将 AgenticPay
视为外部协商 substrate，并在其上增加角色分工、风险门控、审计、升级与归因
等控制能力。

## 概览

本仓库主要围绕三个问题展开：

- 如何把原始语言动作映射为可执行的经济动作？
- 系统应当在什么情况下批准、改写、阻断或升级一个动作？
- 在部分可观测、多轮交互条件下，平台侧控制应如何被评估？

当前公开代码主要包括：

- [`aimai_ocl`](aimai_ocl) 中的 OCL 实现
- [`scripts`](scripts) 中的实验脚本
- [`benchmarks/conversational_consumer_selection_v1`](benchmarks/conversational_consumer_selection_v1)
  中的本地 benchmark 模块
- [`tests`](tests) 中的测试

## 核心模块

- `角色分工`
  区分规划、执行和升级职责
- `风险门控`
  对候选动作打分，并选择 approve、rewrite、block 或 escalate
- `审计轨迹`
  记录结构化控制决策，便于事后分析
- `升级与重规划`
  处理不可行或高风险动作
- `归因`
  支持对不同算法组件做 post-hoc 贡献分析

## 研究设置

当前仓库使用两层 benchmark 载体：

- `AgenticPay`
  作为外部协商运行时与任务 substrate
- `Conversational Consumer Selection V1`
  作为本地的引导式商品选择 benchmark，用于研究用户意图不完整时的平台控制

这个 benchmark 模块的说明见
[`benchmarks/conversational_consumer_selection_v1/README.md`](benchmarks/conversational_consumer_selection_v1/README.md)。
它对平台暴露的接口是：

- 输入：`Observation`
- 输出：`SelectionAction`

并支持四种意图可见性设置：

- `v0_structured`
- `v1_direct_intent`
- `v1_partial_intent`
- `v2_hidden_intent`

## 仓库结构

```text
aimai_ocl/
  controllers/    角色、门控、审计、升级、控制曲面
  runners/        single-agent 与 OCL 执行路径
  adapters/       AgenticPay 适配层
  schemas/        动作、审计与约束 schema
scripts/
  run_demo.py
  run_batch_eval.py
  run_ablation_matrix.py
  run_tau_sweep.py
benchmarks/conversational_consumer_selection_v1/
  src/conversational_consumer_selection/
  tests/
tests/
```

## 安装

先创建 Python 环境并安装本项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

再单独安装 AgenticPay 这个外部依赖：

```bash
pip install "git+https://github.com/SafeRL-Lab/AgenticPay.git"
```

如果你也想把引导式商品选择 benchmark 当作包来运行，可以再执行：

```bash
pip install -e benchmarks/conversational_consumer_selection_v1
```

使用 OpenAI 后端时，需要设置：

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.4-mini
```

## 快速开始

运行一次协商 episode：

```bash
python scripts/run_demo.py --arm ocl_full --model gpt-5.4-mini --seed 42
```

运行成对 batch evaluation：

```bash
python scripts/run_batch_eval.py \
  --arms single,ocl_full \
  --episodes-per-arm 20 \
  --seed-base 42 \
  --output-dir outputs/main_result_v1
```

运行 `tau` 控制强度 sweep：

```bash
python scripts/run_tau_sweep.py \
  --tau-values 0.0,0.25,0.5,0.75,1.0 \
  --episodes-per-arm 20 \
  --seed-base 42 \
  --output-root outputs/tau_sweep_v1
```

运行引导式商品选择 benchmark demo：

```bash
cd benchmarks/conversational_consumer_selection_v1
PYTHONPATH=src python -m conversational_consumer_selection.single_agent_demo --backend demo
```

## 关键文件

- [`aimai_ocl/controllers/ocl_controller.py`](aimai_ocl/controllers/ocl_controller.py)
  OCL 主控制路径
- [`aimai_ocl/controllers/risk_gate.py`](aimai_ocl/controllers/risk_gate.py)
  风险门控算法，包括 `tau` 控制家族
- [`aimai_ocl/controllers/coordinator.py`](aimai_ocl/controllers/coordinator.py)
  角色分工逻辑
- [`aimai_ocl/evaluation_metrics.py`](aimai_ocl/evaluation_metrics.py)
  实验指标与汇总字段
- [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)
  算法注册表与实验组合
- [`benchmarks/conversational_consumer_selection_v1/src/conversational_consumer_selection/env.py`](benchmarks/conversational_consumer_selection_v1/src/conversational_consumer_selection/env.py)
  引导式商品选择环境
- [`benchmarks/conversational_consumer_selection_v1/src/conversational_consumer_selection/schemas.py`](benchmarks/conversational_consumer_selection_v1/src/conversational_consumer_selection/schemas.py)
  benchmark 输入输出契约

## 测试

运行主测试集：

```bash
pytest
```

运行 benchmark 专项测试：

```bash
pytest benchmarks/conversational_consumer_selection_v1/tests/test_benchmark.py
```

## 状态

这是一个活跃演进中的研究仓库。随着论文实现推进，接口和实验设置仍可能调整。

## 引用

论文发布后会补充 citation 信息。
