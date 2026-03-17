# AiMai OCL

`aimai_ocl` 是一个独立项目，用于在 AgenticPay 之上构建 Organizational Control Layer（OCL，组织控制层）。

项目设计遵循以下论文方向：

`Agent Infrastructure Matters: Organizational Control for Reliable Multi-Agent Economic Systems`

核心思路是将 AgenticPay 作为底层经济协商基础设施，而 `aimai_ocl` 作为控制层研究与工程实现载体，负责：

- 组织控制
- 角色分解
- 风险门控
- 审计追踪
- 升级与重规划
- 归因与激励
- 可复现基准

## 项目定位

此仓库并不用于替代 AgenticPay，也不是 AgenticPay 的分支实现。

相反，它将 AgenticPay 用作：

- 任务与环境库
- 协商状态机
- 基线评估基底

并补齐论文提案所需但缺失的控制平面能力：

- 原始动作到可执行动作的映射
- 显式约束与风险检查
- 标准化审计轨迹
- 升级逻辑
- 归因与激励信号

## 与提案对齐

该项目的结构与论文提案的实验设计保持一致：

- `Single-agent` 基线
- `OCL multi-agent` 系统
- 围绕 `role`、`gate`、`audit`、`escalation` 的在线消融实验
- `attribution` 作为 post-hoc 分析模块（当前阶段）
- 对抗性与长时程评估
- 后续映射到 AiMai 平台实验

完整规划文档位于 [`docs/PAPER_PROPOSAL.md`](docs/PAPER_PROPOSAL.md)。

## 核心算法（论文关键）

### 1) 角色分工（Role Decomposition）

- 算法 ID：
  - `role_v1_rule`
  - `role_v1_seller_only`
  - `role_v2_state_machine`
- 核心公式：

$$
\begin{aligned}
\mathrm{phase}_t &= f_{\mathrm{phase}}(\mathrm{round}_t, \mathrm{buyer}_t, \mathrm{max\_rounds}) \\
\mathrm{decision\_role}_t &= \pi_{\mathrm{role}}(\mathrm{phase}_t) \\
\mathrm{execution\_role}_t &= A_s \\
\mathrm{escalation\_role}_t &= A_p
\end{aligned}
$$

`role_v2_state_machine` 使用显式阶段：
`discovery / recommendation / negotiation / closing / escalation`。

- 实现位置：
  - [`aimai_ocl/controllers/coordinator.py`](aimai_ocl/controllers/coordinator.py)
  - [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

### 2) 风险门控（Risk Gating as Control Barrier）

- 算法 ID：
  - `gate_v1_default`
  - `gate_v1_strict`
  - `gate_v1_lenient`
  - `gate_v2_barrier`
  - `gate_v2_barrier_strict`
- 核心公式：

$$
\rho_t = f_{\mathrm{risk}}(a_t^{\mathrm{raw}}, s_t, h_t), \quad \rho_t \in [0, 1]
$$

$$
\mathrm{action}_t =
\begin{cases}
\mathrm{approve}, & \rho_t < \tau_{\mathrm{rewrite}} \\
\mathrm{rewrite/confirm}, & \tau_{\mathrm{rewrite}} \le \rho_t < \tau_{\mathrm{block}} \\
\mathrm{block/escalate}, & \rho_t \ge \tau_{\mathrm{block}}
\end{cases}
$$

Barrier 风险门控中包含显式漏检概率代理 `epsilon_miss`，
用于高风险误放行分析。

- 实现位置：
  - [`aimai_ocl/controllers/risk_gate.py`](aimai_ocl/controllers/risk_gate.py)
  - [`aimai_ocl/controllers/ocl_controller.py`](aimai_ocl/controllers/ocl_controller.py)
  - [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

### 3) 责任归因与激励（Attribution）

- 算法 ID：
  - `shapley_v1_exact`
  - `shapley_v1_reward_only`
  - `counterfactual_v1`
- 核心公式：

$$
\begin{aligned}
V(\mathrm{trace}) &= w_s \cdot \mathrm{success}
+ w_r \cdot \mathrm{seller\_reward}
+ w_g \cdot \mathrm{global\_score} \\
&\quad - w_v \cdot \mathrm{violations}
- w_e \cdot \mathrm{escalations}
- w_t \cdot \mathrm{rounds}
\end{aligned}
$$

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!}\left(V(S \cup \{i\}) - V(S)\right)
$$

$$
w_i = \frac{\max(\phi_i, 0)}{\sum_j \max(\phi_j, 0)}
$$

`counterfactual_v1` 支持对稀疏联盟取值进行 Monte-Carlo Shapley 估计。

- 实现位置：
  - [`aimai_ocl/attribution_shapley.py`](aimai_ocl/attribution_shapley.py)
  - [`aimai_ocl/attribution_counterfactual.py`](aimai_ocl/attribution_counterfactual.py)
  - [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

### Attribution 阶段策略（当前共识）

- `Phase A（当前）`：Attribution 仅用于 post-hoc 分析，不回流到 OCL 在线决策。
- `Phase B（TODO）`：引入跨 episode 的 credit memory，并将 credit 映射到
  coordinator/gate/escalation 的控制参数（不训练模型参数）。

## AgenticPay 依赖

该项目现在假设 AgenticPay 在当前 Python 环境中可导入。推荐设置如下：

```bash
pip install "git+https://github.com/SafeRL-Lab/AgenticPay.git"
```

完成后，`aimai_ocl` 使用标准 `import agenticpay` 导入，不再需要本地路径引导层。

## 模型运行时配置

模型选择已集中在
[`aimai_ocl/model_runtime.py`](aimai_ocl/model_runtime.py) 中管理。当前运行时
已简化为 OpenAI-only，直接使用 AgenticPay 的
`agenticpay.models.openai_llm.OpenAILLM`。

运行级模型字段定义在
[`RunConfig`](aimai_ocl/experiment_config.py)：

- `model`

可选环境变量默认值：

- `AIMAI_MODEL`
- `OPENAI_MODEL`
- `OPENAI_API_KEY`（必需）

可通过 CLI 覆盖的脚本：

- [`scripts/run_demo.py`](scripts/run_demo.py)
- [`scripts/run_batch_eval.py`](scripts/run_batch_eval.py)
- [`scripts/run_shapley_min.py`](scripts/run_shapley_min.py)

## AgenticPay 数据契约（实际传递内容）

`aimai_ocl` 将 AgenticPay 视为严格的输入/输出边界。当前运行时实际传递与消费的数据如下：

1. `agenticpay.make(env_id, **env_kwargs)`
- `env_id` 用于选择任务（例如 `Task1_basic_price_negotiation-v0`）。
- `env_kwargs` 由适配器原样透传，常见字段包括：
  `buyer_agent`、`seller_agent`、`max_rounds`、`initial_seller_price`、
  `buyer_max_price`、`seller_min_price`。

2. `env.reset(**reset_kwargs)`
- `reset_kwargs` 会先经过 `enforce_single_product_scenario` 再透传。
- v1 要求必须是单商品：
  提供 `product_info` 字典，或提供仅含一个元素的 `products`（会被归一化）。
- 常用 reset 字段：
  `user_requirement`、`product_info.name`、`product_info.price`、`user_profile`。

3. 每轮参与方调用与 `env.step(...)`
- 运行器读取 `observation["current_round"]` 和
  `observation["conversation_history"]`。
- 买方路径：
  `buyer_agent.respond(conversation_history, current_state)` -> 文本 ->
  直接透传为 `buyer_action`。
- 卖方路径（`single`）：
  卖方文本只做审计记录，然后透传。
- 卖方路径（`ocl_full`）：
  卖方文本 -> `RawAction` -> 角色/约束/风险/升级 ->
  `ExecutableAction.final_text`。
- 每轮最终写入 AgenticPay 的调用是：
  `env.step(buyer_action=<str|None>, seller_action=<str|None>)`。

4. 回合终止信息（最终一步的 `info`）
- `aimai_ocl` 当前消费这些键：
  `status`、`round`、`termination_reason`、`agreed_price`、
  `buyer_price`、`seller_price`、`buyer_reward`、`seller_reward`、
  `global_score`、`buyer_score`、`seller_score`。
- 这些字段会被写入 `EpisodeTrace.final_metrics` 和 CLI 结果记录。

## 不阅读 AgenticPay 内部实现时，应修改哪些位置

如果你要修改“传什么、怎么控、在哪里控”，优先改这些稳定接入点：

- 环境边界与 API 透传：
  [`aimai_ocl/agenticpay_runtime.py`](aimai_ocl/agenticpay_runtime.py)、
  [`aimai_ocl/adapters/agenticpay_env.py`](aimai_ocl/adapters/agenticpay_env.py)
- 文本到动作映射（`RawAction` 解析、意图/价格提取）：
  [`aimai_ocl/adapters/agenticpay_actions.py`](aimai_ocl/adapters/agenticpay_actions.py)
- 场景载荷契约（`reset_kwargs` 校验/归一化）：
  [`aimai_ocl/runners/scenario_validation.py`](aimai_ocl/runners/scenario_validation.py)
- 买方/卖方数据流与控制接入点：
  [`aimai_ocl/runners/single_episode.py`](aimai_ocl/runners/single_episode.py)、
  [`aimai_ocl/runners/ocl_episode.py`](aimai_ocl/runners/ocl_episode.py)
- 控制策略（放行/改写/阻断/升级）：
  [`aimai_ocl/controllers/constraint_engine.py`](aimai_ocl/controllers/constraint_engine.py)、
  [`aimai_ocl/controllers/risk_gate.py`](aimai_ocl/controllers/risk_gate.py)、
  [`aimai_ocl/controllers/escalation_manager.py`](aimai_ocl/controllers/escalation_manager.py)
- CLI 参数到运行时载荷的映射：
  [`scripts/run_demo.py`](scripts/run_demo.py)

如果 AgenticPay 的 observation/info 字段结构发生变化，以上文件是第一优先更新点。

## 当前状态

仓库目前已包含可运行的实验框架（而不只是脚手架）：

- 基线实验臂：`single`、`ocl_full`
- 可插拔算法：`role`、`gate`、`escalation`、`attribution`
- 算法组合：`v1_default`、`v1_role_ablation`、`v2_research`
- 协议输出：`main`、`ablation`、`adversarial`、`repeated`、`roi`
- 针对控制器契约与算法主体的确定性测试

## 最小 AgenticPay 使用方式

当前最小化集成路径如下：

```python
from aimai_ocl import run_single_negotiation_episode

trace, final_info = run_single_negotiation_episode(
    env_id="Task1_basic_price_negotiation-v0",
    buyer_agent=buyer_agent,
    seller_agent=seller_agent,
    env_kwargs={
        "max_rounds": 10,
        "initial_seller_price": 180.0,
        "buyer_max_price": 120.0,
        "seller_min_price": 90.0,
    },
    reset_kwargs={
        "user_requirement": "I need a winter jacket",
        "product_info": {"name": "Winter Jacket", "price": 180.0},
        "user_profile": "Budget-conscious and compares options before buying.",
    },
)
```

在当前阶段，运行器有意保持轻量。它的职责是让 AgenticPay 基准调用路径清晰可见，同时复用现有 AgenticPay 的买方/卖方代理（当前使用 upstream 可用的 `SellerAgent`）。

## 快速演示脚本

当前 v1 范围仅支持单商品协商。

你可以直接从此仓库运行一个基线回合：

```bash
export AIMAI_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...
python scripts/run_demo.py --arm single --seed 42
```

运行 OCL 控制模式：

```bash
python scripts/run_demo.py --arm ocl_full --seed 42
```

在 `ocl_full` 中，买方动作被视为外部用户模拟输入并直接传给环境，而卖方动作会经过 OCL 控制。

对比基线与 OCL：

```bash
python scripts/run_demo.py --arm single --seed 42
python scripts/run_demo.py --arm ocl_full --seed 42
```

`--arm` 字段被视为消融实验契约；其中一些开关是脚手架标志，会在后续步骤中逐步接入实际行为。

在不触发模型调用的情况下预览解析后的配置：

```bash
python scripts/run_demo.py --arm ocl_full --seed 42 --dry-run
```

将完整回合审计轨迹导出为 JSON：

```bash
python scripts/run_demo.py --arm ocl_full --seed 42 --trace-json outputs/trace.json
```

选择显式的算法/协议组合（step-9/10 模块化槽位）：

```bash
python scripts/run_demo.py \
  --arm ocl_full \
  --algorithm-bundle v2_research \
  --experiment-protocol offline_v1 \
  --seed 42
```

只覆盖一个算法组件（用于消融/调试）：

```bash
python scripts/run_demo.py \
  --arm ocl_full \
  --algorithm-bundle v1_default \
  --role-algorithm role_v1_seller_only \
  --gate-algorithm gate_v1_strict \
  --attribution-algorithm shapley_v1_reward_only \
  --seed 42
```

运行配对批量评估（`single` vs `ocl_full`），输出 CSV/JSON 报告：

```bash
python scripts/run_batch_eval.py --episodes-per-arm 5 --seed-base 42
```

运行主要离线实验臂（`single`、`ocl_full`）：

```bash
python scripts/run_batch_eval.py --arms single,ocl_full --episodes-per-arm 5
```

为整批实验覆盖 bundle/protocol id：

```bash
python scripts/run_batch_eval.py \
  --arms single,ocl_full \
  --episodes-per-arm 5 \
  --algorithm-bundle v2_research \
  --experiment-protocol offline_v1
```

批量模式也支持组件级覆盖：

```bash
python scripts/run_batch_eval.py \
  --arms single,ocl_full \
  --episodes-per-arm 5 \
  --algorithm-bundle v1_default \
  --role-algorithm role_v1_seller_only \
  --gate-algorithm gate_v1_strict \
  --escalation-algorithm escalation_v1_no_replan \
  --attribution-algorithm shapley_v1_reward_only
```

指定输出位置并保存每次运行的 trace 文件：

```bash
python scripts/run_batch_eval.py \
  --episodes-per-arm 10 \
  --output-dir outputs/batch_eval_v1 \
  --save-traces
```

批量产物现在包含 `protocol_outputs.json`，其中含有 `main/ablation/adversarial/repeated/roi` 协议输出。
其中 `main` 现在会额外给出 `paired_statistics.ocl_vs_single`，
按 `(episode_index, seed)` 配对后输出 `success/has_violation/round/seller_reward/latency_sec`
五个指标的 `mean_delta`、`delta_ci95` 和 `sign_flip_pvalues`（含 `p_two_sided`）。

可通过以下参数控制统计抽样规模：

```bash
python scripts/run_batch_eval.py \
  --arms single,ocl_full \
  --episodes-per-arm 20 \
  --bootstrap-samples 5000 \
  --permutation-samples 50000
```

一键运行默认消融矩阵（E0~E5）并汇总
`Δ_default - Δ_ablation`：

```bash
python scripts/run_ablation_matrix.py \
  --episodes-per-arm 20 \
  --seed-base 42 \
  --bootstrap-samples 1000 \
  --permutation-samples 20000 \
  --output-root outputs/ablation_matrix_v1
```

输出包含：
- `ablation_summary.json`
- `ablation_metrics.csv`
- `ablation_contributions.csv`

常用自定义参数：

```bash
python scripts/run_demo.py \
  --arm single \
  --seed 123 \
  --max-rounds 12 \
  --buyer-max-price 125 \
  --seller-min-price 95 \
  --product-name "Premium Winter Jacket" \
  --product-price 185
```

## 当前代码地图

核心运行时与 AgenticPay 桥接：

- [`aimai_ocl/agenticpay_runtime.py`](aimai_ocl/agenticpay_runtime.py)
- [`aimai_ocl/adapters/agenticpay_env.py`](aimai_ocl/adapters/agenticpay_env.py)
- [`aimai_ocl/adapters/agenticpay_actions.py`](aimai_ocl/adapters/agenticpay_actions.py)

最小 OCL 形式化接口：

- [`aimai_ocl/schemas/actions.py`](aimai_ocl/schemas/actions.py)
- [`aimai_ocl/schemas/constraints.py`](aimai_ocl/schemas/constraints.py)
- [`aimai_ocl/schemas/audit.py`](aimai_ocl/schemas/audit.py)

OCL 控制器与算法：

- [`aimai_ocl/controllers/role_policy.py`](aimai_ocl/controllers/role_policy.py)
- [`aimai_ocl/controllers/risk_gate.py`](aimai_ocl/controllers/risk_gate.py)
- [`aimai_ocl/controllers/ocl_controller.py`](aimai_ocl/controllers/ocl_controller.py)
- [`aimai_ocl/controllers/coordinator.py`](aimai_ocl/controllers/coordinator.py)
- [`aimai_ocl/controllers/escalation_manager.py`](aimai_ocl/controllers/escalation_manager.py)

归因算法：

- [`aimai_ocl/attribution_shapley.py`](aimai_ocl/attribution_shapley.py)
- [`aimai_ocl/attribution_counterfactual.py`](aimai_ocl/attribution_counterfactual.py)

插件与实验协议注册：

- [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

回合执行：

- [`aimai_ocl/runners/single_episode.py`](aimai_ocl/runners/single_episode.py)
- [`aimai_ocl/runners/ocl_episode.py`](aimai_ocl/runners/ocl_episode.py)

## 最小 OCL 接口

第一批正式 OCL 对象现位于 `aimai_ocl.schemas`：

- `RawAction`
- `ExecutableAction`
- `ConstraintCheck`
- `AuditEvent`
- `EpisodeTrace`

这些是后续实现以下能力所需的最小对象：

- `g_Pi: (m_1:t, a_raw_1:t) -> a_exec_1:t`
- 显式约束评估
- 结构化审计轨迹
- 后续的归因与激励逻辑

## OCL 控制器栈

当前控制器栈（v1 + v2 研究变体）：

- `RolePolicy` 验证 `role -> intent` 权限
- `ConstraintEngine` 应用确定性硬检查（格式、预算/底价、隐私）
- `RiskGate` 与 `BarrierRiskGate` 提供两类门控策略
- `OCLController` 返回 `OCLControlResult`，包含：
  - `ExecutableAction`
  - 收集到的 `ConstraintCheck` 记录
  - 生成的 `AuditEvent` 记录

当前角色分解支持：

- `Coordinator`（基于规则）
- `SellerOnlyCoordinator`（角色消融）
- `StateMachineCoordinator`（算法化角色状态机）

升级层通过 `EscalationManager` 接入：

- 被阻断/高风险的卖方动作会触发 `ESCALATION_TRIGGERED`
- 可恢复的价格违规会执行一次确定性的 `REPLAN_APPLIED`
- 不可行/冲突场景会建议人工接管

决策/违规契约：

- `ControlDecision`：`approve / rewrite / block / escalate`
- `ViolationType`：失败或需重点关注检查项的规范化分类标签

## 路线图来源

路线图与完成状态维护于：

- [`TODO.md`](TODO.md)

## 文档

- 提案：[`docs/PAPER_PROPOSAL.md`](docs/PAPER_PROPOSAL.md)
- 下一层交接：[`docs/NEXT_LAYER_HANDOFF.md`](docs/NEXT_LAYER_HANDOFF.md)
- AgenticPay 加载器：[`aimai_ocl/agenticpay_runtime.py`](aimai_ocl/agenticpay_runtime.py)
