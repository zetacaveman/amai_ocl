# AiMai OCL TODO (1 -> 10, Engineering + Research)

目标：从一开始就把核心算法做成可替换模块，保证后续实验/讨论可快速迭代。
注：本文件已替代旧版 `P0/P1/P2/P3/P4` 分段写法，当前以 `1..10` 为唯一顺序。

## 核心三算法（重点标注）

### A. 角色分工（Role Decomposition）

- 算法 ID：`role_v1_rule / role_v1_seller_only / role_v2_state_machine`
- 公式：

```text
phase_t = f_phase(round_t, buyer_text_t, max_rounds)
decision_role_t = pi_role(phase_t)
execution_role_t = A_s
escalation_role_t = A_p
```

- 实现文件：
  - `aimai_ocl/controllers/coordinator.py`
  - `aimai_ocl/plugin_registry.py`

### B. 风险门控（Risk Gating as Control Barrier）

- 算法 ID：`gate_v1_default / gate_v1_strict / gate_v1_lenient / gate_v2_barrier / gate_v2_barrier_strict`
- 公式：

```text
rho_t = f_risk(a_raw_t, s_t, h_t),  rho_t in [0,1]
if rho_t < tau_rewrite:                 approve
if tau_rewrite <= rho_t < tau_block:    rewrite/confirm
if rho_t >= tau_block:                  block/escalate
```

```text
violation_upper_bound_factor = 1 - epsilon_miss
```

- 实现文件：
  - `aimai_ocl/controllers/risk_gate.py`
  - `aimai_ocl/controllers/ocl_controller.py`
  - `aimai_ocl/plugin_registry.py`

### C. 责任归因与激励（Attribution）

- 算法 ID：`shapley_v1_exact / shapley_v1_reward_only / counterfactual_v1`
- 公式：

```text
V(trace) = ws*success + wr*seller_reward + wg*global_score
         - wv*violations - we*escalations - wt*rounds
```

```text
phi_i = sum_{S subseteq N\\{i}} [|S|!(n-|S|-1)!/n!] * (V(S union {i}) - V(S))
w_i = max(phi_i, 0) / sum_j max(phi_j, 0)
```

- 实现文件：
  - `aimai_ocl/attribution_shapley.py`
  - `aimai_ocl/attribution_counterfactual.py`
  - `aimai_ocl/plugin_registry.py`

### C.x 阶段策略（Attribution）

- `Phase A（当前）`：post-hoc attribution only（不进入在线协作决策）。
- `Phase B（TODO）`：credit feedback loop（跨 episode 回流到 OCL 控制参数）。

1. [x] 冻结 baseline 运行配置
- 固定 `env_id / model / max_rounds / seed`。
- 固化 demo/batch 输出字段，保证横向对比可复现。

2. [x] 定义 OCL 决策契约（Decision API）
- 统一决策类型：`approve / rewrite / block / escalate`。
- 统一 violation taxonomy（预算越界、底价越界、越权、隐私等）。

3. [x] 实现硬约束引擎 v1（ConstraintEngine）
- 先用确定性规则，作为风险门控前的硬边界。
- 将检查结果接入 `OCLController` 并进入审计轨迹。

4. [x] 补齐测试地基
- 规则单测、控制契约单测、buyer/seller 隔离单测、集成单测。
- 保障后续算法替换时主流程不回归。

5. [x] 批量评测脚本 v1
- 一键跑 `single / ocl_full`。
- 输出 `success/violation/round/reward/latency` 到 CSV/JSON。

6. [x] 角色分工算法模块化（Role Decomposition）
- 角色算法 registry：`role_v1_rule`、`role_v1_seller_only`、`role_v2_state_machine`。
- coordinator 通过可替换算法实例驱动，不和 runner 硬编码耦合。

7. [x] 风险门控算法模块化（Risk Gating）
- 风险门控 registry：`gate_v1_default / gate_v1_strict / gate_v1_lenient / gate_v2_barrier`。
- OCL 控制器从 registry 注入，支持公平对照与消融。

8. [x] 归因与激励算法模块化（Attribution）
- 归因算法 registry：`shapley_v1_exact / shapley_v1_reward_only / counterfactual_v1`。
- 保持四个接口稳定：
  - `run_episode(role_mask=S, seed=...) -> trace`
  - `compute_V(trace) -> float`
  - `fallback_policy(role, state) -> action`
  - `compute_shapley({V(S)}) -> {phi_i, w_i}`
- Shapley 算法实现位置：
  - `aimai_ocl/attribution_shapley.py`（`shapley_v1_exact / shapley_v1_reward_only`）
  - `aimai_ocl/attribution_counterfactual.py`（稀疏 coalition 时的 MC Shapley 估计）

9. [x] 统一插件接线（Bundle + CLI + Trace）
- 支持 bundle 与组件级 override 共存：
  - `algorithm_bundle`
  - `role/gate/escalation/attribution` 单独 override
- `run_demo` / `run_batch_eval` 的结果 JSON/CSV 全部透传并记录算法 ID（trace 仅保留必要运行元信息）。

10. [ ] 论文实验全套（可替换协议）
- [x] 协议接口模块化：`ExperimentProtocolBundle` + registry。
- [x] harness 输出 `protocol_outputs.json`，预留主结果/消融/对抗/重复/ROI。
- [ ] 主结果：`single / ocl-multi`。
- [ ] 消融（Phase A）：`w/o role / gate / audit / escalation`。
- [ ] Attribution（Phase A）：仅 post-hoc 质量评测，不计入在线协作消融。
- [ ] 对抗实验 + 重复交互实验 + ROI 显著性分析。

10.1 [ ] Phase B：Attribution 回流闭环（不训练）
- [ ] 增加跨 episode `credit_state` 存储与更新。
- [ ] 将 credit 映射到 coordinator/gate/escalation 控制参数。
- [ ] 增加 `ocl_credit` vs `ocl_static` vs `ocl_credit_random` 对照组。

11. [ ] 多商品导购迁移计划（Deferred，不影响当前单商品实验）
- [ ] M0：多商品输入 + 单商品成交（推荐先做）。
- [ ] M0.1：配置层支持 `products` 列表输入，保留 `product_info` 向后兼容。
- [ ] M0.2：scenario 校验从“必须单商品”改为“支持多商品候选，但本轮只允许一个 `selected_product` 进入议价”。
- [ ] M0.3：runner 增加 `selection_stage`（导购选品）并把 `selected_product` 注入 seller state / risk gate state。
- [ ] M0.4：指标补充 `selection_success`、`selected_product_id`，不改变主指标口径（success/violation/round/reward）。
- [ ] M0.5：Ablation 保持可比：single/ocl 两组都走同一选品输入协议。
- [ ] M1：同一 episode 内真实多商品博弈（后续）。
- [ ] M1.1：允许回合内切换商品、跨商品比较、bundle 报价。
- [ ] M1.2：约束与风险门控升级为“按商品上下文”的检查（价格边界、承诺一致性、政策约束）。
- [ ] M1.3：归因从单轨迹扩展到“选品贡献 + 议价贡献”的分解（Shapley/counterfactual 双层）。
- [ ] M1.4：新增多商品专项评测协议（selection quality、cross-item consistency、multi-item ROI）。

### 11.x 接口/类型改动（未来实现时）

- [ ] `RunConfig` 增加 `products: list[dict] | None`，保留 `product_name/product_price` 兼容旧脚本。
- [ ] `reset_kwargs` 统一接受 `product_info` 或 `products`，内部规范化为 `products + selected_product`。
- [ ] `scenario_validation` 从 `enforce_single_product_scenario` 升级为 `normalize_product_scenario(mode)`；默认 `mode=single_product`。
- [ ] `risk_gate/constraint_engine` 输入状态增加 `selected_product` 与 `catalog_context`，规则按当前选中商品判定。
- [ ] `trace/final_metrics` 增加 `selected_product_id`、`selection_stage_rounds`、`selection_success`。
- [ ] `experiment_protocol` 增加多商品扩展指标槽位，但默认协议不启用，保证现有结果可复现。

### 11.x 测试与验收（未来实现时必须满足）

- [ ] 向后兼容：旧单商品脚本和 3 个 arm 跑法不改命令即可通过。
- [ ] 配置校验：`products` 多于 1 时在 `M0` 不报错，但必须落地一个 `selected_product` 才进入议价。
- [ ] 行为一致性：single/ocl 在相同 `products` 输入下共享同一选品输入协议。
- [ ] 风险规则：越界价格、越权承诺、敏感信息检查在多商品上下文下继续有效。
- [ ] 评测输出：CSV/JSON 新字段存在且不破坏旧字段解析。
- [ ] 回归测试：当前 53 个测试全部通过，再新增多商品专项测试组。

### 11.x Assumptions & Defaults

- [ ] 默认仍是单商品模式；多商品功能默认关闭。
- [ ] 推荐先做 `M0`，不直接做 `M1`。
- [ ] `M0` 中一次会话只允许一个 `selected_product` 进入议价，避免一次引入过多变量。
- [ ] 论文主实验在 `M0` 前不改口径，先保证 single/ocl 主线稳定。
