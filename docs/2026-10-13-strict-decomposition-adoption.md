# 2026-10-13 — 正式采用合规 reference 与统一分解方法规则

**后续题面构建与 judge 自检：** 九类题面现由各自 `b_problem.py` 直接构建方法要求，已去掉 runtime 文本修补并同步 canonical 的路径说明；81 份生成／反思正文与此前 Docker 快照逐字一致。最新 reference 的真实 Docker judge 自检为 9/9 PASS，详见 [instance prompts and reference judge](2026-10-13-instance-prompts-and-reference-judge.md)。本文下述 v2 快照作为本轮采用时的历史证据保留，准备计划现指向 v3 题面构建版本。

用户在确认 reference 的独立校准结果后，授权“你去实现上面说的这些”。本次实现把 BP/GAP/TSSO 的校准方案纳入正式 reference，统一九类题的生成、反思和 judge 方法边界，并为后续新 cohort 准备独立协议。模型实验继续暂停在原有 31 对，本次授权只涉及实现和离线 reference 验证。

**最终正式版本通过 81/81：BD 45/45、CG 36/36。** 全部使用原有 native 环境、60 秒进程预算、单线程和 1% 目标阈值；最慢 **27.25 秒**，最大目标差 **0.1363%**。源码、输入、输出、环境、线程、清理与数值证书的 artifact audit 均通过。原数据及旧 cohort 的 **8,451 个文件哈希全部未变**，模型 controller、hourly review 和 Docker worker 均未启动。

## Reference 的正式修改

| Family | 正式实现 |
| --- | --- |
| UFL、FCTP、KM、LRP | 保留原分解算法，所有模型接入同一计时预算；master 的 incumbent objective 与 certified bound 分开记录；完整 recourse 结束后才提交匹配的 incumbent。LRP 的固定 depot 整数 recourse 仍允许。 |
| TSSO | 保留原 Benders cuts 和场景 recourse；使用自身上下界的 0.5% gap 目标，保存完整 incumbent，在预算结束时返回此前完整可行解。 |
| BP | 单箱 knapsack pricing 使用 solution pool 积累候选列，最后求 integer restricted master；在提交结果前删除冗余箱子并同步目标值、列使用量和输出。 |
| GAP | 删除完整 assignment feasibility MIP 初始化；以空列、singleton columns、人工变量和逐机器 pricing 完成 phase-I，再用真实成本做 CG，最后只求累计列上的 integer restricted master。 |
| CSP、SHIFT | 保留原 CG，统一预算与求解记录；定价结束后的 LP 下界考虑 reduced-cost 数值容差。 |

共享预算实现见 [reference_runtime.py](../src/utils/reference_runtime.py)、[BD driver](../src/utils/base_bd_class.py)、[CG driver](../src/utils/base_cg_class.py)。所有 model 的 Threads 都设为 1；每次 optimize 都按同一墙钟的剩余时间收紧 TimeLimit。默认 60 秒总预算中保留 6 秒给结果整理与输出；CG 在 LP/pricing 阶段另留 5 秒给最终整数 master。没有为每次子问题重新分配一份 60 秒预算。

`run()` 的时间覆盖加载、初始化、全部分解阶段、最终整数求解和结果构造；Python API 另把 reference 导入和 output 转换计入 `total_time`。完整 solver 进程的执行时间由 native executor 独立记录，包含导入与写 output.json，作为本次 60 秒验收的主要时间指标。保留时间是工程余量，外层仍执行原有 60 秒进程限制。

输出区分 `status=feasible`、`target_gap_met` 与 `proven_optimal`，数值最优容差单列为 `1e-8`。达到 TSSO 的 0.5% 内部目标或 benchmark 的 1% 外部阈值都不自动等于证明全局最优。CG 只有完成所有 pricing 后才使用与之配对的 LP 下界；最终整数 restricted master 的最优性不会独自变成完整问题的最优性。超过预算时，不完整场景或未完成 pricing 不会覆盖先前有效记录；如果没有完整可行 incumbent，则明确失败，不把 master relaxation 当成答案。

## Agent 与 judge 的共同要求

九类题的 renderer、generation runner 和 reflection 使用同一份 [decomposition_policy.py](../src/agent/decomposition_policy.py)；[algorithm_verifier.py](../src/agent/algorithm_verifier.py) 直接引用相同规则，并只插入所需的 BD 或 CG 方法段落。canonical 与 Docker 工作树中的相关 reference 和 prompt 实现已同步；未来 Docker/native cohort 的执行入口仍是 Docker 工作树。

- 禁止独立求解完整原始模型来做初始化、初始 incumbent、warm start、fallback、最终答案或 optimality certificate；零目标的完整可行性模型同样禁止。该规则覆盖源代码中可达的条件 fallback，生成端和 judge 端一致。
- 允许启发式初始化、直接构造 seed columns、带人工变量和 pricing 的 phase-I、有效初始 bounds/cuts，以及对合法 decomposition master 的 warm start。master、固定决策后的 LP/MIP 子问题、pricing 和最终 integer restricted master 都允许。
- BD 允许 classical、integer 和 logic-based variants，不强求 LP dual。CG 要有 restricted-master/dual/pricing 的信息反馈；不要求复现 reference 的 pricing pool、特定 cuts 或具体实现。
- 不规定最少迭代次数，不要求每道题实际增加 cut/column 或改善初始 incumbent，允许合法的立即收敛或停止。
- judge 追踪入口、可达调用、反馈与 output 来源；不会仅凭函数名称判定，也不会仅因未调用 helper 或不可达代码判 FAIL。条件分支只根据源码说明触发条件，不声称观察过运行。
- judge 评估是否实质实现并使用分解算法，不承担逐条 cut、bound、pricing 的完整数学证明；执行、可行性、目标一致性和目标质量仍由其他检查负责。

FCTP 的 `# or "monolithic"` 已从未来模板及其对应 output contract 中删除，`x` 改为与报告的 incumbent `y` 和 objective 对应。所有生成模板和 runner 明确允许 NumPy，使依赖条件与 reference 相同。

## 版本、快照与新实验边界

| 项目 | 新标识 |
| --- | --- |
| Reference | `strict-calibrated-references-20261013-v2` |
| 共同方法规则 | `strict-decomposition-20261013-v2` |
| 题面 | `strict-decomposition-prompts-20261013-v2` |
| Judge | `strict-decomposition-source-review-v2` |
| Output contract | `strict-decomposition-output-contract-v2` |

[协议快照 manifest](../outputs/diagnostics/strict_reference_adoption_20261013/protocol_final_v2/manifest.json) 保存 31 个 source 文件的哈希与副本、全部 81 个实例的实际渲染题面／生成／反思 prompt、BD/CG judge 全文和哈希。snapshot SHA-256 为 `5275aae48fced58c3f95101774da688fcdb6d6aa54536de26831d599951f95ed`。这里的 generation/reflection 文件是客户端补入 workspace delivery 指令前的正文；既有 workspace 指令保持原版本。

[新计划](2026-10-13-strict-decomposition-experiment-plan.json) 使用独立名称 `fixed81_strict_decomposition_20261013` 与独立 proposed root；仍为九个模型、729 对、原有 81 个固定输入、60 秒、单线程、1% 目标阈值和原有配对设置。GPT 与 Claude 的 effort 保持 medium，原有 Gemini 配置保持不变。计划已通过 `campaign_plan` 校验，状态是 prepared only，需要用户明确开始新实验后才会初始化或执行。没有创建生产 campaign，没有调用被测模型或 judge。

原 cohort `fixed81_qualitative_20261012_r1` 的 31 对结果、冻结 controller、dataset 和原判词全部保留，继续按其原协议解释。新 controller 会拒绝旧 prompt plan，旧 frozen controller 则继续保存其原代码；不会把新 prompt 覆盖到未完成的旧 cohort。Sol 的既有 cohort eligibility 规则没有借本次修改扩大到历史结果；新 cohort 真正建立后仍需以其实际冻结身份登记。

## 验证证据与复现

最终正式版本的机器可读记录见 [audited-summary.json](../outputs/diagnostics/strict_reference_adoption_20261013/formal_final_v2/audited-summary.json)，全部 81 道题均通过。单个 case 保存正式 source inventory、打包后的 solver、native execution receipt、output 与独立 verifier 结果。统计时间取完整 native 进程，不取 reference 的局部迭代计时。

| Family | 通过 | 最慢完整执行（秒） | 最大目标差 |
| --- | ---: | ---: | ---: |
| UFL | 9/9 | 4.23 | 约 0 |
| FCTP | 9/9 | 8.01 | 约 0 |
| KM | 9/9 | 6.86 | 约 0 |
| LRP | 9/9 | 27.25 | 约 0 |
| TSSO | 9/9 | 20.97 | 0.1362% |
| BP | 9/9 | 0.54 | 约 0 |
| CSP | 9/9 | 0.40 | 约 0 |
| GAP | 9/9 | 0.36 | 约 0 |
| SHIFT | 9/9 | 0.34 | 约 0 |

这验证了当前固定题在相同执行条件下的可达性，不把全部达标等同于全部证明精确最优。正式输出中 54 个 case 标记 `optimal`、27 个标记 `feasible`；后者也可能恰好达到外部 gold，但其算法内部上下界尚未闭合到数值最优容差。

第一轮正式集成的 81/81 结果保留在 [preliminary audit](../outputs/diagnostics/strict_reference_adoption_20261013/formal_full_v2/audited-summary.json)。其后补充的三个边界修正有独立反例测试，修正后的最终版本又完整跑了 81 道题。本次 reference 验证共 3 个 smoke case 和两轮完整复测；不与此前临时 variants 的校准合并计数。

[保全审计](../outputs/diagnostics/strict_reference_adoption_20261013/preservation-audit.json) 确认原数据和旧 campaign 的 8,451 个文件 SHA-256 全部一致，显式恢复 guard 仍为 true。最终资源监控正常退出，CPU 峰值为一核 100%，归属进程 RSS 峰值约 1.30 GiB；未触发 resource guard。

测试涵盖所有 81 道题实际渲染和反思继承、FCTP method schema、部分 BD recourse 中断、部分 CG pricing 中断、时间预算缩减、最终整数 timeout incumbent、无可行解时的输出保护、上下界与 incumbent objective 分离，以及三个新增边界反例。Docker 工作树中的 308 项集成回归通过；最终修正后的 18 项 reference 边界回归及 canonical 的 184 项 reference／历史一致性检查通过，九类全部 81 份 prompt 的检查也已通过。这些测试集合有重叠，不相加为独立测试总数。相关测试和准备计划日志保存在 [verification](../outputs/diagnostics/strict_reference_adoption_20261013/verification/)；过程中的三项 synthetic timeout 测试需要系统进程可见性，使用相应权限重跑后通过。

以下命令从 canonical repository 执行。输出目录要求新建，既有校准证据在源码变化时拒绝覆盖；native execution 和 resource guard 使用本机已固定的执行环境。

```bash
.venv/bin/python scripts/reference_calibration.py --output outputs/diagnostics/strict_reference_adoption_20261013/formal_final_v2 --variant formal-v2
.venv/bin/python scripts/audit_reference_calibration.py outputs/diagnostics/strict_reference_adoption_20261013/formal_final_v2 --preservation-inventory outputs/diagnostics/strict_reference_adoption_20261013/prechange_inventory.json
```

本次实际完整运行另由 `src.agent.resource_monitor.supervise` 包裹，CPU、RSS、memory 与 thermal 记录保存在 [monitor_formal_final_v2](../outputs/diagnostics/strict_reference_adoption_20261013/monitor_formal_final_v2/)。长于 5 秒的参考执行之后保留 10 秒 cooling。此前校准脚本中的临时 variant 没有替代最终验证：`formal-v2` 只打包正式 source、移除未使用绘图代码并观察求解调用，不接管正式算法或修改其求解参数。
