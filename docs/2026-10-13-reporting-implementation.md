# October 13 reporting implementation and first saved snapshots

用户在 October 13 明确授权：先运行 GPT 队列，同时由独立 agent 处理 review 第 9、10、14 条的统计提取和报告；报告侧可以读取正在产生的记录，与模型执行隔离。本次实现放在 canonical `DecompBench`，读取已冻结的 native/hybrid campaign，派生产物写入 `outputs/reports/` 或 `outputs/analysis/`。实验 controller、prompt、评分、原始输出及 frozen controller 保持原文件。本次没有发出模型调用、运行候选 solver 或改变实验状态。

## 已实现的报告路径

`src.05_report` 识别 `campaign.json` 后使用新的 [campaign_report.py](../src/campaign_report.py)，继续保留原 `run_config.json` 布局的 legacy exporter。新路径读取冻结 inventory、模型队列、reference、case identity、共享首轮、score/CLI receipts、完整 `calls/<invocation_id>/` 目录及 solver execution。每个文件只取一个读取版本，保存读取窗口，并在结束时检测文件替换；变化中或读取/审计失败的配对从成功统计分母排除。在下一次快照中重新读取，不等待或锁住 controller。

审计核对 frozen campaign/controller/dataset 哈希、各 backend 的独立调用证据、实际请求模型、Docker image/cleanup、reference objective、评分与实例汇总、执行及产物哈希、成功后的复用槽位和 reflection 机会。首轮绑定 canonical Plain 目录、相同的完整评分、实际 generation/judge invocation ID；Reflection 后追加的反思 metrics 与首轮 solver 评分分开核对。已验证的 output 通过该 campaign 自己的 frozen `OutputVerifier` 再检查；这个受控子进程仅验证保存的 JSON，不执行生成的 solver。历史 terminal receipt 继续使用已有的严格验证器，保留真实 attempts 和未执行槽位。

完整 policy 使用已有 `policy_is_complete` 的语义：完整预算槽位（成功后的槽位来自既有复用规则），或者证据完整的 terminal receipt。仅有 `complete=true` 和短成功前缀不能被当成已完成。完成的 model/case 配对进入统计；尚未结束的一侧不会被补成失败。`small`、`medium`、`large` 直接来自 frozen inventory；各层计划和完成分母分别求和核对整体。`src.03_analysis` 的分层标签已同步，native campaign 的显式入口及扫描发现都会走同一报告路径，将派生文件放在实验根目录之外。

## 指标与成本口径

主要终点是每个 attempt budget 下的累计 solveQ：通过执行/可行性和保存的 algorithm judge，并达到原协议的 objective-gap threshold。报告同时保留 solveE、solveV，以及 Plain-only、Reflection-only、both、neither 配对计数。两侧和全部预算使用相同的完整配对集合，按模型、suite、problem family 和 size 分组。每组以完整 case pair 进行 4,000 次可复现 bootstrap，保留两种策略及所有预算的对应关系，输出 Reflection − Plain 的 95% percentile 区间；少于两对时不输出区间，退化区间单独标记。区间描述一次保存运行下的 case variation；数据集 seed 是不同实例，不是同一 case 的多次模型重复。方法参照 [SciPy paired bootstrap 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。顺序执行和早期完成选择会影响未完成 campaign 的临时报表，因此所有未完成报告标记为 PROVISIONAL。

物理成本扫描全部实际调用目录，包括正常调用、未交付、被拒绝的 judge、失败重试、成功但尚未提交的返回以及中断记录。`invocation_id` 去重 generation/judge/reflection，`shared_first` 身份去重 solver execution；顶层投影与恢复复用不产生新成本。只创建 `request.json` 的记录可能还在等待 call gate，另列为 start-unconfirmed request，不冒充已经开始的物理调用。用户最终可见的成本口径以 proven invocation records 为准，同时显示尚未确认启动的请求数。

每种策略的成本包含共同首轮，用于比较同一策略的完整开销；实际实验总成本中共同首轮只计一次。每对另列 shared-first、Plain incremental、Reflection incremental、额外失败/恢复调用。`paired_cost_differences.csv` 按阶段和 backend 给出 token/time 差值；有缺失的两侧总数保留 null。solver 时间单独去重，客户端 generation/judge/reflection 的 elapsed 单独汇总。阶段累计工作时间和并行实验的实际经过时间分开；目前没有从不完整的暂停/等待/恢复日志推算 active wall time。

Token 字段按 backend 保留原名，包含 input/output、cache、reasoning/thinking 和 total 等；这些字段可能包含彼此，报告逐字段展示，不把它们全部相加。新 Docker adapter 的 `usage_available` 在 invocation 中，因此联合读取 invocation 与 metrics；旧记录从实际非空 usage 判断可用性。每个字段显示 observed sum、available calls、missing calls。没有上游 usage 的记录保存 null，零只代表上游明确返回的零或没有该类调用。费用价格换算留给明确的计价数据，不把 token 表伪装成美元账单。

## Cohort eligibility

[reporting_policy.py](../src/reporting_policy.py) 将历史 `gpt-5.6-sol` 排除保留为默认规则，对新授权 cohort 设置精确例外：protocol ID `e6a01d3d1cf966de676323a7d9f9e5b344552ee678281b46feb92823b4a65c73`，同时验证 manifest 身份、计划名、冻结 inventory、Docker editing/native Seatbelt execution 和不混入历史结果的标志。旧、未知或仅模仿目录名的 Sol cohort 继续排除。新报告完整分母为 9 模型 × 81 = 729 对；当前操作范围为 5 个 GPT、405 对，操作范围单列，科学分母保持冻结队列。历史 Astra 只出现在历史报告，未进入新 cohort。

## 本次验收与真实数据快照

专项和既有报告/terminal 回归结果为 **65 passed in 15.17s**，[JUnit XML](../outputs/reports/2026-10-13-reporting-regression.xml)。新增 22 个测试覆盖 Codex/Claude/AGY 的 native/Docker 调用、共享首轮提前成功、预算耗尽、reflection omission、失败重试、缺失 usage、request-only、未完成配对、被篡改的输出/物理 usage/shared ID、错误 completion flag、大小分母、paired bootstrap、terminal prefix、只读导出及新旧 Sol。对损坏共享/调用证据的测试同时核对 completed pair 和 policy coverage，避免仍保留已完成分母。

```sh
.venv/bin/python -m pytest -q tests/test_campaign_report.py tests/test_evaluation_report.py tests/test_reporting_policy.py tests/test_terminal_policy_completion.py --tb=short --junitxml=outputs/reports/2026-10-13-reporting-regression.xml
```

生产快照读取于 **2026-10-13 13:57:47 EDT**：[新实验报告](../outputs/reports/2026-10-13-fresh-draft-20261013T175747Z/report.md)，[完整 JSON](../outputs/reports/2026-10-13-fresh-draft-20261013T175747Z/evaluation.json)。当时 **7/729 对完成、2 对进行中、720 对未开始**；7 对全部属于首个 GPT 模型。记录 **26 个有物理调用证据的 invocation、2 个启动未确认请求、13 次不同 solver execution**。输入读取、全局和逐 policy 审计错误均为 0，读取期间变化文件为 0。该快照是固定的临时报告，之后的运行结果需要重新提取。

另对保存的历史 native campaign 完整走通读取和审计：[历史报告](../outputs/reports/2026-10-13-historical-draft-20261013T175747Z/report.md)。结果为 **489 个完成配对、981 个完成 policy、1,777 个调用记录、725 次不同 solver execution**，历史 Sol 被过滤，Gemini 的已有 terminal adjudication 正确保留 **1 次真实 attempt、9 个未执行槽位**。全局、读取和逐 policy 审计错误均为 0。历史报告与新报告分目录生成，统计不混合。机器可读摘要为 [snapshot evidence](../outputs/reports/2026-10-13-reporting-snapshot-evidence.json)。

## 后续使用

运行以下命令生成新快照；未指定 `--output` 时，native 报告默认写到带 UTC 时间戳的 `outputs/reports/`，不写进原始 campaign。正在运行时保留 `--allow-partial`；完整正式报告默认要求全部配对完成且审计通过。

```sh
.venv/bin/python -m src.05_report --input-root outputs/experiments/fixed81_qualitative_20261012_r1 --allow-partial --no-charts
```

输出包括 `report.md`、完整 `evaluation.json`、`paired_curves.csv`、`physical_calls.csv`、`case_costs.csv`、`paired_cost_differences.csv`、`solver_executions.csv` 和 `coverage_by_size.csv`。本轮完成可审核的表格和原始统计导出；publication figures、美元计价及完整暂停/等待 wall-time timeline 尚未生成。实验跑完后应重新运行完整报告，不把本次 7 对的临时快照作为全部新 cohort 的统计结论。既有 frozen dashboard 的实现仍来自原 controller；正式统计和 cohort 资格以此 canonical 报告命令为准。
