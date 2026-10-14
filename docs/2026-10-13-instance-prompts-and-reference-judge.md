# 2026-10-13 — Instance 题面构建与 reference 的真实 judge 自检

用户要求澄清 Docker 编辑上下文与未来 solver 文件路径的区别，指出方法规则应由各类 problem builder 直接构建，并要求用最新 judge 真正评审最新版 reference。本轮已完成题面构建调整、两个入口的生成正文一致性检查，以及九类 reference 的真实 judge 调用，结果为 **9/9 PASS**。本轮没有重新执行 solver，也没有恢复原有模型 campaign。

## 编辑 context 与 runtime 路径

Docker 的 `DockerCliClient.edit_file` 已将任务正文及 context 文件内容直接放进模型输入。生成阶段的 context 包含 `data.json`；agent 在 `/work` 编辑现有 `solver.py`。随后 harness 在独立 native runtime 中执行这份代码，`../../data.json` 指的是该执行目录相对实例数据的位置，`output.json` 则写在执行目录。它不是让 Docker 编辑阶段去上两级寻找数据。

Canonical 的 [prompt_builder.py](../src/agent/prompt_builder.py) 已同步成相同的编辑／执行阶段说明，删除“读取 `../../prompt.txt`”的旧指令。两个入口的生成正文现在一致；各客户端仍按自己的实际 workspace 添加编辑目标说明。

## 方法规则由 problem template 直接构建

九个 `b_problem.py` 的 `BASE_PROMPT_*` 都直接声明 `{method_requirements}`，各自的 `render_prompt(data)` 同时填入本实例的数据描述和所属 BD/CG 的方法要求，生成完整题面。规则文本仍在 [decomposition_policy.py](../src/agent/decomposition_policy.py) 共用，供各模板和 judge 使用；其中已删除 `ensure_method_requirements`，没有对旧题面做 `.replace()` 或检查标题后补文字的逻辑。

生成与反思 runner 读取 instance 已保存的题面，补充执行路径、依赖与预算，不再增改算法要求。新发布的数据或新 cohort 应在构建 instance 时调用当前 `render_prompt(data)`，将结果保存为该 instance 的 `prompt.txt`；已有冻结数据保留其原题面。Docker 新 cohort 的 `snapshot_prompt` 正是在准备 dataset 时从每个 `data.json` 调用该 renderer，保存新题面和原题面副本，不是在运行时改写旧文本。

可直接查看 [FCTP 模板](../src/bd/fctp/b_problem.py)、[GAP 模板](../src/cg/gap/b_problem.py)，以及新版按实例生成的 [FCTP generation](../outputs/diagnostics/instance_prompt_refactor_20261013/protocol_instance_templates_v3/prompts/bd/fctp/fctp_medium_seed000/generation.txt) 和 [GAP generation](../outputs/diagnostics/instance_prompt_refactor_20261013/protocol_instance_templates_v3/prompts/cg/gap/gap_medium_seed000/generation.txt)。

题面构建版本更新为 `instance-bound-decomposition-prompts-20261013-v3`；共同方法契约、judge 和 output contract 保持 v2。[准备计划](2026-10-13-strict-decomposition-experiment-plan.json) 已指向该构建版本，仍为 prepared only，没有创建生产 cohort。

[完整快照](../outputs/diagnostics/instance_prompt_refactor_20261013/protocol_instance_templates_v3/manifest.json) 的 SHA-256 是 `95b5afd3ef88927ff65498306387ff294b6cc88abb80a0fff10d14d0a802e82a`。对全部 81 个实例逐字核对后，problem、generation、reflection 正文均与前一版 Docker 快照相同，BD/CG judge 文字也相同；canonical 的生成与反思正文现在也与 Docker 对齐。此次改变的是题面的构建位置与 canonical 的路径说明，没有改变此前 Docker 题面的方法要求。结果见 [prompt_equivalence.json](../outputs/diagnostics/instance_prompt_refactor_20261013/prompt_equivalence.json)。

## Reference 的真实 judge 结果

使用准备计划中的 **GPT-6 Astra、high**，经相同 Docker image、CLI、`DockerCliClient`、`algorithm_review_prompt(suite)` 和 `algorithm_judge` role 发起调用。输入使用上一轮 81/81 正式验证实际执行过的完整 `solver.py`，包含基类、具体算法、预算控制、输出转换和入口；没有只提交一个未被调用的算法类。每类九个实例的 solver 源码 SHA-256 完全一致，因此评审九个独立源码，覆盖 81 个固定实例使用的实现。

Judge 只获得当前 judge prompt 和该 `solver.py`，未提供 gold、数值达标标记或先前判词。每类只调用一次，没有改变 judge 文字来获得 PASS，也没有从多次 verdict 中挑选最好结果。以下时间与 tokens 来自 CLI 的原始 metrics；cached input 是 input 的子项。

| Family | Verdict | Agent 秒数 | Input tokens | Cached input | Output tokens | 证据 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FCTP | PASS | 26.31 | 69,005 | 45,184 | 599 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/bd/fctp/alg_verifier.md) |
| KM | PASS | 21.20 | 65,934 | 43,136 | 556 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/bd/km/alg_verifier.md) |
| LRP | PASS | 23.15 | 72,183 | 47,488 | 610 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/bd/lrp/alg_verifier.md) |
| TSSO | PASS | 24.72 | 69,308 | 45,440 | 570 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/bd/tsso/alg_verifier.md) |
| UFL | PASS | 21.87 | 65,697 | 52,736 | 556 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/bd/ufl/alg_verifier.md) |
| BP | PASS | 26.41 | 63,606 | 51,200 | 577 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/cg/bp/alg_verifier.md) |
| CSP | PASS | 20.54 | 62,615 | 50,688 | 537 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/cg/csp/alg_verifier.md) |
| GAP | PASS | 25.65 | 67,507 | 44,160 | 581 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/cg/gap/alg_verifier.md) |
| SHIFT | PASS | 24.81 | 61,499 | 49,920 | 601 | [判词](../outputs/diagnostics/reference_judge_20261013_v2/cg/shift/alg_verifier.md) |

九次调用累计 agent 时间 **214.67 秒**；CLI 报告 input tokens **597,354**，其中 cached input **429,952**，output tokens **5,187**，另保存 `reasoning_output_tokens=361`。完整 invocation、effective prompt、events、worker metadata、容器清理记录、时间与 token 用量均在各 family 目录保存。[summary.json](../outputs/diagnostics/reference_judge_20261013_v2/summary.json) 汇总原始 verdict 和 metrics，[audit.json](../outputs/diagnostics/reference_judge_20261013_v2/audit.json) 核对了输入、判词哈希、请求模型／effort、实际发送的上下文及清理结果。

判词具体识别了 BD 的 master／固定决策 recourse／cuts 反馈；其中 LRP 的整数 route recourse 被视为固定 depot 决策后的合法子问题。CG 判词识别了 master duals、pricing 与列反馈，并明确接受 BP pricing solution pool、GAP 人工变量 phase-I，以及各类最终 integer restricted master。SHIFT 枚举候选 shift 用于 pricing 也获得认可。这与本次方法边界一致。

该结果回答的是“最新版参考实现在当前 judge 下能否通过”：本轮九类均通过。数值可行性与 1% 目标质量仍由此前完整的 [81 道正式 reference 验证](2026-10-13-strict-decomposition-adoption.md) 提供证据，judge 自检不代替该数值检查。

## 验证与保全

Canonical 的题面／反思回归为 88 项通过；Docker 的题面、快照、客户端、反馈及计划回归为 125 项通过。新增的 instance 题面完整性断言进一步确认，在进入 runner 之前，保存的题面就已经具有完整方法要求；该组 82 项通过。这些测试集合有重叠，不相加为独立总数。

[保全审计](../outputs/diagnostics/instance_prompt_refactor_20261013/preservation-audit.json) 再次确认原数据与旧 cohort 的 8,451 个受保护文件全部未变。原有 31 对实验继续暂停，显式恢复 guard 保留；本轮用户只授权这九次 reference judge 自检，未恢复被测模型的生成队列或 hourly schedule。

自检脚本为 [judge_references.py](../scripts/judge_references.py)。它记录稳定的源码／prompt 身份，复用已有 verdict 或恢复已有容器句柄，避免重复调用；更换输入或配置时要求使用新目录保留旧证据。此次题面修改同步到了 canonical 与 Docker 工作树，仍未 commit 或 merge。
