# 2026-10-13 — Commit 分组与单 worktree 整理

用户要求先把未提交代码按 logical unit 提交，再将多个 worktree 整理成一个，并明确旧 Docker 只迁移 experiments，不再保留其活动源码。本次最终保留 `code/DecompBench` 的 `main`，Docker V2 已合并进该目录；旧 Docker、V2、transfer 的工作目录及旧 Docker 兼容符号链接均已移除。旧 Docker 源码只保留在历史 Git 分支，没有合并进 main。

## 提交清单

| Commit | 分支／归属 | 主题 |
| --- | --- | --- |
| `c6619a5` | main | `fix(references): calibrate decomposition within shared budgets` |
| `1592a06` | main | `refactor(prompts): bind decomposition rules to instance templates` |
| `6c52f47` | main | `feat(reporting): audit paired campaign outcomes and physical costs` |
| `a26939b` | main | `feat(dashboard): show scoped fresh-cohort progress and reports` |
| `0b2faa0` | main | `docs(experiments): record calibrated protocol and reporting decisions` |
| `53dcc35` | V2，已随 merge 保留 | `fix(references): adopt calibrated BD and CG implementations` |
| `deec0b6` | V2，已随 merge 保留 | `refactor(prompts): freeze instance-owned strict decomposition inputs` |
| `fdceedc` | V2，已随 merge 保留 | `feat(controller): require explicit model scope and pause after drain` |
| `7a6b2b4` | 旧 Docker 历史分支 | `feat(legacy-runtime): archive isolated Docker and WLS execution` |
| `dfaaf29` | 旧 Docker 历史分支 | `feat(legacy-campaign): archive recovery and monitoring tools` |
| `4ee9eb0` | 旧 Docker 历史分支 | `feat(legacy-auth): archive isolated AGY credential handoff` |
| `da89cf6` | main merge | `merge: consolidate Docker V2 with canonical reporting` |

随后单独提交当前单目录入口、路径更新和本整理记录，主题为 `chore(workspace): document the single canonical checkout`。本次全部为本地提交，没有 push。

## 目录与历史归属

| 原目录 | 处理 |
| --- | --- |
| `DecompBench` | 保留为唯一 worktree，分支 main；原 `.venv`、数据、诊断和两轮冻结 campaign 保持原位置。 |
| `DecompBench-docker-v2` | `fdceedc` 及其所有祖先已合并进 main，随后移除 worktree。 |
| `DecompBench-docker-old-20261007` | 原有源码改动分三组提交，历史分支 `codex/docker-benchmark` 保留在 `4ee9eb0`；仅 experiments 迁入主目录，随后移除 worktree。 |
| `DecompBench-docker` | 原本就是指向旧 Docker worktree 的符号链接；链接已移除。 |
| `DecompBench-transfer` | 干净 worktree，位于 main 已包含的 `6883e15`；没有独有的未跟踪文件或实验结果，已移除。 |

历史文档及快照中的旧 worktree 路径保留为来源信息。需要恢复当时源码时，可按上述 Git commit 查阅；当前执行和维护入口统一使用 canonical `DecompBench`。旧目录中的 Python／pytest 缓存及被忽略的旧 `grbprobe` 工具随目录清理，不进入主目录。

## 旧 Docker 实验归档

旧实验原样保存在 [legacy_docker_20261007_archive](../outputs/experiments/legacy_docker_20261007_archive/README.md)。其 `experiments/` 包含原来的 `docker_gpt_smoke_20261007`、`fixed81_20261007` 和 `single`，共 **1,999 个文件、12,637,074 bytes**，复制前后逐文件 SHA-256 一致。

[archive.json](../outputs/experiments/legacy_docker_20261007_archive/archive.json) 标注原目录、历史 Git commit、`agent_execution=docker`、`solver_execution=docker-wls`，并提供旧路径到归档路径的映射及完整文件哈希。实验目录内部的冻结代码、数据、回执和日志均属于原始实验记录，保留原字节；没有将旧 worktree 的 `src/`、Docker 工具或 probe 二进制复制进当前源码。

GPT-5.6 Sol 的历史结果仍被现行 reporting policy 排除。归档的位置或 Docker 标签不会赋予新 cohort 的 eligibility。当前已授权的新 cohort 仍按其精确冻结 protocol identity 判断，旧 Docker 记录不会混入新实验分母。

## 合并后的功能与验证

主目录保留 V2 的 Docker 编辑、受限 native solver、输入隔离、严格 output contract、reflection 机会计数、错误分类和显式模型运行范围；同时保留 canonical 的 paired report、cohort eligibility 与新 dashboard。`decomposition_policy.py`、BD/CG 算法和当前 prompt 规则没有因目录整理而删改。

合并后的完整测试为 **719 passed、13 skipped**；13 项是需要显式开启的 Docker／native 集成测试。合并检查发现旧 reference 数值测试没有 candidate 的 `status/method` 字段，因此增加单独的 reference 数值验证入口，保留正式 agent 的严格 schema 检查；历史 gold 没有被添加虚假的 decomposition 标签。报告测试中的 synthetic candidate 输出则补齐其应有字段。

从 canonical 成功校验了准备计划、native Python/Gurobi 环境、Docker image/worker 文件哈希和三个 backend 的既有 readiness 证据，没有重建镜像或重新调用模型。V2 已提交且参与镜像哈希的 `agy_credentials.py` 末尾有既存空行；保留其原字节，其余合并文件通过 whitespace 检查。

移除 worktree 后，再次生成的原 cohort 报告仍为 **31/729 对完成、state=paused、无 audit error**；归档中的历史 Sol 继续被排除。整理前固定的 canonical 数据与 campaign 文件共 **71,754 个**，最终保全结果和运行验证见 [consolidation diagnostics](../outputs/diagnostics/worktree_consolidation_20261013/)。本次没有恢复任何实验、watchdog 或 hourly review。

## 以后从哪里操作

```sh
cd /Users/scheng326/Desktop/GT_ML_PhD/z_oo/code/DecompBench
.venv/bin/python -m src.campaign_plan --plan docs/2026-10-13-strict-decomposition-experiment-plan.json
```

该命令只校验准备计划。新实验仍需要用户明确授权；现有冻结 campaign 的 controller 与恢复规则保持原样。`judge_references.py`、`snapshot_strict_protocol.py` 和 `export_agent_input_walkthrough.py` 的 `--controller` 现在默认当前仓库，后续使用这些工具无需再指定已删除的 V2 worktree。

实验输出、诊断回执、生成的全过程输入原文和快照仍由既有 `.gitignore` 规则排除，全部保留在本地；本次提交的是源代码、测试、导出工具和人工撰写的计划／验证说明。
