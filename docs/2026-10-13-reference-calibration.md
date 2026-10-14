# 2026-10-13 — Benders / Column Generation reference 的 60 秒校准

**后续采用已完成：** 用户随后明确授权实施。正式 reference、agent/judge 规则和最终 81/81 复测见 [strict decomposition adoption](2026-10-13-strict-decomposition-adoption.md)。下文保留采纳前的独立校准记录；其中“尚未替换”等表述指该轮校准当时的状态。

用户要求检查作者 reference 是否遵守分解方法限制，以及固定实例是否能由合理的 Benders/CG 实现在 60 秒内达到 benchmark 的数值要求。本轮独立校准得到 **81/81 个合规、达标的 case–implementation 结果：BD 45/45、CG 36/36**。最慢的选用结果为 **27.14 秒**，最大目标差为 **0.1363%**，低于原有 1% 阈值。原有 reference 并非全部原样满足要求：GAP 的初始化使用完整分配可行性 MIP，BP 的列集在部分实例上不足以得到达标整数解，原始 TSSO 在两道大实例上触及 60 秒。校准采用原版加按 problem family 固定的独立变体，没有更改固定输入、目标阈值、已有 reference 文件、正式 prompt 或 agent 实验结果。

机器可读总表与 81 个结果索引见 [calibration-summary.json](../outputs/diagnostics/reference_calibration_20261013/full/calibration-summary.json)。实验 campaign 继续暂停在 31 对；本轮没有调用任何被测模型或 algorithm judge，也没有恢复 controller 或 hourly schedule。

## 方法和验收边界

输入来自暂停的新 cohort 的全部 81 道固定题，每类 9 道，覆盖 small/medium/large 与三个 seed。它们与 canonical `data/` 中对应输入的 SHA-256 全部一致。每次参考运行都经过该 cohort 自己的 frozen `NativeSandboxExecutor`，使用原有 macOS native sandbox、Python 3.12.11、Gurobi/gurobipy 11.0.3、60 秒进程执行限制、单线程求解、网络和 reference 文件隔离。每个 Gurobi model 的实际 Threads 字段均记录为 1；外层 resource guard 监测 CPU、RSS、可用内存和 thermal，较长的执行后保留 10 秒 cooling。

报告中的时间为 executor 的 `execution_seconds`，覆盖 solver 导入、初始化、分解迭代、最终求解和输出，并包含 executor 的少量准备/清理开销。正常返回、保存输出通过 frozen `OutputVerifier` 的结构/可行性/目标重算检查，且 `abs(objective - gold) / max(1, abs(gold)) <= 0.01`，才记为数值达标。方法合规性通过源码和求解调用路径检查，不要求 reference 的每一条数学推导得到形式化证明。保存的输出与 reference objective 仅由沙箱外的 evaluator 比较；solver 只读 `data.json`，没有获得 gold 或预存解。

参考实现被打包到每个 case 的独立 `solver.py`，保留求解算法、输入类和输出转换，移除未使用的 plotting 导入/方法。原版只强制线程数，不改变其算法和停止规则。参考代码仍沿用现有 NumPy 矩阵接口；本轮未安装新依赖。此次校准验证的是分解方法在这些数值环境和预算下的可达性，NumPy 这一参考实现依赖也明确保留在记录中。

每个 case 目录保存 `solver.py`、原 source 哈希、输入副本、完整 native execution receipt、输出、日志与 `calibration.json`。最终复核确认选用的 81 个输入、代码、输出哈希和 native 环境均吻合，执行清理均成功，无 input integrity violation。缓存恢复会检查输入、生成代码和已保存输出的哈希；源码变化时要求使用新 variant 或新目录，不覆盖旧证据。

## 原始 reference 和校准版本的结果

| Family | 原版本次达标 | 选用版本 | 选用版本达标 | 最慢执行（秒） | 最大目标差 |
| --- | --- | --- | --- | ---: | ---: |
| UFL | 9/9 | 原 Benders | 9/9 | 4.22 | 数值精度范围内为 0 |
| FCTP | 9/9 | 原 Benders | 9/9 | 8.39 | 数值精度范围内为 0 |
| KM | 9/9 | 原 Benders | 9/9 | 6.72 | 数值精度范围内为 0 |
| LRP | 9/9 | 原 Benders，含固定 depot 决策的整数 recourse | 9/9 | 27.14 | 0 |
| TSSO | 7/9 | 保留 incumbent、按预算和 gap 停止的 Benders | 9/9 | 21.67 | 0.1363% |
| BP | 5/9 | pricing solution pool 扩充列集 + integer restricted master | 9/9 | 0.53 | 0 |
| CSP | 9/9 | 原 CG | 9/9 | 0.42 | 0 |
| GAP | 未纳入严格原版运行：初始化不合规 | phase-I restricted master + pricing + integer restricted master | 9/9 | 0.41 | 数值精度范围内为 0 |
| SHIFT | 9/9 | 原 CG | 9/9 | 0.60 | 0 |

完整正式校准执行了 99 个 case–variant 组合：72 个原版（跳过初始化不符合新规则的 GAP）、18 个 BP/GAP 变体、9 个 TSSO 变体；另有四个独立 smoke probes。原版总计 66/72 达标：TSSO large seed000/001 两次达到外部 60 秒限制，BP 有四个实例的整数目标未达到 1% 阈值。旧保存的 BP reference 有五个实例未达标，本次统一单线程下是四个，因此报告区分历史记录和本次实测，不把它们混为同一次实验。

上述九类选用实现按 family 固定；没有针对某道题硬编码答案或目标值。原版和变体使用的 gold 都只存在于外部验收程序。这里的 81/81 是本机本轮可达性证据，不意味着这些实现是每类问题最强的算法，或对任意更大实例均能在同一预算完成。

## Reference 方法合规性的具体检查

五个 BD reference 没有独立求解完整原始模型来初始化或收尾。LRP 的 `_solve_integer_recourse` 只优化 route 变量，depot 决策已经固定为 `y_vals`；这是允许的整数子问题。其他 BD 也都保留 master 与固定决策 recourse 的信息反馈。

GAP 原版 `initial_patterns()` 中的 `gap_feasible_seed` 为所有任务—机器对建立二元 `x[i,j]`，加入完整分配与机器容量约束，再调用 `optimize()`，据此生成初始列。零目标改变了求解目的，但它仍是完整原始可行域上的 MIP 初始化。在严格禁止完整模型初始化的规则下，这个步骤需要替换，不能因为它只求可行性就当作普通手工 seed。

BP、CSP、SHIFT 的原版采用直接构造的简单初始 patterns；pricing 和最终 integer restricted master 都属于 CG 的正常部分。原 BP 结果的问题是最终 restricted column set 的整数质量，不是使用了完整原始 packing MIP。

## 独立变体做了什么

**BP：** 保留标准 restricted-master/dual/pricing 流程，将 pricing 中的单箱 knapsack 用 solution pool 产生多个候选 patterns，再在累计列上解整数 master。部分额外列的 reduced cost 可以非负，它们用于改善整数列集；没有另外建立完整 bin-packing MIP。九道题全部达到冻结 reference 的整数目标。

**GAP：** 用空模式、单任务模式和人工变量建立 phase-I restricted master，通过 master duals 驱动每台机器的 knapsack pricing，先消除人工变量，再切换到真实成本的 CG；最后只在累计 patterns 上求整数 master。严格入口没有调用继承类中的完整分配 MIP 初始化函数。源码和实际 model-solve 记录都确认最终执行路径为 phase-I/CG/pricing/restricted integer master。

**TSSO：** 保留原 Benders 求解和 cuts，用算法自身的上下界在 0.5% gap 时停止，并持续保存最佳 incumbent；设 54 秒内部截止，为 60 秒外部限制留出返回和写文件时间。若达到截止，则返回已经由分解得到的 incumbent。该内部判断不读取 gold。三道 large 实例本次分别约 21.35、21.67、17.62 秒，最大实际 objective gap 为 0.1363%。这满足当前 benchmark 的 1% 要求，未额外要求 60 秒内证明精确全局最优。

实现为 [reference_calibration.py](../scripts/reference_calibration.py) 和 [reference_calibration_variants.py](../scripts/reference_calibration_variants.py)。这些是独立校准工具；尚未将变体替换为公开 reference，也没有据此修改正式 prompt 或重新给已有 agent 输出打分。

## Benders 允许的初始化空间

严格限制保留了非完整模型求解的初始化方式。例如，按问题约束用贪心/规则构造初始 master 决策，固定该决策求 recourse 来检查可行性并计算上界，再以该解 warm-start master；也可加入基于问题数据推导的初始有效约束或下界。master 和固定决策后的 LP/MIP 子问题始终允许。禁止范围针对通过完整原始模型优化来取得初始化、fallback 或证书，不是禁止所有 heuristic initialization。

## 后续采用前的工作

这次结果支持保留 60 秒预算作为当前固定实例的可达目标。正式采用前，应审阅并选择 reference 版本，同步 generator/reflection/judge 的方法边界，明确依赖与时间口径，再冻结相应的新协议。原始 31 对 agent 结果和所有 reference 历史记录继续保存。若需要发表“稳定运行时间”或比较求解速度，应另做重复计时；本轮针对的是能否在预算内达到验收质量。
