"""Capture benchmark-owned agent inputs without starting Docker, models, or solvers."""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
FAMILIES = {"ufl": "bd", "fctp": "bd", "km": "bd", "lrp": "bd", "tsso": "bd", "bp": "cg", "csp": "cg", "gap": "cg", "shift": "cg"}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def placeholder(kind, attempt=None):
    suffix = f"_ATTEMPT_{attempt:02d}" if attempt is not None else ""
    return f"<<{kind}{suffix}>>"


class CapturedPrompt(Exception):
    def __init__(self, payload):
        self.payload = payload


class CaptureRuntime:
    """The real Docker client assembles its job; dispatch stops at this boundary."""

    def run(self, image, command, **kwargs):
        assert command[0] == "agent"
        assert kwargs["payload"]["credential"] is None
        raise CapturedPrompt(kwargs["payload"])


def no_execution(*args, **kwargs):
    raise AssertionError("Documentation capture must never execute a solver or external process")


def capture(operation):
    try:
        operation()
    except CapturedPrompt as captured:
        payload = dict(captured.payload)
        assert payload.pop("credential") is None
        assert payload["initial_content"] == ""
        return payload
    raise AssertionError("The real input path did not reach the capture boundary")


class Walkthrough:
    def __init__(self, controller, destination, work):
        self.controller, self.destination, self.work = controller, destination, work
        sys.path.insert(0, str(controller))
        from src.agent.docker_client import DockerCliClient, worker_file_hashes
        from src.agent.paired_runner import PairedRunner
        from src.agent.experiment_runner import AttemptResult
        from src.agent.exec_logger import ExecLogger
        from src.agent.algorithm_verifier import AlgorithmVerifier
        from src.agent.prompt_policy import OBJECTIVE_RETRY, reflection_delivery_feedback
        from src.prompt_snapshot import PROMPT_PROTOCOL

        self.Client, self.Runner, self.Result = DockerCliClient, PairedRunner, AttemptResult
        self.Logger, self.Verifier = ExecLogger, AlgorithmVerifier
        self.objective_retry = OBJECTIVE_RETRY
        self.delivery_feedback = reflection_delivery_feedback
        self.prompt_protocol = PROMPT_PROTOCOL
        self.plan_path = REPO / "docs/2026-10-13-strict-decomposition-experiment-plan.json"
        self.plan = json.loads(self.plan_path.read_text())
        assert self.plan["prompt_protocol"] == PROMPT_PROTOCOL
        self.environment = json.loads(Path(self.plan["runtime_manifest"]).read_text())
        assert self.environment["worker_files_sha256"] == worker_file_hashes()
        self.calls, self.documents, self.proofs = [], {}, []
        self.specs = {backend: next(item for item in self.plan["model_queue"] if item["backend"] == backend) for backend in ("codex-cli", "claude-code-cli", "agy-cli")}

    def client(self, backend, role="generation"):
        spec = self.specs[backend]
        model = self.plan["execution"]["judge_model"] if role == "algorithm_judge" else spec["model"]
        effort = self.plan["execution"]["judge_effort"] if role == "algorithm_judge" else spec["effort"]
        client = self.Client(model, backend=backend, image_id=self.environment["image_id"], runtime=CaptureRuntime(), evidence_root=self.work / "unused", effort=effort, timeout=900, role=role, credential_provider=lambda _: None)
        # Preflight and credentials are outside prompt construction; neither is invoked.
        client.prepare = lambda: {}
        return client

    def task(self, problem, backend, scenario):
        suite = FAMILIES[problem]
        case_name = f"{problem}_medium_seed000"
        source = self.work / "sources" / backend / scenario / case_name
        source.mkdir(parents=True, exist_ok=True)
        data = json.loads((REPO / "data" / suite / problem / case_name / "data.json").read_text())
        body = importlib.import_module(f"src.{suite}.{problem}.b_problem").render_prompt(data).rstrip() + "\n"
        (source / "data.json").write_text(placeholder("DATA_JSON") + "\n")
        (source / "prompt.txt").write_text(body)
        task = self.Runner(client=self.client(backend), reflection_client=self.client(backend, "reflection"), judge_client=self.client("codex-cli", "algorithm_judge"), solver_executor=no_execution, call_lock=lambda _: nullcontext(), model=self.specs[backend]["model"], suite=suite, problem=[problem], solver_timeout=60, solver_threads=1, max_attempts=10, protocol_id="documentation-capture-only")
        root = self.work / "cases" / backend / scenario / problem
        cases = {mode: task.prepare(source, root, mode) for mode in ("plain", "reflection")}
        assert task.generation_prompt(cases["plain"]) == task.generation_prompt(cases["reflection"])
        return task, cases, body

    def generation(self, task, case, attempt, mode):
        task.mode = mode
        directory = case / "attempts" / f"attempt_{attempt:02d}"
        return capture(lambda: task.run_attempt(attempt, directory, task.problems[0], None))

    def candidate_files(self, directory, attempt, kind="quality"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "solver.py").write_text(placeholder("SOLVER_PY", attempt) + "\n")
        verdict = "PASS" if kind == "quality" else "FAIL" if kind == "quality_and_algorithm" else placeholder("JUDGE_VERDICT", attempt)
        notes = f"VERDICT: {verdict}\nNOTES:\n{placeholder('JUDGE_NOTES', attempt)}\n"
        (directory / "alg_verifier.md").write_text(notes)
        lines = self.Logger.build_exec_lines(placeholder("SOLVER_STDOUT", attempt), placeholder("SOLVER_STDERR", attempt))
        if kind != "quality":
            self.Logger.append_output_message(lines, placeholder("OPTIONAL_OUTPUT_MESSAGE", attempt) if kind == "verification" else None)
            issues = ["algorithm_verifier_failed: fail"] if kind == "quality_and_algorithm" else [placeholder("VERIFICATION_ISSUES", attempt)]
            self.Logger.append_verification_issues(lines, issues)
        self.Logger.append_llm_verifier(lines, notes)
        self.Logger.write_exec_log(directory, lines)
        if kind != "execution":
            (directory / "output.json").write_text(placeholder("OUTPUT_JSON", attempt) + "\n")

    def reflection(self, task, case, attempt, kind="quality"):
        task.mode = "reflection"
        directory = case / "attempts" / f"attempt_{attempt:02d}"
        self.candidate_files(directory, attempt, kind)
        feasible = kind in {"quality", "quality_and_algorithm"}
        result = self.Result(attempt, "objective_gap_too_large" if feasible else "failed", kind != "execution", False, 2.0 if feasible else None, 1.0 if feasible else None, 1.0 if feasible else None, [], feasibility_ok=feasible)
        payload = capture(lambda: task.ensure_reflection(directory, result))
        assert (self.objective_retry in payload["prompt"]) == feasible
        (directory / "debug.md").write_text(placeholder("DEBUG_MD", attempt) + "\n")
        return payload

    def missing_generation(self, task, case):
        directory = case / "attempts/attempt_01"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "solver.py").write_text("")
        generation = {"ok": False, "error": placeholder("GENERATION_DIAGNOSTIC")}
        result = self.Result(1, "agent_failed", False, False, None, None, None, [], metrics={"generation": generation})
        save_json(directory / "agent_receipt.json", {"result": generation})
        save_json(directory / "score_receipt.json", {"result": result.to_dict(), "artifact_sha256": {"solver.py": file_sha(directory / "solver.py")}})
        task.mode = "reflection"
        return capture(lambda: task.ensure_reflection(directory, result))

    def one_backend(self, problem, backend):
        task, cases, body = self.task(problem, backend, "longest")
        captures = {}
        for attempt in range(1, 11):
            name = "shared_generation_01" if attempt == 1 else f"plain_generation_{attempt:02d}"
            captures[name] = self.generation(task, cases["plain"], attempt, "plain")
            assert captures[name]["prompt"] == captures["shared_generation_01"]["prompt"]
        for attempt in range(1, 11):
            if attempt > 1:
                captures[f"reflection_generation_{attempt:02d}"] = self.generation(task, cases["reflection"], attempt, "reflection")
            if attempt < 10:
                captures[f"reflection_debug_{attempt:02d}"] = self.reflection(task, cases["reflection"], attempt)
        for kind in ("verification", "execution", "quality_and_algorithm"):
            task, cases, _ = self.task(problem, backend, kind)
            captures[f"branch_{kind}_failure_reflection"] = self.reflection(task, cases["reflection"], 1, kind)
        task, cases, _ = self.task(problem, backend, "generation_omission")
        captures["branch_missing_solver_reflection"] = self.missing_generation(task, cases["reflection"])
        for next_attempt in (2, 10):
            task, cases, _ = self.task(problem, backend, f"reflection_omission_{next_attempt}")
            for prior in range(1, next_attempt):
                directory = cases["reflection"] / "attempts" / f"attempt_{prior:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "solver.py").write_text(placeholder("SOLVER_PY", prior) + "\n")
                (directory / "debug.md").write_text("")
                (directory / "reflection_feedback.md").write_text(self.delivery_feedback(prior) + "\n")
            captures[f"branch_missing_debug_generation_{next_attempt:02d}"] = self.generation(task, cases["reflection"], next_attempt, "reflection")
        return captures, body

    def record(self, problem, backend, name, payload):
        group = "agy" if backend == "agy-cli" else "codex_claude"
        relative = Path("raw") / problem / group / (name + ".txt")
        path = self.destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload["prompt"])
        contexts = re.findall(r"^--- BEGIN CONTEXT: (.+) ---$", payload["prompt"], flags=re.MULTILINE)
        stdin = json.dumps({"event": "user", "message": {"content": payload["prompt"]}}) + "\n" if group == "agy" else payload["prompt"]
        self.calls.append({"problem": problem, "scenario": name, "text_file": str(relative), "backends": [backend] if group == "agy" or problem == "judge" else ["codex-cli", "claude-code-cli"], "target": payload["target"], "initial_content": payload["initial_content"], "role": payload["role"], "example_model": payload["model"], "example_effort": payload["effort"], "timeout_seconds": payload["timeout"], "context_labels_in_order": contexts, "prompt_sha256": sha(payload["prompt"]), "stdin_sha256": sha(stdin), "full_prompt_with_context_placeholders": True})
        return relative

    def stage_section(self, problem, name, title, captures, *, folded=False, description=""):
        codex = captures["codex-cli"][name]
        agy = captures["agy-cli"][name]
        raw = Path("raw") / problem
        intro = f"### {title}\n\n目标文件：`{codex['target']}`；调用开始时该文件为空。{description}\n\n"
        content = intro + "GPT／Codex 与 Claude Code 的完整用户输入：\n\n```text\n" + codex["prompt"] + "\n```\n\n"
        content += f"[纯文本原件]({raw}/codex_claude/{name}.txt)\n\n"
        content += "<details>\n<summary>Gemini／AGY 的完整用户输入</summary>\n\n```text\n" + agy["prompt"] + "\n```\n\n"
        content += f"[AGY 纯文本原件]({raw}/agy/{name}.txt)\n\n</details>\n\n"
        if folded:
            return f"<details>\n<summary>{title}</summary>\n\n{content}</details>\n\n"
        return content

    def problem_document(self, problem):
        captures, body = {}, None
        for backend in ("codex-cli", "claude-code-cli", "agy-cli"):
            captures[backend], body = self.one_backend(problem, backend)
        assert set(captures["codex-cli"]) == set(captures["claude-code-cli"]) == set(captures["agy-cli"])
        for name, codex in captures["codex-cli"].items():
            assert codex["prompt"] == captures["claude-code-cli"][name]["prompt"], (problem, name)
            for backend in ("codex-cli", "agy-cli"):
                self.record(problem, backend, name, captures[backend][name])
        self.proofs.append({"problem": problem, "scenarios_per_backend": len(captures["codex-cli"]), "codex_claude_all_text_identical": True, "plain_attempts_1_to_10_text_identical": True, "template_instance": f"{problem}_medium_seed000", "instance_data_sha256": file_sha(REPO / "data" / FAMILIES[problem] / problem / f"{problem}_medium_seed000" / "data.json"), "problem_prompt_sha256": sha(body)})
        raw = self.destination / "raw" / problem / "problem.txt"
        raw.write_text(body)
        text = f"# {problem.upper()} — Agent 全过程输入\n\n代表实例：`{problem}_medium_seed000`；方法：`{FAMILIES[problem]}`。题面的具体规模和示例数字取自该实例，完整 JSON、候选代码、结果和反思内容用 `<<...>>` 占位。代码块内是当前 Docker 入口实际组装的文本；标题及中文阅读说明位于输入之外。\n\n[总览与占位约定](README.md) · [共享 judge 输入](judge.md) · [CLI 设置与传输](transport.md) · [单独题面](raw/{problem}/problem.txt)\n\n"
        text += "## 1. 共享首轮：生成 solver.py\n\nPlain 和 reflection 共用这一次生成、执行和 judge 结果。下面展示唯一的首轮生成输入；judge 的完整输入统一放在 [judge.md](judge.md)。\n\n"
        text += self.stage_section(problem, "shared_generation_01", "Attempt 1 — 生成", captures)
        text += "## 2. Plain：第二至第十轮\n\n若前一轮未通过，plain 开启新会话，使用与第 1 节逐字相同的输入及空的 solver.py。此前的 solver、日志、judge 判词和 debug.md 不进入 plain 下一轮上下文。每次生成后照常执行并调用 judge；一旦成功就结束本支路。以下文件已分别从各轮入口捕获，并校验与首轮完全相同。\n\n| 轮次 | GPT／Claude 原文 | AGY 原文 |\n| --- | --- | --- |\n"
        for attempt in range(2, 11):
            name = f"plain_generation_{attempt:02d}.txt"
            text += f"| {attempt} | [输入](raw/{problem}/codex_claude/{name}) | [输入](raw/{problem}/agy/{name}) |\n"
        text += "\n## 3. Reflection：写 debug.md，再生成下一轮 solver\n\n当前 controller 先完成 plain 支路，再处理 reflection 支路。下列最长路径以“每轮可行但目标未达标、judge PASS”为示例，展开完整十轮结构；它不是已有实验的真实输出记录。首轮使用第 1 节已共享的候选；每个未成功的第 1–9 轮之后写 debug.md，下一轮重新生成 solver.py。任意一轮成功就停止；第 10 轮之后不再调用 reflection。\n\n"
        for attempt in range(1, 11):
            if attempt > 1:
                text += self.stage_section(problem, f"reflection_generation_{attempt:02d}", f"Attempt {attempt} — 根据此前全部 reflection 历史生成 solver.py", captures, folded=attempt > 2, description="上下文按轮次累积此前的 solver.py、debug.md，以及存在时的 reflection_feedback.md。")
            if attempt < 10:
                text += self.stage_section(problem, f"reflection_debug_{attempt:02d}", f"Attempt {attempt} 失败后 — Reflection 写 debug.md", captures, folded=attempt > 1, description="此分支包含目标质量未达标的定性反馈。")
        text += "## 4. 其他失败分支的完整输入\n\n以下分支会改变反馈句或上下文文件集合。占位符替代动态内容，文件标记和固定诊断文字完整保留。\n\n"
        alternatives = [
            ("branch_verification_failure_reflection", "可行性／输出检查／algorithm judge 失败", "此例没有同时触发目标质量失败，因此没有额外的目标质量反馈句；具体问题随 exec.log 与 judge 判词给出。此例同时展示可选的 OUTPUT MESSAGE 和 VERIFICATION ISSUES 日志区段。"),
            ("branch_quality_and_algorithm_failure_reflection", "解可行，但目标质量与 algorithm judge 同时失败", "此例同时保留目标质量定性句和 algorithm_verifier_failed: fail。是否加入定性句由执行、可行性和目标质量决定，不要求 judge PASS。"),
            ("branch_execution_failure_reflection", "Solver 执行失败，未产生 output.json", "此例没有 output.json context block；已交付 solver 的情况下，judge 仍会收到代码。"),
            ("branch_missing_solver_reflection", "生成调用结束，却未交付 solver.py", "solver.py context 为空；exec.log 为框架生成阶段诊断；没有 alg_verifier.md 或 output.json block。固定 reflection 正文仍保留其通用文件描述，按当前实现原样展示。"),
            ("branch_missing_debug_generation_02", "首轮 reflection 未交付 debug.md，进入第二轮生成", "空 debug.md block 与框架的 reflection_feedback.md 均原样保留。该轮 reflection 机会已消耗，不把错路径文件的内容交给下一轮。"),
            ("branch_missing_debug_generation_10", "前九次 reflection 都未交付文件，进入第十轮生成", "此例展示最大范围的缺失反思反馈累积：每轮 solver、空 debug.md 和 framework feedback。"),
        ]
        for name, title, description in alternatives:
            text += self.stage_section(problem, name, title, captures, folded=True, description=description)
        self.documents[problem + ".md"] = text

    def judge_document(self):
        text = "# Judge — 共享的完整输入\n\n所有 problem type 共用这个入口，BD 和 CG 各有一份方法正文。实际 judge 为 GPT-6 Astra、high，经 Codex Docker CLI 调用；目标文件为 alg_verifier.md，初始为空。它获得当前 solver.py 的完整内容，问题题面、data.json、exec.log、output.json 与 gold 不在此次 context 中。\n\n生成成功交付 solver.py 后，即使 solver 执行失败，仍调用 judge；未交付 solver.py 时跳过执行与 judge。每一轮 judge 使用新会话。\n\n"
        client = self.client("codex-cli", "algorithm_judge")
        for suite in ("bd", "cg"):
            for attempt in range(1, 11):
                directory = self.work / "judge" / suite / f"attempt_{attempt:02d}"
                directory.mkdir(parents=True)
                (directory / "solver.py").write_text(placeholder("CURRENT_SOLVER_PY") + "\n")
                payload = capture(lambda: self.Verifier.run_llm_verifier(directory, client, suite))
                name = f"{suite}_judge_{attempt:02d}"
                self.record("judge", "codex-cli", name, payload)
                if attempt == 1:
                    text += f"## {suite.upper()} judge 全文\n\n```text\n{payload['prompt']}\n```\n\n[纯文本原件](raw/judge/codex_claude/{name}.txt)\n\n"
        text += "## 第二至第十轮\n\n指令逐字相同，context 标记中的 attempt_01 随当前轮次变化。所有轮次的原文如下。\n\n| 轮次 | BD | CG |\n| --- | --- | --- |\n"
        for attempt in range(1, 11):
            text += f"| {attempt} | [输入](raw/judge/codex_claude/bd_judge_{attempt:02d}.txt) | [输入](raw/judge/codex_claude/cg_judge_{attempt:02d}.txt) |\n"
        self.documents["judge.md"] = text

    def transport_document(self):
        worker_path = self.controller / "docker/agents/worker.py"
        tree = ast.parse(worker_path.read_text())
        names = {"WORK", "RECORD", "AUTH", "CODEX_POLICY", "CODEX_POLICY_ARGS", "DISABLED"}
        nodes = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id in names for target in node.targets for item in ast.walk(target)) or isinstance(node, ast.FunctionDef) and node.name == "cli_command"]
        scope = {"Path": Path, "json": json, "os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(worker_path), "exec"), scope)
        text = "# CLI 设置与传输层\n\n每个 problem 文档的代码块已经包含 DockerCliClient 的外层指令和全部 context 标记，对应传给 CLI 的用户输入正文。目标文件预先在 /work 创建且内容为空；data.json、前轮代码、日志及反思通过正文内的 context blocks 提供。模型可在本次调用中读取／编辑目标文件；每次调用启动独立会话。\n\n本目录完整展示 benchmark 提交的用户正文、上下文文件内容和 CLI 配置。CLI 内置的 system prompt、工具定义，以及模型运行中的自主工具往返属于客户端和运行时上下文，不在这次源码捕获范围内。\n\n## Backend 与模型设置\n\n| 模型 | Backend | Generation / reflection effort | 计划范围 |\n| --- | --- | --- | --- |\n"
        for item in self.plan["model_queue"] + self.plan.get("optional_queue", []):
            scope_label = "可选" if item in self.plan.get("optional_queue", []) else "正式队列"
            text += f"| {item['model']} | {item['backend']} | {item['effort']} | {scope_label} |\n"
        text += f"\n共享 judge：{self.plan['execution']['judge_model']}，effort {self.plan['execution']['judge_effort']}。Agent 调用上限 900 秒；未来 solver 的总预算 60 秒、单线程。AGY 的 effort 已编码在模型 slug 中。\n\n"
        for backend in ("codex-cli", "claude-code-cli", "agy-cli"):
            spec = self.specs[backend]
            job = {"model": spec["model"], "effort": spec["effort"], "timeout": 900}
            command = scope["cli_command"](backend, job)
            text += f"## {backend} 的实际 CLI 参数\n\n以下使用该 backend 的一个模型配置示例，其余模型替换 model／effort 字段。命令由当前 worker 的 cli_command 原函数生成。\n\n```sh\n{shlex.join(command)}\n```\n\n"
        judge_command = scope["cli_command"]("codex-cli", {"model": self.plan["execution"]["judge_model"], "effort": self.plan["execution"]["judge_effort"], "timeout": 900})
        text += f"## Judge 的 CLI 参数\n\n```sh\n{shlex.join(judge_command)}\n```\n\n"
        text += '## stdin 格式\n\nCodex 和 Claude 使用纯文本 stdin，其内容就是各 .txt 原件。AGY 把相同正文放入下面这一行 JSON 的 message.content 字段，随后加一个换行；JSON 转义不改写正文文字。\n\n```json\n{"event": "user", "message": {"content": "<<EXACT_AGY_TEXT_FROM_THE_LINKED_TXT_FILE>>"}}\n```\n\n原始 prompt 和 AGY 编码后 stdin 的 SHA-256 都记录在 manifest.json。两个 backend 的可见正文差异已在每个阶段给出完整版本：Codex／Claude 的 context 结束标记带文件标签，AGY 使用统一 END CONTEXT，并有其专用外层指令。\n'
        self.documents["transport.md"] = text

    def readme(self):
        text = "# DecompBench — Agent 全过程输入审阅包\n\n从这里按 problem type 阅读当前准备协议将交给 agent 的完整输入。每类以 medium_seed000 展示真实题面；输入 JSON、生成代码、输出、日志和反思等动态内容以 <<...>> 占位，固定指令和 context 边界保留原文。\n\n## 按 problem type 阅读\n\n| 方法 | Problem type | 全过程 |\n| --- | --- | --- |\n"
        for problem, suite in FAMILIES.items():
            text += f"| {suite.upper()} | {problem.upper()} | [{problem}.md]({problem}.md) |\n"
        text += "\n[共享 judge：BD／CG 两个实际版本](judge.md) · [模型、CLI 参数与传输层](transport.md) · [机器可读校验清单](manifest.json)\n\n## 阅读顺序\n\n1. 每类第 1 节：共享首轮生成 solver.py 的完整输入。\n2. 共享 judge：读取该轮 solver.py，写 alg_verifier.md。\n3. 每类第 2 节：plain 第二至第十轮使用与首轮逐字相同的输入，逐轮原文均有链接。\n4. 每类第 3 节：reflection 写 debug.md，再将此前所有轮次的代码／反思交给下一轮生成；展开到第十轮。\n5. 每类第 4 节：可行性／judge 失败、solver 执行失败、未交付 solver、未交付 debug 等分支的完整输入。\n\n成功后停止该支路；第十轮之后不再写 reflection。共享首轮只发生一次。最长路径使用“可行但目标未达标”的示例状态来展示所有输入结构，动态内容的占位符代表未来运行产物，不表示这些结果已经发生。\n\n## 原文与 placeholder 的边界\n\n代码块中，<<DATA_JSON>> 代表该实例完整输入；SOLVER_PY、OUTPUT_JSON、DEBUG_MD、STDOUT／STDERR、JUDGE_NOTES 等代表相应文件内容。空 debug.md／solver.py block 保留为空，以展示真实缺失文件分支。题面中的规模、参数示例、字段定义和所有规则完整保留。\n\n正文外的中文标题、流程解释与表格用于阅读，不发送给 agent。纯文本原件在 raw/，保留输入正文的精确末尾换行；Markdown 围栏的显示换行不是额外 prompt。GPT／Codex 和 Claude 的用户正文已逐条比对相同，因此共用一份原文；每个阶段另附 AGY 的完整原文。\n\n## 捕获方式\n\n本包执行当前 problem renderer、PairedRunner.run_attempt／ensure_reflection、DockerCliClient.edit_file 和 record_workspace_prompt 的实际组装路径，在 Docker runtime 接收 job payload 的边界截取文本。预检和认证由文档捕获适配器绕过，runtime 调度即停止，solver executor 被替换为禁止执行的哨兵。不会启动容器、调用模型或运行 solver。\n\nJudge 使用实际 AlgorithmVerifier 与同一 Docker client 组装；CLI 参数从 worker 的纯 cli_command 函数提取。源文件 SHA-256、每个输入正文／stdin 的 SHA-256、context 顺序及目标文件都记录在 manifest.json。该包对应当前准备的 v3 题面构建协议；旧 cohort 与所有数值校准结果保持原样。\n\n## 当前输入中值得直接检查的细节\n\n- debug.md 就是 reflection 调用的目标文件，当前流程没有独立的另一次 debugger 调用。\n- 后续 reflection-mode 生成读取此前各轮的 solver.py、debug.md，以及存在时的 reflection_feedback.md；日志、output 和 judge 判词在对应 reflection 调用中提供。\n- 目标质量失败时加入定性反馈句，不提供 gold 或数值 gap。可行性等其他失败通过日志和 judge 反馈说明。\n- 生成没有交付 solver 时，通用 reflection 正文仍保留对执行日志和 judge 文件的通用描述；同一输入中的 framework diagnostic 则明确说明没有执行或 judge。这一分支按现有代码完整展示，留待你审阅措辞。\n- 基础设施恢复会恢复已有调用或重试同一阶段，所用 prompt 保持相同；没有附加一段新的修复 prompt。完成但未交付 debug 的情况则提供 reflection_feedback.md，并消耗该次 reflection 机会。\n"
        self.documents["README.md"] = text

    def export(self):
        self.destination.mkdir(parents=True, exist_ok=True)
        for problem in FAMILIES:
            self.problem_document(problem)
        self.judge_document()
        self.transport_document()
        self.readme()
        for name, contents in self.documents.items():
            (self.destination / name).write_text(contents)
        files = [*self.controller.glob("src/*/*/b_problem.py")]
        files.extend(self.controller / name for name in ("src/agent/paired_runner.py", "src/agent/experiment_runner.py", "src/agent/prompt_builder.py", "src/agent/prompt_policy.py", "src/agent/decomposition_policy.py", "src/agent/algorithm_verifier.py", "src/agent/docker_client.py", "src/agent/files_manager.py", "src/agent/exec_logger.py", "src/prompt_snapshot.py", "docker/agents/worker.py"))
        manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "controller_source": str(self.controller), "prompt_protocol": self.prompt_protocol, "plan_sha256": file_sha(self.plan_path), "worker_image_id": self.environment["image_id"], "cli_versions": self.environment["cli_versions"], "generator_sha256": file_sha(Path(__file__)), "source_sha256": {str(path.relative_to(self.controller)): file_sha(path) for path in files}, "model_calls": 0, "solver_executions": 0, "docker_containers_started": 0, "scope": "Benchmark-owned complete user prompts and CLI configuration, with explicit dynamic-context placeholders.", "family_proofs": self.proofs, "inputs": self.calls, "document_sha256": {name: sha(contents) for name, contents in self.documents.items()}}
        save_json(self.destination / "manifest.json", manifest)
        print(json.dumps({"folder": str(self.destination), "problem_documents": len(FAMILIES), "prompt_text_files": len(self.calls), "model_calls": 0, "solver_executions": 0}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="decompbench-input-walkthrough-") as temporary:
        Walkthrough(args.controller.resolve(), args.output.resolve(), Path(temporary)).export()


if __name__ == "__main__":
    main()
