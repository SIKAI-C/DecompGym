"""
Execution log helper for the agent. Responsible for building and writing exec logs.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.files_manager import ArtifactManager


class ExecLogger:
    @staticmethod
    def build_exec_lines(stdout: str, stderr: str) -> list[str]:
        return [
            "=== EXECUTION ===",
            f"STDOUT:\n{stdout}",
            "",
            f"STDERR:\n{stderr}",
        ]

    @staticmethod
    def append_output_message(exec_lines: list[str], output_msg: str | None) -> None:
        if not output_msg:
            return
        exec_lines.append("")
        exec_lines.append("=== OUTPUT MESSAGE ===")
        exec_lines.append(str(output_msg))

    @staticmethod
    def append_verification_issues(exec_lines: list[str], issues: list[str]) -> None:
        if not issues:
            return
        exec_lines.append("")
        exec_lines.append("=== VERIFICATION ISSUES ===")
        exec_lines.extend(issues)

    @staticmethod
    def append_llm_verifier(exec_lines: list[str], llm_notes: str | None) -> None:
        exec_lines.append("")
        exec_lines.append("=== LLM VERIFIER ===")
        exec_lines.append(llm_notes or "No verifier output.")

    @staticmethod
    def write_exec_log(attempt_dir: Path, exec_lines: list[str]) -> Path:
        return ArtifactManager.write_text(attempt_dir, "exec.log", "\n".join(exec_lines))
