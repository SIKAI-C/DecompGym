"""Build and write the exec.log that the judge and reflection steps read."""

from __future__ import annotations

from pathlib import Path


class ExecLogger:
    """Helpers for building and writing exec logs."""
    @staticmethod
    def build_exec_lines(stdout: str, stderr: str) -> list[str]:
        """Build the base exec log sections for stdout/stderr."""
        return [
            "=== EXECUTION ===",
            f"STDOUT:\n{stdout}",
            "",
            f"STDERR:\n{stderr}",
        ]

    @staticmethod
    def append_output_message(exec_lines: list[str], output_msg: str | None) -> None:
        """Append output message section if present."""
        if not output_msg:
            return
        exec_lines.append("")
        exec_lines.append("=== OUTPUT MESSAGE ===")
        exec_lines.append(str(output_msg))

    @staticmethod
    def append_verification_issues(exec_lines: list[str], issues: list[str]) -> None:
        """Append verification issues section if any."""
        if not issues:
            return
        exec_lines.append("")
        exec_lines.append("=== VERIFICATION ISSUES ===")
        exec_lines.extend(issues)

    @staticmethod
    def append_llm_verifier(exec_lines: list[str], llm_notes: str | None) -> None:
        """Append LLM verifier section."""
        exec_lines.append("")
        exec_lines.append("=== LLM VERIFIER ===")
        exec_lines.append(llm_notes or "No verifier output.")

    @staticmethod
    def write_exec_log(attempt_dir: Path, exec_lines: list[str]) -> Path:
        """Write exec.log in the attempt directory."""
        path = attempt_dir / "exec.log"
        path.write_text("\n".join(exec_lines))
        return path
