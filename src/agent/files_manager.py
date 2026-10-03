"""
Files Manager for the agent. Responsible for managing the files (artifacts, instances, etc.) of the agent.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from src.utils.time_utils import get_current_time


class ArtifactManager:
    """Utility helpers for creating artifact directories and files."""

    @staticmethod
    def ensure_dir(path: Path, *, exist_ok: bool = False) -> Path:
        """Create a directory path (and parents) and return it."""
        path.mkdir(parents=True, exist_ok=exist_ok)
        return path

    @staticmethod
    def write_text(
        parent_dir: Path,
        filename: str,
        content: str,
    ) -> Path:
        """
        Create a Path under parent_dir, ensure parent dir exists, then write text.
        """
        path = parent_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    @staticmethod
    def ensure_output_root(suite: str, problem: str, model_name: str, note: str | None = None) -> Path:
        """Create and return a unique output root directory."""
        model_safe = model_name.replace("/", "_").replace(":", "_")
        timestamp = get_current_time()
        if note:
            note_safe = note.replace(" ", "_")
            out_dir = Path("outputs") / "experiments" / "single" / suite / problem / model_safe / note_safe / timestamp
        else:
            out_dir = Path("outputs") / "experiments" / "single" / suite / problem / model_safe / timestamp
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir

    @staticmethod
    def copy_instance(src: Path, dest_root: Path, max_attempts: int, suite: str) -> tuple[Path, Path]:
        """Copy instance files into a new output directory and create attempts root."""
        dest = dest_root / src.name
        dest.mkdir(parents=True, exist_ok=False)
        attempts_root = dest / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)

        base_files = ["data.json", "prompt.txt"]
        for fname in base_files:
            src_path = src / fname
            dst_path = dest / fname
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
        return dest, attempts_root

    @staticmethod
    def collect_previous_debug(attempts_root: Path, current_idx: int) -> str:
        """Aggregate prior debug.md contents up to current_idx."""
        pieces: list[str] = []
        for idx in range(1, current_idx):
            dbg = attempts_root / f"attempt_{idx:02d}" / "debug.md"
            if dbg.exists():
                pieces.append(dbg.read_text())
        return "\n\n".join(pieces)

    @staticmethod
    def collect_previous_attempt_files(attempts_root: Path, current_idx: int) -> list[str]:
        """Collect read-only prior attempt files for context."""
        ro: list[str] = []
        for idx in range(1, current_idx):
            a_dir = attempts_root / f"attempt_{idx:02d}"
            for name in ("solver.py", "debug.md"):
                p = a_dir / name
                if p.exists():
                    ro.append(str(p.resolve()))
        return ro

class InstanceManager:
    """Utility helpers for managing instances."""

    @staticmethod
    def find_instances(root: Path) -> list[Path]:
        """Find instance directories that contain required input files."""
        if not root.exists():
            return []
        out: list[Path] = []
        for p in sorted(root.iterdir()):
            if p.is_dir() and InstanceManager.ensure_files_exist(p, ["data.json", "prompt.txt"]):
                out.append(p)
        return out

    @staticmethod
    def ensure_files_exist(parent_dir: Path, filenames: list[str]) -> bool:
        """Return True if all filenames exist under parent_dir."""
        for name in filenames:
            if not (parent_dir / name).exists():
                return False
        return True

    @staticmethod
    def infer_problem_from_path(path: Path, all_problems: set[str]) -> str:
        """Infer problem name by walking parents and matching known problems."""
        for parent in path.resolve().parents:
            if parent.name in all_problems:
                return parent.name
        raise ValueError(f"Could not infer problem type from path {path}")

    @staticmethod
    def infer_suite_from_path(path: Path, suite_problems: dict[str, list[str]]) -> str:
        """Infer suite name by walking parents and matching known suites."""
        for parent in path.resolve().parents:
            if parent.name in suite_problems:
                return parent.name
        raise ValueError(f"Could not infer suite from path {path}")
