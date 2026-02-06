"""
Artifact Manager for the agent. Responsible for managing the artifacts of the agent.
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
        model_safe = model_name.replace("/", "_").replace(":", "_")
        timestamp = get_current_time()
        if note:
            note_safe = note.replace(" ", "_")
            out_dir = Path("z_artifacts") / "output" / "exp2" / suite / problem / model_safe / note_safe / timestamp
        else:
            out_dir = Path("z_artifacts") / "output" / "exp2" / suite / problem / model_safe / timestamp
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir

    @staticmethod
    def copy_instance(src: Path, dest_root: Path, max_attempts: int, suite: str) -> tuple[Path, list[Path]]:
        dest = dest_root / src.name
        dest.mkdir(parents=True, exist_ok=False)
        attempts_root = dest / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempt_dirs: list[Path] = []

        base_files = ["data.json", "prompt.txt", "gold_monolithic.json", "meta.json"]
        for fname in base_files:
            src_path = src / fname
            dst_path = dest / fname
            if src_path.exists():
                shutil.copy2(src_path, dst_path)

        for idx in range(1, max_attempts + 1):
            a_dir = attempts_root / f"attempt_{idx:02d}"
            a_dir.mkdir(parents=True, exist_ok=True)
            attempt_dirs.append(a_dir)
        return dest, attempt_dirs