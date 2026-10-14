"""Artifact collection rejects links and nonfiles without waiting for their writers."""
import os

import pytest

from src.agent.safe_artifacts import UnsafeArtifact, read_regular, read_regular_text


def test_regular_utf8_and_missing_file_handling(tmp_path):
    (tmp_path / "output.json").write_text('{"hello":"世界"}')
    assert read_regular_text(tmp_path, "output.json") == '{"hello":"世界"}'
    assert read_regular(tmp_path, "missing", missing_ok=True) is None
    with pytest.raises(UnsafeArtifact, match="missing"):
        read_regular(tmp_path, "missing")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory", "oversize", "invalid_utf8", "parent_symlink"])
def test_untrusted_file_shapes_are_rejected(tmp_path, kind):
    root = tmp_path / "work"
    root.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    original = private / "output.json"
    original.write_text("PRIVATE_REFERENCE")
    target = root / "output.json"
    relative = "output.json"
    if kind == "symlink":
        target.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, target)
    elif kind == "fifo":
        os.mkfifo(target)
    elif kind == "directory":
        target.mkdir()
    elif kind == "oversize":
        target.write_bytes(b"x" * 33)
    elif kind == "invalid_utf8":
        target.write_bytes(b"\xff")
    else:
        (root / "nested").symlink_to(private, target_is_directory=True)
        relative = "nested/output.json"
    with pytest.raises(UnsafeArtifact):
        read_regular_text(root, relative, maximum=32)
    assert original.read_text() == "PRIVATE_REFERENCE"


@pytest.mark.parametrize("relative", ["../private", "/etc/passwd", ""])
def test_artifacts_cannot_escape_the_trusted_root(tmp_path, relative):
    with pytest.raises(UnsafeArtifact):
        read_regular(tmp_path, relative)
