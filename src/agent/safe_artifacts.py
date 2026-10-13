"""Read untrusted files without following links or opening special devices."""
from __future__ import annotations

import os
from pathlib import Path
import stat


MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


class UnsafeArtifact(ValueError):
    """A delivered file violates the regular-file output contract."""


def read_regular(root: Path, relative: str, *, maximum: int = MAX_ARTIFACT_BYTES, missing_ok: bool = False) -> bytes | None:
    """Anchor every untrusted component to an opened, trusted root directory."""
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise UnsafeArtifact("Artifact must be a relative descendant")
    if maximum < 0:
        raise ValueError("maximum must be nonnegative")
    descriptors = []
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(descriptor)
        for part in path.parts[:-1]:
            descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            descriptors.append(descriptor)
        descriptor = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise UnsafeArtifact("Artifact must be a single-link regular file within the size limit")
        contents = bytearray()
        while len(contents) <= maximum:
            block = os.read(descriptor, min(65536, maximum + 1 - len(contents)))
            if not block:
                break
            contents.extend(block)
        after = os.fstat(descriptor)
        if len(contents) > maximum:
            raise UnsafeArtifact("Artifact exceeds the size limit")
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_nlink) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_nlink):
            raise UnsafeArtifact("Artifact changed while being collected")
        return bytes(contents)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise UnsafeArtifact("Required artifact is missing") from None
    except OSError as error:
        raise UnsafeArtifact(f"Artifact cannot be opened safely (errno {error.errno})") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_regular_text(root: Path, relative: str, **kwargs) -> str | None:
    contents = read_regular(root, relative, **kwargs)
    if contents is None:
        return None
    try:
        return contents.decode("utf-8")
    except UnicodeError as error:
        raise UnsafeArtifact("Artifact must contain UTF-8 text") from error
