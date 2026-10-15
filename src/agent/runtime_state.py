"""Private controller state and locks shared by all worktrees and invocations."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import tempfile


def state_root() -> Path:
    root = Path(tempfile.gettempdir()) / f"decompbench-runtime-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("The benchmark runtime state directory must be private and owned by this user")
    return root


@contextmanager
def runtime_lock(name: str, *, blocking=True):
    descriptor = os.open(state_root() / f"{name}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
