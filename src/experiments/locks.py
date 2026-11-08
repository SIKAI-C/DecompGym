"""Private controller state and locks shared by all worktrees and invocations."""

from contextlib import contextmanager
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import stat
import sys
import tempfile


@lru_cache(maxsize=None)
def user_temporary_directory() -> Path:
    """The user's own temporary directory, independent of the launch environment's TMPDIR.

    On macOS this is confstr(_CS_DARWIN_USER_TEMP_DIR), the /var/folders/.../T directory that a login shell receives
    as TMPDIR, so the state directory is the one every earlier invocation from an ordinary shell used, while a launch
    with another or no TMPDIR (a supervisor, a test runner, `env -u TMPDIR`) still shares the same locks. Elsewhere the
    standard temporary directory."""
    if sys.platform != "darwin":
        return Path(tempfile.gettempdir())
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    name = 65537  # _CS_DARWIN_USER_TEMP_DIR (<unistd.h>)
    size = libc.confstr(name, None, 0)
    buffer = ctypes.create_string_buffer(max(size, 1))
    if size <= 1 or libc.confstr(name, buffer, size) != size or not buffer.value.startswith(b"/"):
        raise RuntimeError("The macOS per-user temporary directory is unavailable")
    return Path(os.fsdecode(buffer.value))


def state_root() -> Path:
    root = user_temporary_directory() / f"decompbench-runtime-{os.getuid()}"
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
