"""PID 1 inside the XL bubblewrap sandbox, and the loopback relay for the Gurobi license tunnel.

Standalone standard-library script: the executor copies it into the sandbox's private control directory and starts it as
``python -I -B -S init.py CONFIG STATUS_FD`` under ``bwrap --as-pid-1``. It is framework code, not candidate code.

As PID 1 of the sandbox's PID namespace it
- applies the OS resource limits (address space, processes, file size, open files, CPU time, no core files), which every
  process of the sandbox inherits; the process limit counts tasks of the sandbox's user namespace only;
- raises its OOM score so that the kernel's OOM killer picks the sandbox before the executor;
- starts the license relay (TCP 127.0.0.1:PORT -> the bound Unix socket of the executor's forwarder) when configured;
- starts the payload (``python solver.py`` or one compiler command) and reports its exit status and resource usage on the
  status pipe, then exits. When PID 1 exits, the kernel kills every remaining process of the namespace (children,
  grandchildren and reparented daemons alike), so no candidate process outlives its payload.
Members of the namespace cannot signal PID 1: before anything else it resets every Python-installed handler (SIGINT's
KeyboardInterrupt) to SIG_DFL, and the kernel drops SIG_DFL signals sent to a namespace's init from inside the namespace
(SIGKILL and SIGSTOP included); cross-process prlimit64 and ptrace are denied by the executor's seccomp filter. It marks
itself non-dumpable so that sandbox processes cannot open its status pipe or /proc files. Children get SIGPIPE and SIGXFSZ
back at SIG_DFL before exec. Its status messages are diagnostics; the executor's authoritative facts are its own clock,
bwrap's exit status and the kernel's OOM counter.

Relay mode: ``python -I -B -S init.py --relay UNIX_SOCKET PORT READY_FD``.
"""
import json
import os
import resource
import select
import selectors
import signal
import socket
import sys
import time

LIMITS = {"address_space": resource.RLIMIT_AS, "processes": resource.RLIMIT_NPROC, "file_size": resource.RLIMIT_FSIZE,
          "open_files": resource.RLIMIT_NOFILE, "core": resource.RLIMIT_CORE, "cpu_seconds": resource.RLIMIT_CPU}
CHUNK = 65536


def emit(fd, event, **fields):
    if fd is None:
        return
    try:
        os.write(fd, (json.dumps({"event": event, "t": time.monotonic(), **fields}) + "\n").encode())
    except OSError:
        pass


def non_dumpable():
    """PR_SET_DUMPABLE 0, confirmed with PR_GET_DUMPABLE; any failure or exception is False (fail closed)."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE = 4
            return False
        return libc.prctl(3, 0, 0, 0, 0) == 0  # PR_GET_DUMPABLE = 3
    except Exception:  # noqa: BLE001 - every failure refuses the launch
        return False


def apply_limits(limits):
    applied = {}
    for name, value in limits.items():
        if value is None:
            continue
        kind = LIMITS[name]
        if name == "cpu_seconds":
            resource.setrlimit(kind, (int(value), int(value) + 5))
        else:
            resource.setrlimit(kind, (int(value), int(value)))
        applied[name] = list(resource.getrlimit(kind))
    return applied


def reset_python_handlers():
    """SIG_DFL for every signal with a Python-level handler (only SIGINT by default): PID 1 must not run Python handlers."""
    reset = []
    for number in signal.valid_signals():
        if number in (signal.SIGKILL, signal.SIGSTOP):
            continue
        try:
            if callable(signal.getsignal(number)):
                signal.signal(number, signal.SIG_DFL)
                reset.append(int(number))
        except (OSError, ValueError, RuntimeError):
            pass
    return reset


def spawn(argv, env, cwd, keep_fd=None, quiet=False):
    """fork + exec without any inherited descriptor except stdin (/dev/null), stdout, stderr and keep_fd."""
    pid = os.fork()
    if pid == 0:
        try:
            for number in (signal.SIGPIPE, signal.SIGXFSZ):  # CPython ignores these; SIG_IGN would survive exec
                signal.signal(number, signal.SIG_DFL)
            null = os.open("/dev/null", os.O_RDWR)
            os.dup2(null, 0)
            if quiet:
                os.dup2(null, 1)
                os.dup2(null, 2)
            if keep_fd is None:
                os.closerange(3, 65536)
            else:
                os.set_inheritable(keep_fd, True)
                os.closerange(3, keep_fd)
                os.closerange(keep_fd + 1, 65536)
            os.chdir(cwd)
            os.execve(argv[0], argv, env)
        except BaseException as error:  # noqa: BLE001 - report and die in the child
            try:
                os.write(2, f"[sandbox-init] cannot start {argv[0]}: {error}\n".encode())
            finally:
                os._exit(127)
    return pid


def start_relay(relay, status_fd):
    read_end, write_end = os.pipe()
    argv = [sys.executable, "-I", "-B", "-S", os.path.abspath(__file__), "--relay", relay["socket"], str(relay["port"]), str(write_end)]
    pid = spawn(argv, {"PATH": "/usr/bin:/bin"}, "/", keep_fd=write_end, quiet=True)
    os.close(write_end)
    readable, _, _ = select.select([read_end], [], [], float(relay.get("startup_seconds", 15)))
    ready = os.read(read_end, 16) if readable else b""
    os.close(read_end)
    if ready != b"ready":
        emit(status_fd, "error", stage="relay", message="license relay did not become ready")
        return None
    emit(status_fd, "relay_ready", pid=pid, port=relay["port"])
    return pid


def main(config_path, status_fd):
    handlers_reset = reset_python_handlers()
    status_fd = int(status_fd)
    os.set_inheritable(status_fd, False)
    with open(config_path) as stream:
        config = json.load(stream)
    try:  # before PR_SET_DUMPABLE: a non-dumpable task's /proc files belong to root, which is unmapped here
        with open("/proc/self/oom_score_adj", "w") as stream:
            stream.write(str(int(config.get("oom_score_adj", 1000))))
        oom_adj = True
    except OSError:
        oom_adj = False
    dumpable_off = non_dumpable()
    if not dumpable_off:  # without it sandbox processes could open PID 1's status pipe through /proc: never start candidate code
        emit(status_fd, "error", stage="dumpable", message="PR_SET_DUMPABLE 0 failed or could not be confirmed")
        return 73
    try:
        applied = apply_limits(config.get("limits", {}))
    except (OSError, ValueError) as error:
        emit(status_fd, "error", stage="limits", message=str(error))
        return 70
    emit(status_fd, "ready", pid=os.getpid(), uid=os.getuid(), gid=os.getgid(), affinity=sorted(os.sched_getaffinity(0)),
         limits=applied, non_dumpable=dumpable_off, oom_score_adj=oom_adj, handlers_reset=handlers_reset)
    if config.get("relay"):
        if start_relay(config["relay"], status_fd) is None:
            return 71
    argv, env, cwd = config["argv"], config["env"], config.get("cwd", "/")
    payload = spawn(argv, env, cwd)
    emit(status_fd, "payload_started", pid=payload, argv0=argv[0])
    while True:
        try:
            pid, status, usage = os.wait4(-1, 0)
        except ChildProcessError:
            emit(status_fd, "error", stage="wait", message="payload vanished")
            return 72
        if pid != payload:
            continue  # reap reparented orphans and the relay
        code = os.waitstatus_to_exitcode(status)
        emit(status_fd, "payload_exit", exit_code=code if code >= 0 else None, signal=-code if code < 0 else None,
             maxrss_kib=usage.ru_maxrss, user_seconds=usage.ru_utime, system_seconds=usage.ru_stime)
        return code if code >= 0 else 128 - code


# ----------------------------------------------------------------------------------------------- relay (inside the sandbox)

def relay_main(unix_path, port, ready_fd):
    """TCP 127.0.0.1:PORT <-> the forwarder's Unix socket, one pair per connection; an EOF on one side is passed on as
    SHUT_WR to the other (data already read is always sent first), and a pair closes once both directions have ended."""
    ready_fd = int(ready_fd)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", int(port)))
    server.listen(16)
    os.write(ready_fd, b"ready")
    os.close(ready_fd)
    selector = selectors.DefaultSelector()
    selector.register(server, selectors.EVENT_READ, None)
    peers, ended = {}, set()

    def close_pair(sock):
        other = peers.pop(sock, None)
        for item in (sock, other):
            if item is None:
                continue
            peers.pop(item, None)
            ended.discard(item)
            try:
                selector.unregister(item)
            except (KeyError, ValueError):
                pass
            item.close()

    while True:
        for key, _ in selector.select():
            if key.data is None:
                client, _ = server.accept()
                upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    upstream.connect(unix_path)
                except OSError:
                    client.close()
                    upstream.close()
                    continue
                peers[client], peers[upstream] = upstream, client
                selector.register(client, selectors.EVENT_READ, "pipe")
                selector.register(upstream, selectors.EVENT_READ, "pipe")
                continue
            sock = key.fileobj
            try:
                data = sock.recv(CHUNK)
            except OSError:
                close_pair(sock)
                continue
            if data:
                try:
                    peers[sock].sendall(data)
                except (OSError, KeyError):
                    close_pair(sock)
                continue
            ended.add(sock)  # EOF from sock: stop reading it and pass the FIN on
            selector.unregister(sock)
            try:
                peers[sock].shutdown(socket.SHUT_WR)
            except OSError:
                pass
            if peers.get(sock) in ended:
                close_pair(sock)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--relay":
        relay_main(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 3:
        sys.stdout.flush()
        os._exit(main(sys.argv[1], sys.argv[2]))
    else:
        sys.exit("usage: init.py CONFIG STATUS_FD | init.py --relay UNIX_SOCKET PORT READY_FD")
