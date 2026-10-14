"""Trusted container worker; credentials never enter the prompt or host run records."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time

sys.path.insert(0, "/opt/decompbench")
from cli_events import parse_events
from safe_artifacts import UnsafeArtifact, read_regular_text


WORK, RECORD, AUTH = Path("/work"), Path("/record"), Path("/auth")
RECORD_FILES = {"events.jsonl", "stderr.log", "response.md", "worker.json", "delivery.json"}
CODEX_POLICY = 'permissions.benchmark.filesystem={":root"="deny",":minimal"="read","/work"="write","/tmp"="write","/usr/local/lib/node_modules"="read","/home/node/.codex/tmp"="read","/home/node/.codex/auth.json"="deny","/auth"="deny","/record"="deny"}'
CODEX_POLICY_ARGS = ["-c", CODEX_POLICY, "-c", "permissions.benchmark.network.enabled=false"]
DISABLED = ("plugins", "remote_plugin", "apps", "hooks", "memories", "multi_agent", "skill_search", "shell_snapshot")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def secrets_in(value):
    if isinstance(value, dict):
        return set().union(*(secrets_in(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(secrets_in(item) for item in value))
    return {value} if isinstance(value, str) and len(value) >= 20 else set()


def redact(text, secrets):
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED_CREDENTIAL]")
    return text


def token_secrets(contents):
    try:
        return secrets_in(json.loads(contents))
    except (ValueError, UnicodeError):
        value = contents.decode("utf-8", errors="ignore").strip()
        return {value} if value else set()


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(name)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream)
    temporary.replace(path)


def checkpoint_json(path, initial, updated):
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_regular_text(path.parent, path.name, missing_ok=True)
        current = json.loads(existing) if existing else None
        if current is not None and current != initial and updated == initial:
            return  # Another worker has already refreshed the shared credential.
        if current and current != initial and current.get("last_refresh", "") > updated.get("last_refresh", ""):
            return
        private_json(path, updated)


def credentials(backend, job):
    environment = os.environ.copy()
    secrets = set()
    checkpoint = lambda: None
    if backend == "codex-cli":
        saved = read_regular_text(AUTH, "codex.json", missing_ok=True)
        auth = json.loads(saved) if saved else job.get("credential")
        if not isinstance(auth, dict) or auth.get("auth_mode") != "chatgpt" or auth.get("OPENAI_API_KEY") or not auth.get("tokens", {}).get("access_token"):
            raise RuntimeError("Codex subscription authentication is required")
        destination = Path.home() / ".codex/auth.json"
        private_json(destination, auth)
        secrets.update(secrets_in(auth))
        def checkpoint():
            updated = json.loads(read_regular_text(destination.parent, destination.name))
            secrets.update(secrets_in(updated))
            checkpoint_json(AUTH / "codex.json", auth, updated)
    elif backend == "claude-code-cli":
        auth = job.get("credential") or {}
        token = auth.get("oauth_token")
        if not token and not auth.get("credentials"):
            saved_token = read_regular_text(AUTH, "claude_token.json", missing_ok=True)
            token = json.loads(saved_token)["oauth_token"] if saved_token else None
        if token:
            environment["CLAUDE_CODE_OAUTH_TOKEN"] = token
            secrets.add(token)
            private_json(AUTH / "claude_token.json", {"oauth_token": token})
        else:
            saved = read_regular_text(AUTH, "claude.json", missing_ok=True)
            value = json.loads(saved) if saved else auth.get("credentials")
            if not isinstance(value, dict) or not value.get("claudeAiOauth"):
                raise RuntimeError("Claude subscription authentication is required")
            destination = Path.home() / ".claude/.credentials.json"
            private_json(destination, value)
            secrets.update(secrets_in(value))
            def checkpoint():
                updated = json.loads(read_regular_text(destination.parent, destination.name))
                secrets.update(secrets_in(updated))
                checkpoint_json(AUTH / "claude.json", value, updated)
        environment.update(DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    elif backend == "agy-cli":
        import agy_credentials as vault
        password = (job.get("credential") or {}).get("keyring_passphrase")
        if not password:
            raise RuntimeError("The dedicated AGY keyring must be unlocked by its owner")
        secrets.add(password)
        bus = subprocess.run(["dbus-daemon", "--session", "--fork", "--print-address=1"], capture_output=True, text=True, check=True, timeout=10)
        environment["DBUS_SESSION_BUS_ADDRESS"] = bus.stdout.strip()
        os.environ["DBUS_SESSION_BUS_ADDRESS"] = environment["DBUS_SESSION_BUS_ADDRESS"]
        keyring = Path.home() / ".local/share/keyrings"
        keyring.parent.mkdir(parents=True, exist_ok=True)
        keyring.symlink_to(AUTH / "keyrings", target_is_directory=True)
        vault.unlock_keyring(password)
        if not vault.restore_native_token():
            raise RuntimeError("The AGY keyring has no saved native login")
        token = vault.read_native_token()
        secrets.update(token_secrets(token))
        settings = Path.home() / ".gemini/antigravity-cli/settings.json"
        write_json(settings, {"enableTelemetry": False, "useG1Credits": False, "showTips": False, "showFeedbackSurvey": False, "altScreenMode": "never"})
        environment.update(SSH_CONNECTION="127.0.0.1 1 127.0.0.1 22", AGY_CLI_DISABLE_AUTO_UPDATE="true")
        def checkpoint():
            updated = vault.read_native_token()
            if updated:
                secrets.update(token_secrets(updated))
                vault.seal_native_token()
    else:
        raise ValueError("Unsupported backend")
    return environment, secrets, checkpoint


def cli_command(backend, job):
    if backend == "codex-cli":
        command = ["codex", "--no-daemon", "--strict-config", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--json", "--color", "never", "--cd", str(WORK), "--model", job["model"], *CODEX_POLICY_ARGS,
                   "-c", 'default_permissions="benchmark"', "-c", "model_reasoning_effort=" + json.dumps(job["effort"]), "-c", 'model_provider="openai"', "-c", 'cli_auth_credentials_store="file"', "-c", 'approval_policy="never"', "-c", "project_doc_max_bytes=0",
                   "-c", 'developer_instructions=""', "-c", 'web_search="disabled"', "-c", "allow_login_shell=false", "-c", 'shell_environment_policy.inherit="none"', "-c", 'shell_environment_policy.set={PATH="/usr/local/bin:/usr/bin:/bin"}', "--enable", "skip_host_skill_discovery", "--output-last-message", str(RECORD / "response.md")]
        for feature in DISABLED:
            command.extend(["--disable", feature])
        return command + ["-"]
    if backend == "claude-code-cli":
        denied = [f"{tool}(//{path}/**)" for tool in ("Read", "Edit", "Write") for path in ("auth", "record", "proc", "home/node/.claude")]
        return ["claude", "--print", "--model", job["model"], "--effort", job["effort"], "--input-format", "text", "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--restricted", "--safe-mode", "--no-chrome",
                "--tools", "Read,Edit,Write", "--allowedTools", "Read,Edit,Write", "--disallowedTools", ",".join(denied), "--permission-mode", "acceptEdits", "--permission-prompts", "none", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands"]
    return ["agy", "--input-format", "stream-json", "--output-format", "stream-json", "--model", job["model"], "--new-project", "--mode", "accept-edits", "--sandbox", "--disable-slash-commands", "--print-timeout", str(job["timeout"]) + "s", "--log-file", os.devnull]


def limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def monitor_owner(last_seen, stopped):
    for line in sys.stdin:
        if line.strip() == "heartbeat":
            last_seen[0] = time.monotonic()
        if stopped.is_set():
            break
    last_seen[0] = 0


def run_agent(backend, job):
    target = job["target"]
    if target not in {"solver.py", "debug.md", "alg_verifier.md"}:
        raise ValueError("Unsupported editing target")
    limits()
    last_seen, stopped = [time.monotonic()], threading.Event()
    threading.Thread(target=monitor_owner, args=(last_seen, stopped), daemon=True).start()
    environment, secrets, checkpoint = credentials(backend, job)
    if time.monotonic() - last_seen[0] > 5:
        raise RuntimeError("Controller disconnected before inference")
    (WORK / target).write_text(job.get("initial_content", ""))
    command = cli_command(backend, job)
    prompt = json.dumps({"event": "user", "message": {"content": job["prompt"]}}) + "\n" if backend == "agy-cli" else job["prompt"]
    started = time.monotonic()
    interruption = None
    with (RECORD / "events.jsonl").open("w") as output, (RECORD / "stderr.log").open("w") as error:
        process = subprocess.Popen(command, cwd=WORK, env=environment, stdin=subprocess.PIPE, stdout=output, stderr=error, text=True, start_new_session=True)
        def send_prompt():
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threading.Thread(target=send_prompt, daemon=True).start()
        while process.poll() is None:
            if time.monotonic() - last_seen[0] > 5:
                interruption = "controller_connection_lost"
            elif time.monotonic() - started > job["timeout"]:
                interruption = "agent_timeout"
            if interruption:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                break
            stopped.wait(0.1)
    stopped.set()
    checkpoint_error = None
    try:
        checkpoint()
    except Exception as error:
        checkpoint_error = type(error).__name__
    for name in ("events.jsonl", "stderr.log", "response.md"):
        contents = read_regular_text(RECORD, name, missing_ok=True)
        if contents is not None:
            (RECORD / name).write_text(redact(contents, secrets))
    parsed = parse_events(backend, (RECORD / "events.jsonl").read_text(), returncode=process.returncode, requested_model=job["model"])
    if parsed["response"]:
        (RECORD / "response.md").write_text(parsed.pop("response"))
    rejected = None
    try:
        candidate = read_regular_text(WORK, target, missing_ok=True)
        if candidate is not None and any(secret in candidate for secret in secrets):
            candidate = None
            rejected = "credential_in_target"
            (WORK / target).unlink()
    except UnsafeArtifact:
        candidate, rejected = None, "unsafe_target"
    misplaced = {}
    misplaced_bytes = 0
    if not candidate or not candidate.strip():
        for directory, directories, files in os.walk(WORK, followlinks=False):
            directories[:] = [name for name in directories if not (Path(directory) / name).is_symlink()]
            if target in files:
                path = Path(directory) / target
                if path == WORK / target:
                    continue
                relative = str(path.relative_to(WORK))
                try:
                    value = read_regular_text(WORK, relative)
                except UnsafeArtifact:
                    continue
                if value.strip():
                    sanitized = redact(value, secrets)
                    size = len(json.dumps(sanitized).encode())
                    if misplaced_bytes + size > 2 * 1024 * 1024:
                        break
                    misplaced[relative] = sanitized
                    misplaced_bytes += size
                if len(misplaced) >= 32:
                    break
    write_json(RECORD / "delivery.json", {"rejected_target": rejected, "misplaced": misplaced})
    version = subprocess.run([command[0], "--version"], capture_output=True, text=True, timeout=10)
    metadata = parsed | {"backend": backend, "requested_model": job["model"], "requested_effort": job.get("effort"), "role": job["role"], "returncode": process.returncode,
                         "agent_seconds": time.monotonic() - started, "cli_version": version.stdout.strip(), "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(), "target": target,
                         "interruption": interruption, "credential_checkpoint_error": checkpoint_error, "candidate_present": bool(candidate and candidate.strip()), "command": command}
    if interruption:
        metadata.update(terminal_state="infrastructure_failed", error=interruption)
    write_json(RECORD / "worker.json", metadata)
    print(json.dumps({"worker_complete": True}))
    return process.returncode


def collect(names):
    artifacts, rejected = {}, []
    for name in names:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Invalid artifact name")
        try:
            value = read_regular_text(RECORD if name in RECORD_FILES else WORK, name, missing_ok=True)
        except UnsafeArtifact:
            if name in RECORD_FILES:
                raise
            rejected.append(name)
            continue
        if value is not None:
            artifacts[name] = value
    print(json.dumps({"ok": True, "artifacts": artifacts, "rejected": rejected}))
    return 0


def preflight():
    # This checks the actual Codex tool sandbox before any credential is introduced.
    local = Path.home() / ".codex/auth.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("SYNTHETIC_CREDENTIAL_CANARY")
    (WORK / "credential-alias").symlink_to(local)
    (RECORD / "private-canary").write_text("PRIVATE_RECORD_CANARY")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        code = """import json,socket
from pathlib import Path
checks={}
Path('/work/output.txt').write_text('allowed')
checks['work_read_write']=Path('/work/output.txt').read_text()=='allowed'
for name,path in [('auth','/home/node/.codex/auth.json'),('alias','/work/credential-alias'),('record','/record/private-canary'),('proc','/proc/1/root/home/node/.codex/auth.json')]:
 try:Path(path).read_text();checks[name]=False
 except OSError:checks[name]=True
try:socket.create_connection(('127.0.0.1',PORT),timeout=1);checks['network']=False
except OSError:checks['network']=True
print(json.dumps(checks))
""".replace("PORT", str(port))
        run = subprocess.run(["codex", "--no-daemon", "sandbox", "-C", str(WORK), "-P", "benchmark", *CODEX_POLICY_ARGS, "--", "python3", "-c", code], capture_output=True, text=True, timeout=30)
    checks = json.loads(run.stdout) if run.returncode == 0 else {}
    versions = {name: subprocess.check_output([name, "--version"], text=True).strip() for name in ("codex", "claude", "agy")}
    files = {name: hashlib.sha256((Path("/opt/decompbench") / name).read_bytes()).hexdigest() for name in ("worker.py", "agy_credentials.py", "safe_artifacts.py", "cli_events.py")}
    result = {"ok": run.returncode == 0 and bool(checks) and all(checks.values()), "checks": checks, "versions": versions, "worker_files_sha256": files, "credential_free": True, "diagnostic": run.stderr[-2000:]}
    write_json(RECORD / "worker.json", result)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


def main():
    mode = sys.argv[1]
    if mode == "collect":
        return collect(sys.argv[2:])
    if mode == "preflight":
        return preflight()
    job = json.loads(sys.stdin.readline())
    if mode == "auth-check":
        backend = sys.argv[2]
        _, _, checkpoint = credentials(backend, job)
        checkpoint()
        print(json.dumps({"credential_loaded": True, "backend": backend, "inference_submitted": False}))
        return 0
    if mode == "agent":
        return run_agent(sys.argv[2], job)
    raise ValueError("Unsupported worker mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Credential-plane exception values are intentionally never exported.
        print(json.dumps({"worker_error": type(error).__name__}))
        raise SystemExit(1)
