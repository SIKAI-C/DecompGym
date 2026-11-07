"""Login, store and no-inference checks for one auth volume, inside a one-off container of the agent image.

Started only by `python src/02_experiment.py docker auth ...` (src/experiments/agent_auth.py), never directly. That
command bind-mounts this file read-only (the agent image and its worker hashes stay unchanged), mounts exactly one auth
volume, and holds that volume's host lock until this container has exited. The helper never submits a model prompt
and never prints a credential value; a value that must enter it (a Claude token, a Codex import, the AGY passphrase)
arrives as one JSON line on standard input. Every non-interactive mode prints one JSON report line.

Modes:
- codex-login --slot N --volume NAME --policy P [--replace]: interactive `codex login --device-auth` (ChatGPT
  device-code sign-in) into the container's tmpfs; only a completed sign-in is staged and atomically promoted to
  /auth/codex.json. A saved credential is replaced only with --replace, and the replaced file is kept as
  codex.json.replaced-<UTC> (never used automatically). meta.json records the origin, time and CLI version.
- codex-import: {"auth": ..., "replace": bool, "origin": "host-copy", ...}: the same promotion for an explicit import.
- codex-check: {"models": [...]}: initialize the pinned app-server (stdio JSON-RPC), then account/read
  (refreshToken false), account/rateLimits/read and model/list. These read the account and its usage windows from the
  backend with the saved token, without a turn. A token the app-server refreshed is written back to the volume.
- claude-store: {"token": ..., "replace": bool, "origin": ...}: store a setup token as /auth/claude_token.json (a legacy
  refreshing claude.json is moved aside, so the worker can never fall back to it).

Commit protocol of a stored credential: the new meta.json is written first as meta.json.pending, the credential is then
replaced atomically, and the pending meta is renamed to meta.json. A pending file left behind marks an interrupted
commit; every check reports it as commit_incomplete until a new --login --replace completes.
- claude-check: local only (presence, setup-token format, storage age); Claude Code has no inference-free online check.
- agy-check: {"passphrase": ..., "models": [...]}: unlock the keyring, restore the login, run `agy models`, seal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

sys.path.insert(0, "/opt/decompbench")
sys.path.insert(0, "/opt/login")
from safe_artifacts import UnsafeArtifact, read_regular, read_regular_text  # noqa: E402

AUTH = Path("/auth")
META, PENDING = "meta.json", "meta.json.pending"
CODEX_FILE, CLAUDE_FILE = "codex.json", "claude_token.json"
SETUP_TOKEN_PREFIX = "sk-ant-oat01-"
CODEX_CONFIG = ["-c", 'cli_auth_credentials_store="file"']
CREDENTIAL = re.compile(r"\b401\b|unauthorized|could not be refreshed|sign in again|log in again|refresh_token|invalidated|token[_ ]expired|token is expired|"
                        r"not logged in|authentication required|login required|no credentials|not signed in|please sign in|unauthenticated", re.I)
RATE = re.compile(r"\b429\b|too many requests|rate[_ ]limit|usage[_ ]limit|quota|resource_exhausted", re.I)
NETWORK = re.compile(r"error sending request|connection refused|connection reset|timed out|timeout|dns|failed to lookup|lookup address|network is unreachable|"
                     r"tcp connect|could not resolve|unreachable|tls|certificate|no route to host|temporary failure in name resolution", re.I)


class Refused(RuntimeError):
    """A saved credential exists and --replace was not given."""


def now():
    return datetime.now(timezone.utc)


def stamp():
    return now().strftime("%Y%m%dT%H%M%SZ")


def report(value):
    print(json.dumps(value, sort_keys=True), flush=True)


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


def regular(path: Path) -> bool:
    """True for a regular file; a link or another special entry at a credential path is refused."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{path.name} in the auth volume is not a regular file")
    return True


def write_private(path: Path, value) -> None:
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".staging-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_meta():
    try:
        text = read_regular_text(AUTH, META, missing_ok=True)
        value = json.loads(text) if text else {}
        return value if isinstance(value, dict) else {}
    except (ValueError, UnsafeArtifact):
        return {"unreadable": True}


def keep_replaced(name: str, label: str) -> str:
    """Copy the current file to <name>.replaced-<UTC> (a new single-link file; the worker reads only the exact name)."""
    descriptor = os.open(AUTH / name, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        contents = stream.read()
    for index in range(100):
        candidate = f"{name}.replaced-{label}" + (f"-{index}" if index else "")
        try:
            descriptor = os.open(AUTH / candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        return candidate
    raise RuntimeError("Could not name the replaced credential file")


def promote(name: str, value, *, replace: bool, meta: dict) -> str | None:
    """Stage `value`, then commit it as AUTH/name with its meta (see the commit protocol above). The replaced credential
    and meta are kept as .replaced-<UTC> copies and never used automatically."""
    existing = regular(AUTH / name)
    if existing and not replace:
        raise Refused(name)
    label = stamp()
    descriptor, staged = tempfile.mkstemp(prefix="." + name + ".staging-", dir=AUTH)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replaced = keep_replaced(name, label) if existing else None
        if replaced and regular(AUTH / META):
            keep_replaced(META, label)
        write_private(AUTH / PENDING, meta | {"replaced_file": replaced})  # 1. the intent
        sync_directory(AUTH)
        os.replace(staged, AUTH / name)  # 2. the credential
        os.replace(AUTH / PENDING, AUTH / META)  # 3. the commit
    finally:
        Path(staged).unlink(missing_ok=True)
    sync_directory(AUTH)
    return replaced


def commit_problem():
    """The check status of an interrupted commit, or None."""
    return "commit_incomplete" if regular(AUTH / PENDING) else None


def cli_version(name):
    try:
        return subprocess.run([name, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def token_secrets(contents):
    """Every secret-like string of a credential file's bytes (JSON values, or the whole text), for redaction."""
    if not contents:
        return set()
    try:
        return secrets_in(json.loads(contents))
    except (ValueError, UnicodeError):
        value = contents.decode("utf-8", errors="ignore").strip() if isinstance(contents, bytes) else str(contents).strip()
        return {value} if value else set()


def credential_meta(backend, origin, **fields):
    return {"schema_version": 1, "backend": backend, "origin": origin, "created_at": now().isoformat(), "generation": uuid.uuid4().hex,
            "cli_version": cli_version("codex" if backend == "codex-cli" else "claude"), **fields}


# ----------------------------------------------------------------------------------------------------------------------
# Codex


def codex_shape_problem(auth):
    if not isinstance(auth, dict):
        return "not a JSON object"
    if auth.get("auth_mode") != "chatgpt" or auth.get("OPENAI_API_KEY"):
        return "not a ChatGPT subscription login (API keys are not used)"
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict) or not tokens.get("access_token") or not tokens.get("refresh_token"):
        return "no access and refresh token"
    return None


def codex_login(arguments):
    parser = argparse.ArgumentParser(prog="codex-login")
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(arguments)
    if regular(AUTH / CODEX_FILE) and not args.replace:
        print(f"Slot {args.slot} ({args.volume}) already holds a saved Codex credential; rerun with --replace to replace it.", file=sys.stderr)
        return 3
    home = Path.home() / ".codex"
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(f"Codex device sign-in for call-gate slot {args.slot} (auth volume {args.volume}).\n"
          "Open the URL below in your browser, sign in to ChatGPT and enter the one-time code. If ChatGPT reports that device-code sign-in is disabled,\n"
          "enable it in ChatGPT Settings > Security and rerun this command. No benchmark prompt is sent.", flush=True)
    returncode = subprocess.call(["codex", "login", *CODEX_CONFIG, "--device-auth"], env=os.environ | {"CODEX_HOME": str(home)})
    if returncode:
        print("The sign-in did not complete; the saved credential (if any) is unchanged.", file=sys.stderr)
        return 1
    try:
        auth = json.loads(read_regular_text(home, "auth.json"))
    except (ValueError, UnsafeArtifact, FileNotFoundError):
        print("Codex reported success but wrote no readable file credential; the saved credential is unchanged.", file=sys.stderr)
        return 1
    problem = codex_shape_problem(auth)
    if problem:
        print(f"The new Codex login is unusable ({problem}); the saved credential is unchanged.", file=sys.stderr)
        return 1
    meta = credential_meta("codex-cli", "container-device-auth", slot=args.slot, volume=args.volume, auth_policy=args.policy, credential_file=CODEX_FILE)
    try:
        replaced = promote(CODEX_FILE, auth, replace=args.replace, meta=meta)
    except Refused:
        print("A credential appeared in this slot during the sign-in; rerun with --replace.", file=sys.stderr)
        return 3
    report({"stored": True, "backend": "codex-cli", "slot": args.slot, "volume": args.volume, "origin": meta["origin"], "created_at": meta["created_at"], "replaced_file": replaced})
    return 0


def codex_import(request):
    auth = request.get("auth")
    problem = codex_shape_problem(auth)
    if problem:
        report({"stored": False, "status": "credential_invalid", "detail": problem})
        return 1
    meta = credential_meta("codex-cli", request.get("origin", "host-copy"), slot=request.get("slot"), volume=request.get("volume"), auth_policy=request.get("policy"),
                           credential_file=CODEX_FILE, warning="shares the host's OAuth refresh lineage; a host-side refresh can invalidate it")
    try:
        replaced = promote(CODEX_FILE, auth, replace=bool(request.get("replace")), meta=meta)
    except Refused:
        report({"stored": False, "status": "refused", "detail": "a saved credential exists; rerun with --replace"})
        return 3
    report({"stored": True, "backend": "codex-cli", "origin": meta["origin"], "created_at": meta["created_at"], "replaced_file": replaced})
    return 0


def classify_text(text):
    """credential_invalid, rate_limited, network or unknown_error for one error text (credential markers win)."""
    if CREDENTIAL.search(text or ""):
        return "credential_invalid"
    if RATE.search(text or ""):
        return "rate_limited"
    if NETWORK.search(text or ""):
        return "network"
    return "unknown_error"


class AppServer:
    """A minimal stdio JSON-RPC client of `codex app-server` (requests are matched by id; notifications are ignored)."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.process = subprocess.Popen(["codex", *CODEX_CONFIG, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        env=os.environ | {"CODEX_HOME": str(Path.home() / ".codex")})
        self.messages, self.stderr = queue.Queue(), []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                self.messages.put(value)
        self.messages.put(None)

    def _drain(self):
        for line in self.process.stderr:
            if sum(len(item) for item in self.stderr) < 65536:
                self.stderr.append(line)

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def request(self, identifier, method, params):
        """("result", value), ("error", {"code", "message"}), ("timeout", None) or ("closed", None)."""
        try:
            self.send({"id": identifier, "method": method, "params": params})
        except (BrokenPipeError, OSError):
            return "closed", None
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout", None
            try:
                message = self.messages.get(timeout=remaining)
            except queue.Empty:
                return "timeout", None
            if message is None:
                return "closed", None
            if message.get("id") == identifier and "method" not in message:
                if "error" in message:
                    error = message["error"] if isinstance(message["error"], dict) else {"message": str(message["error"])}
                    return "error", error
                return "result", message.get("result")

    def close(self):
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def model_ids(result):
    found = set()
    for item in (result or {}).get("data") or []:
        if isinstance(item, dict):
            found.update(value for value in (item.get("id"), item.get("model")) if isinstance(value, str))
    return found


def app_server_probe(models, timeout=60):
    """The probe's raw outcome: account, usage window, model catalog and the failure text of each step."""
    server = AppServer(timeout)
    outcome = {"steps": {}}
    try:
        kind, value = server.request(1, "initialize", {"clientInfo": {"name": "decompgym-auth-check", "title": None, "version": "1"}})
        outcome["steps"]["initialize"] = kind
        if kind != "result":
            outcome["failure"] = ("timeout" if kind == "timeout" else "protocol_error", f"initialize: {kind} {json.dumps(value) if value else ''}")
            return outcome
        server.send({"method": "initialized"})
        kind, value = server.request(2, "account/read", {"refreshToken": False})
        outcome["steps"]["account/read"] = kind
        if kind != "result":
            status = classify_text(str(value)) if kind == "error" else ("timeout" if kind == "timeout" else "protocol_error")
            outcome["failure"] = (status, f"account/read: {kind} {(value or {}).get('message', '')}")
            return outcome
        account = (value or {}).get("account")
        outcome["account_type"] = account.get("type") if isinstance(account, dict) else None
        outcome["plan_type"] = account.get("planType") if isinstance(account, dict) else None
        if outcome["account_type"] != "chatgpt":
            outcome["failure"] = ("credential_invalid", f"the app-server reports no ChatGPT account (account type {outcome['account_type']})")
            return outcome
        kind, value = server.request(3, "account/rateLimits/read", {"excludeResetCreditDetails": True})
        outcome["steps"]["account/rateLimits/read"] = kind
        if kind != "result":
            text = (value or {}).get("message", "") if kind == "error" else kind
            status = classify_text(text) if kind == "error" else ("timeout" if kind == "timeout" else "protocol_error")
            if status == "unknown_error":
                status = classify_text("".join(server.stderr)[-4000:])
            outcome["failure"] = (status, f"account/rateLimits/read: {kind} {text}")
            return outcome
        limits = (value or {}).get("rateLimits") or {}
        outcome["ordinary_usage_allowed"] = (value or {}).get("ordinaryUsageAllowed")
        outcome["rate_limit_reached"] = limits.get("rateLimitReachedType")
        outcome["primary_used_percent"] = (limits.get("primary") or {}).get("usedPercent")
        outcome["secondary_used_percent"] = (limits.get("secondary") or {}).get("usedPercent")
        if outcome["ordinary_usage_allowed"] is False or outcome["rate_limit_reached"]:
            outcome["failure"] = ("rate_limited", f"usage window exhausted ({outcome['rate_limit_reached'] or 'ordinary usage not allowed'})")
            return outcome
        if models:
            catalog, cursor = set(), None
            for page in range(10):
                kind, value = server.request(10 + page, "model/list", {"includeHidden": True, "cursor": cursor, "limit": 200})
                outcome["steps"]["model/list"] = kind
                if kind != "result":
                    text = (value or {}).get("message", "") if kind == "error" else kind
                    outcome["failure"] = (classify_text(text) if kind == "error" else ("timeout" if kind == "timeout" else "protocol_error"), f"model/list: {kind} {text}")
                    return outcome
                catalog |= model_ids(value)
                cursor = (value or {}).get("nextCursor")
                if not cursor:
                    break
            outcome["models_missing"] = sorted(model for model in models if model not in catalog)
            if outcome["models_missing"]:
                outcome["failure"] = ("model_missing", f"not in this account's model catalog: {outcome['models_missing']}")
        return outcome
    finally:
        server.close()
        outcome["stderr"] = "".join(server.stderr)  # bounded by the reader; redacted before it is cut


def codex_check(request):
    meta = read_meta()
    result = {"backend": "codex-cli", "inference_submitted": False, "origin": meta.get("origin", "unknown-legacy"), "created_at": meta.get("created_at"),
              "generation": meta.get("generation"), "meta_slot": meta.get("slot"), "credential_refreshed": False}
    try:
        text = read_regular_text(AUTH, CODEX_FILE, missing_ok=True)
    except UnsafeArtifact:
        report(result | {"status": "credential_invalid", "detail": "codex.json is not a single-link regular file"})
        return 0
    if commit_problem():
        report(result | {"status": "commit_incomplete", "detail": "an earlier --login was interrupted during its commit; rerun it with --replace"})
        return 0
    if text is None:
        report(result | {"status": "credential_missing", "detail": "no codex.json in this auth volume"})
        return 0
    try:
        auth = json.loads(text)
    except ValueError:
        report(result | {"status": "credential_invalid", "detail": "codex.json is not JSON"})
        return 0
    problem = codex_shape_problem(auth)
    if problem:
        report(result | {"status": "credential_invalid", "detail": problem})
        return 0
    secrets = secrets_in(auth)
    local = Path.home() / ".codex/auth.json"
    local.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(local, auth)
    outcome = {}
    try:
        outcome = app_server_probe(request.get("models") or [], timeout=float(request.get("timeout", 60)))
    finally:
        # Persist a token the app-server refreshed, so the volume never keeps an already-rotated refresh token. Every
        # value seen afterwards joins the redaction set first, whether or not it is usable.
        try:
            raw = read_regular(local.parent, local.name)
        except (UnsafeArtifact, FileNotFoundError):
            raw = None
        secrets |= token_secrets(raw)
        try:
            updated = json.loads(raw) if raw is not None else None
        except ValueError:
            updated = None
        if updated is None or not isinstance(updated, dict):
            result["refresh_persistence"] = "the local credential was missing or unreadable after the check; the saved one is unchanged but may be stale"
        elif updated != auth:
            if codex_shape_problem(updated) is None:
                write_private(AUTH / CODEX_FILE, updated)
                sync_directory(AUTH)
                result["credential_refreshed"] = True
            else:
                result["refresh_persistence"] = "the app-server left an unusable local credential; the saved one is unchanged but may be stale"
    status, detail = outcome.get("failure") or ("ok", None)
    if result.get("refresh_persistence") and status not in ("credential_invalid", "credential_missing"):
        # The saved refresh token may already be rotated away: a campaign must not start on it.
        status, detail = "refresh_not_persisted", result["refresh_persistence"]
    result.update({key: outcome.get(key) for key in ("account_type", "plan_type", "ordinary_usage_allowed", "rate_limit_reached", "primary_used_percent",
                                                      "secondary_used_percent", "models_missing", "steps")},
                  status=status, detail=redact(detail, secrets)[:800] if detail else None, stderr_tail=redact(outcome.get("stderr") or "", secrets)[-1000:] if status != "ok" else None)
    report(result)
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# Claude Code


def claude_store(request):
    token = request.get("token")
    if not isinstance(token, str) or not token.startswith(SETUP_TOKEN_PREFIX) or len(token) < 40 or any(character.isspace() for character in token):
        report({"stored": False, "status": "credential_invalid", "detail": f"not a Claude setup token (expected the {SETUP_TOKEN_PREFIX}... format of `claude setup-token`)"})
        return 1
    meta = credential_meta("claude-code-cli", request.get("origin", "setup-token-prompt"), volume=request.get("volume"), auth_policy=request.get("policy"), credential_file=CLAUDE_FILE,
                           note="created_at is the storage time, not the token's issue or expiry time")
    try:
        replaced = promote(CLAUDE_FILE, {"oauth_token": token}, replace=bool(request.get("replace")), meta=meta)
    except Refused:
        report({"stored": False, "status": "refused", "detail": "a saved token exists; rerun with --replace"})
        return 3
    legacy = None
    if regular(AUTH / "claude.json"):
        # The worker falls back to a refreshing claudeAiOauth file only without claude_token.json; never leave one.
        legacy = keep_replaced("claude.json", stamp())
        (AUTH / "claude.json").unlink()
        sync_directory(AUTH)
    report({"stored": True, "backend": "claude-code-cli", "origin": meta["origin"], "created_at": meta["created_at"], "replaced_file": replaced, "legacy_file_moved": legacy})
    return 0


def claude_check(request):
    meta = read_meta()
    result = {"backend": "claude-code-cli", "inference_submitted": False, "online_validation": False, "origin": meta.get("origin", "unknown-legacy"),
              "stored_at": meta.get("created_at"), "generation": meta.get("generation"), "warnings": []}
    if commit_problem():
        report(result | {"status": "commit_incomplete", "detail": "an earlier --login was interrupted during its commit; rerun it with --replace"})
        return 0
    try:
        text = read_regular_text(AUTH, CLAUDE_FILE, missing_ok=True)
        legacy = read_regular_text(AUTH, "claude.json", missing_ok=True) is not None
    except UnsafeArtifact:
        report(result | {"status": "credential_invalid", "detail": "a credential path in the auth volume is not a regular file"})
        return 0
    if legacy:
        result["warnings"].append("a legacy claude.json (claudeAiOauth) is present; the worker never uses it while claude_token.json exists")
    if text is None:
        report(result | {"status": "credential_missing", "detail": "no claude_token.json in this auth volume"})
        return 0
    try:
        token = json.loads(text).get("oauth_token")
    except (ValueError, AttributeError):
        token = None
    if not isinstance(token, str) or not token.startswith(SETUP_TOKEN_PREFIX):
        report(result | {"status": "credential_invalid", "detail": f"claude_token.json does not hold a {SETUP_TOKEN_PREFIX}... setup token"})
        return 0
    if result["stored_at"]:
        try:
            result["stored_days"] = round((now() - datetime.fromisoformat(result["stored_at"])).total_seconds() / 86400, 1)
        except ValueError:
            pass
        if result.get("stored_days", 0) > 330:
            result["warnings"].append(f"stored {result['stored_days']} days ago; a setup token is valid for about a year from its creation (not recorded), so plan a replacement")
    result["validity"] = "unknown until a model call (`docker auth --backend claude --verify`); the token's expiry is not recorded"
    report(result | {"status": "ok"})
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# Antigravity


def agy_check(request):
    import agy_credentials as vault
    import agy_login
    result = {"backend": "agy-cli", "inference_submitted": False}
    agy_login.prepare_home()
    if not (agy_login.KEYRINGS / "login.keyring").exists():
        report(result | {"status": "credential_missing", "detail": "no benchmark keyring; run `docker agy-login`"})
        return 0
    try:
        vault.unlock_keyring(request.get("passphrase") or "")
    except RuntimeError as error:
        report(result | {"status": "passphrase_invalid", "detail": str(error)})
        return 0
    restored = vault.restore_native_token()
    secrets = {request["passphrase"]} if request.get("passphrase") else set()
    native = vault.read_native_token()
    if not restored and native is None:
        report(result | {"status": "credential_missing", "detail": "the keyring holds no saved AGY login; run `docker agy-login`"})
        return 0
    secrets |= token_secrets(native)
    sealed = False
    try:
        completed = subprocess.run(["agy", "models"], text=True, capture_output=True, timeout=float(request.get("timeout", 60)), env=agy_login.environment())
        output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", completed.stdout + completed.stderr)
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        output, returncode = "agy models timed out", None
    finally:
        secrets |= token_secrets(vault.read_native_token())  # a refreshed login, too
        sealed = vault.seal_native_token()
    result.update(sealed=sealed, restored=restored)
    status = None if returncode == 0 else ("timeout" if returncode is None else classify_text(output))
    if not sealed and status not in ("credential_invalid", "credential_missing"):
        # A login the catalog call may have refreshed must reach the keyring, or the volume keeps a stale one.
        report(result | {"status": "refresh_not_persisted", "detail": "the AGY login could not be sealed into the keyring after the check"})
        return 0
    if status is not None:
        report(result | {"status": status, "detail": redact(output, secrets)[-300:]})
        return 0
    missing = sorted(model for model in request.get("models") or [] if not re.search(r"(?<![a-z0-9-])" + re.escape(model) + r"(?![a-z0-9-])", output))
    report(result | {"status": "model_missing" if missing else "ok", "models_missing": missing, "detail": f"not in the account catalog: {missing}" if missing else None})
    return 0


MODES = {"codex-import": codex_import, "codex-check": codex_check, "claude-store": claude_store, "claude-check": claude_check, "agy-check": agy_check}


def main():
    mode = sys.argv[1]
    if mode == "codex-login":
        return codex_login(sys.argv[2:])
    if mode not in MODES:
        raise ValueError("Unsupported helper mode")
    line = sys.stdin.readline()
    return MODES[mode](json.loads(line) if line.strip() else {})


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Credential-plane exception values are never exported.
        print(json.dumps({"helper_error": type(error).__name__}), flush=True)
        raise SystemExit(1)
