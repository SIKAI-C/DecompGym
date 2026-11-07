"""docker/agents/auth_helper.py on the host: staged promotion, failed-login safety and the classified Codex app-server check.

The helper is imported with the image's flat layout and talks to a fake `codex` executable that implements the
app-server's stdio JSON-RPC for the four methods the check uses. No Docker engine, network, login or model call runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ACCESS, REFRESH = "SAVED_ACCESS_TOKEN_0123456789abcdef", "SAVED_REFRESH_TOKEN_0123456789abcdef"
SAVED = {"auth_mode": "chatgpt", "OPENAI_API_KEY": None, "tokens": {"access_token": ACCESS, "refresh_token": REFRESH}, "last_refresh": "2026-11-01T00:00:00Z"}

FAKE_CODEX = """#!PYTHON
import json, os, sys
from pathlib import Path
scenario = os.environ.get("FAKE_SCENARIO", "ok")
arguments = sys.argv[1:]
home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
if arguments == ["--version"]:
    print("codex-cli 0.162.1")
    raise SystemExit(0)
if "login" in arguments:
    assert "--device-auth" in arguments and 'cli_auth_credentials_store="file"' in arguments
    if scenario == "login_fail":
        raise SystemExit(1)
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "NEW_ACCESS_TOKEN_0123456789", "refresh_token": "NEW_REFRESH_TOKEN_0123456789"}}))
    raise SystemExit(0)
assert arguments[-1] == "app-server" and 'cli_auth_credentials_store="file"' in arguments
ERRORS = {"unauthorized": "unexpected status 401 Unauthorized: Your authentication token has been invalidated. Please try signing in again.",
          "network": "error sending request for url (https://chatgpt.com/backend-api/wham/usage)", "limited": "unexpected status 429 Too Many Requests"}
for line in sys.stdin:
    message = json.loads(line)
    method, identifier = message.get("method"), message.get("id")
    if identifier is None:
        continue
    print(json.dumps({"method": "account/updated", "params": {}}), flush=True)
    params = message.get("params")
    if method == "initialize":  # the request shapes of the pinned 0.162.1 schema
        assert set(params) <= {"clientInfo", "capabilities"} and {"name", "version"} <= set(params["clientInfo"]) <= {"name", "title", "version"}
        result = {"userAgent": "fake"}
    elif method == "account/read":
        assert message["params"] == {"refreshToken": False}
        result = {"account": None if scenario == "no_account" else {"type": "chatgpt", "email": "owner@example.com", "planType": "pro"}, "requiresOpenaiAuth": True}
    elif method == "account/rateLimits/read":
        assert params is None or set(params) <= {"excludeResetCreditDetails", "supportsLunaReserve"}
        token = json.loads((home / "auth.json").read_text())["tokens"]["access_token"]
        if scenario == "corrupt":
            (home / "auth.json").write_text("{not json")
        if scenario == "null":
            (home / "auth.json").write_text("null")
        if scenario == "half":  # a new access token without a refresh token, also printed in a diagnostic
            (home / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "HALF_REFRESHED_ACCESS_TOKEN_0123456789"}}))
            print("ERROR refreshed bearer HALF_REFRESHED_ACCESS_TOKEN_0123456789", file=sys.stderr, flush=True)
            print(json.dumps({"id": identifier, "error": {"code": -32603, "message": "upstream failure"}}), flush=True)
            continue
        if scenario == "leak":
            print(json.dumps({"id": identifier, "error": {"code": -32603, "message": "unexpected status 401 for bearer " + token}}), flush=True)
            continue
        if scenario in ERRORS:
            print(json.dumps({"id": identifier, "error": {"code": -32603, "message": ERRORS[scenario]}}), flush=True)
            continue
        if scenario == "refresh":
            (home / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "ROTATED_ACCESS_TOKEN_0123456789", "refresh_token": "ROTATED_REFRESH_TOKEN_0123456789"}}))
        result = {"rateLimits": {"primary": {"usedPercent": 12}}, "ordinaryUsageAllowed": scenario != "exhausted"}
    elif method == "model/list":
        assert set(params) <= {"cursor", "includeHidden", "limit"}
        result = {"data": [{"id": "gpt-x", "model": "gpt-x"}], "nextCursor": None}
    else:
        result = None
    print(json.dumps({"id": identifier, "result": result}), flush=True)
"""


@pytest.fixture
def helper(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src/agent"))  # safe_artifacts, as in the image's /opt/decompbench
    spec = importlib.util.spec_from_file_location("auth_helper_under_test", ROOT / "docker/agents/auth_helper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    volume, home, binaries = tmp_path / "volume", tmp_path / "home", tmp_path / "bin"
    for directory in (volume, home, binaries):
        directory.mkdir()
    monkeypatch.setattr(module, "AUTH", volume)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    fake = binaries / "codex"
    fake.write_text(FAKE_CODEX.replace("PYTHON", sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    return module


def last_report(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def save(module, name, value):
    (module.AUTH / name).write_text(json.dumps(value))


def test_promotion_is_staged_refuses_without_replace_and_keeps_the_old_credential(helper):
    save(helper, "codex.json", SAVED)
    with pytest.raises(helper.Refused):
        helper.promote("codex.json", {"new": 1}, replace=False, meta={"origin": "first"})
    assert json.loads((helper.AUTH / "codex.json").read_text()) == SAVED
    first = helper.promote("codex.json", {"new": 1}, replace=True, meta={"origin": "first"})
    second = helper.promote("codex.json", {"new": 2}, replace=True, meta={"origin": "second"})
    assert first.startswith("codex.json.replaced-") and second != first
    assert json.loads((helper.AUTH / first).read_text()) == SAVED and json.loads((helper.AUTH / second).read_text()) == {"new": 1}
    assert json.loads((helper.AUTH / "codex.json").read_text()) == {"new": 2} and helper.read_meta()["origin"] == "second"
    assert len(list(helper.AUTH.glob("meta.json.replaced-*"))) == 1
    # Every file stays a single-link regular file, as the worker's safe reader requires.
    assert all(path.stat().st_nlink == 1 for path in helper.AUTH.iterdir())
    assert not list(helper.AUTH.glob(".*staging*"))


def test_device_login_promotes_only_a_completed_signin(helper, monkeypatch, capsys):
    save(helper, "codex.json", SAVED)
    called = []
    original = helper.subprocess.call
    monkeypatch.setattr(helper.subprocess, "call", lambda *args, **kwargs: called.append(args) or original(*args, **kwargs))
    common = ["--slot", "1", "--volume", "decompbench-v3-codex-auth-slot1", "--policy", "fixture-policy"]
    assert helper.codex_login(common) == 3 and not called  # refused before any sign-in
    monkeypatch.setenv("FAKE_SCENARIO", "login_fail")
    assert helper.codex_login([*common, "--replace"]) == 1
    assert json.loads((helper.AUTH / "codex.json").read_text()) == SAVED and not list(helper.AUTH.glob("*.replaced-*"))
    monkeypatch.setenv("FAKE_SCENARIO", "ok")
    capsys.readouterr()
    assert helper.codex_login([*common, "--replace"]) == 0
    stored = last_report(capsys)
    assert stored["origin"] == "container-device-auth" and stored["slot"] == 1 and stored["replaced_file"].startswith("codex.json.replaced-")
    assert json.loads((helper.AUTH / "codex.json").read_text())["tokens"]["access_token"] == "NEW_ACCESS_TOKEN_0123456789"
    meta = helper.read_meta()
    assert meta["origin"] == "container-device-auth" and meta["cli_version"] == "codex-cli 0.162.1" and meta["auth_policy"] == "fixture-policy" and meta["generation"]
    assert "NEW_ACCESS_TOKEN" not in json.dumps(meta) and "NEW_ACCESS_TOKEN" not in json.dumps(stored)


@pytest.mark.parametrize("scenario,models,status", [("ok", ["gpt-x"], "ok"), ("refresh", ["gpt-x"], "ok"), ("unauthorized", [], "credential_invalid"), ("network", [], "network"),
                                                    ("limited", [], "rate_limited"), ("exhausted", [], "rate_limited"), ("ok", ["gpt-y"], "model_missing"),
                                                    ("no_account", [], "credential_invalid"), ("leak", [], "credential_invalid")])
def test_codex_check_classifies_the_app_server_reads(helper, monkeypatch, capsys, scenario, models, status):
    save(helper, "codex.json", SAVED)
    save(helper, "meta.json", {"origin": "container-device-auth", "slot": 0})
    monkeypatch.setenv("FAKE_SCENARIO", scenario)
    assert helper.codex_check({"models": models, "timeout": 20}) == 0
    output = capsys.readouterr().out
    report = json.loads(output.strip().splitlines()[-1])
    assert report["status"] == status and report["inference_submitted"] is False and report["origin"] == "container-device-auth"
    assert ACCESS not in output and REFRESH not in output and "owner@example.com" not in output
    stored = json.loads((helper.AUTH / "codex.json").read_text())
    if scenario == "refresh":  # a token the app-server rotated is persisted, so the volume never keeps a used refresh token
        assert report["credential_refreshed"] and stored["tokens"]["refresh_token"] == "ROTATED_REFRESH_TOKEN_0123456789"
    else:
        assert stored == SAVED and not report["credential_refreshed"]
    if status == "model_missing":
        assert report["models_missing"] == ["gpt-y"]


def test_codex_check_reports_missing_and_malformed_credentials(helper, capsys):
    assert helper.codex_check({}) == 0 and last_report(capsys)["status"] == "credential_missing"
    save(helper, "codex.json", SAVED | {"OPENAI_API_KEY": "sk-not-a-subscription"})
    helper.codex_check({})
    assert last_report(capsys)["status"] == "credential_invalid"


def test_claude_store_accepts_only_setup_tokens_and_the_check_is_local(helper, capsys):
    token = "sk-ant-oat01-" + "A" * 95
    assert helper.claude_store({"token": "not-a-token"}) == 1 and last_report(capsys)["status"] == "credential_invalid"
    assert helper.claude_store({"token": token, "origin": "setup-token-prompt"}) == 0 and last_report(capsys)["stored"]
    assert helper.claude_store({"token": token + "B"}) == 3 and last_report(capsys)["status"] == "refused"
    (helper.AUTH / "claude.json").write_text("{}")
    helper.claude_check({})
    report = last_report(capsys)
    assert report["status"] == "ok" and report["online_validation"] is False and report["origin"] == "setup-token-prompt" and "legacy claude.json" in report["warnings"][0]
    assert token not in json.dumps(report)


def test_error_text_classification(helper):
    assert helper.classify_text("unexpected status 401 Unauthorized") == "credential_invalid"
    assert helper.classify_text("Your access token could not be refreshed") == "credential_invalid"
    assert helper.classify_text("429 Too Many Requests") == "rate_limited"
    assert helper.classify_text("error sending request: connection refused") == "network"
    assert helper.classify_text("something else") == "unknown_error"


@pytest.mark.parametrize("scenario", ["corrupt", "null", "half"])
def test_a_refresh_the_volume_cannot_persist_blocks_the_check_and_is_redacted(helper, monkeypatch, capsys, scenario):
    save(helper, "codex.json", SAVED)
    save(helper, "meta.json", {"origin": "container-device-auth"})
    monkeypatch.setenv("FAKE_SCENARIO", scenario)
    helper.codex_check({"models": [], "timeout": 20})
    output = capsys.readouterr().out
    report = json.loads(output.strip().splitlines()[-1])
    assert report["status"] == "refresh_not_persisted" and json.loads((helper.AUTH / "codex.json").read_text()) == SAVED
    assert "HALF_REFRESHED_ACCESS_TOKEN" not in output and ACCESS not in output


def test_an_interrupted_commit_is_reported_until_a_new_login_completes(helper, monkeypatch, capsys):
    save(helper, "codex.json", SAVED)
    save(helper, "meta.json", {"origin": "container-device-auth"})
    original = helper.os.replace

    def fail_on_credential(source, destination):
        if Path(destination).name == "codex.json":
            raise OSError("interrupted")
        return original(source, destination)

    monkeypatch.setattr(helper.os, "replace", fail_on_credential)
    with pytest.raises(OSError):
        helper.promote("codex.json", {"new": 1}, replace=True, meta={"origin": "container-device-auth"})
    monkeypatch.setattr(helper.os, "replace", original)
    assert json.loads((helper.AUTH / "codex.json").read_text()) == SAVED and (helper.AUTH / "meta.json.pending").exists()
    helper.codex_check({})
    assert last_report(capsys)["status"] == "commit_incomplete"
    save(helper, "claude_token.json", {"oauth_token": "sk-ant-oat01-" + "A" * 95})
    helper.claude_check({})
    assert last_report(capsys)["status"] == "commit_incomplete"
    helper.promote("codex.json", {"new": 2}, replace=True, meta={"origin": "container-device-auth"})
    assert not (helper.AUTH / "meta.json.pending").exists() and json.loads((helper.AUTH / "codex.json").read_text()) == {"new": 2}


def test_storing_a_claude_token_moves_a_legacy_refreshing_login_aside(helper, capsys):
    (helper.AUTH / "claude.json").write_text(json.dumps({"claudeAiOauth": {"accessToken": "LEGACY_ACCESS_TOKEN_0123456789"}}))
    assert helper.claude_store({"token": "sk-ant-oat01-" + "A" * 95}) == 0
    stored = last_report(capsys)
    assert stored["legacy_file_moved"].startswith("claude.json.replaced-") and not (helper.AUTH / "claude.json").exists()
    helper.claude_check({})
    report = last_report(capsys)
    assert report["status"] == "ok" and not report["warnings"] and report["generation"]


def test_agy_check_never_exports_the_restored_or_refreshed_token(helper, tmp_path, monkeypatch, capsys):
    import types
    keyrings = tmp_path / "keyrings"
    keyrings.mkdir()
    (keyrings / "login.keyring").write_text("encrypted")
    tokens = [json.dumps({"access_token": "AGY_RESTORED_TOKEN_0123456789"}).encode(), json.dumps({"access_token": "AGY_REFRESHED_TOKEN_0123456789"}).encode()]
    vault = types.SimpleNamespace(unlock_keyring=lambda password: None, restore_native_token=lambda: True, read_native_token=lambda: tokens.pop(0) if len(tokens) > 1 else tokens[0],
                                  seal_native_token=lambda: True)
    login = types.SimpleNamespace(prepare_home=lambda: None, KEYRINGS=keyrings, environment=lambda: dict(os.environ))
    monkeypatch.setitem(sys.modules, "agy_credentials", vault)
    monkeypatch.setitem(sys.modules, "agy_login", login)
    agy = Path(os.environ["PATH"].split(os.pathsep)[0]) / "agy"
    agy.write_text(f"#!{sys.executable}\nimport sys\nprint('Please sign in: AGY_RESTORED_TOKEN_0123456789 AGY_REFRESHED_TOKEN_0123456789 benchmark-passphrase', file=sys.stderr)\nraise SystemExit(1)\n")
    agy.chmod(0o755)
    helper.agy_check({"passphrase": "benchmark-passphrase", "models": ["gemini-x"]})
    output = capsys.readouterr().out
    report = json.loads(output.strip().splitlines()[-1])
    assert report["status"] == "credential_invalid" and report["sealed"]
    assert "AGY_RESTORED_TOKEN" not in output and "AGY_REFRESHED_TOKEN" not in output and "benchmark-passphrase" not in output
    # A successful catalog whose login cannot be sealed back does not pass.
    agy.write_text(f"#!{sys.executable}\nprint('gemini-x')\n")
    tokens[:] = [b'{"access_token": "AGY_RESTORED_TOKEN_0123456789"}', b'{"access_token": "AGY_RESTORED_TOKEN_0123456789"}']
    vault.seal_native_token = lambda: False
    helper.agy_check({"passphrase": "benchmark-passphrase", "models": ["gemini-x"]})
    assert last_report(capsys)["status"] == "refresh_not_persisted"
    # Also when the catalog call itself failed on the network: the unsealed login decides.
    agy.write_text(f"#!{sys.executable}\nimport sys\nprint('error sending request: connection refused', file=sys.stderr)\nraise SystemExit(1)\n")
    tokens[:] = [b'{"access_token": "AGY_RESTORED_TOKEN_0123456789"}', b'{"access_token": "AGY_RESTORED_TOKEN_0123456789"}']
    helper.agy_check({"passphrase": "benchmark-passphrase", "models": ["gemini-x"]})
    assert last_report(capsys)["status"] == "refresh_not_persisted"
