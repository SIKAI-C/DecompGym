"""One-time Antigravity sign-in inside the agent image; seals the native login into the encrypted keyring volume.

Started by `python src/02_experiment.py docker agy-login` (or `agy-models`), never directly. It runs under
dbus-run-session in the agent image with the keyring volume at /auth/keyrings. It submits no model prompt and
prints only booleans, never the passphrase or the token.
"""
import getpass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

sys.path.insert(0, "/opt/decompbench")
import agy_credentials as vault  # noqa: E402


REQUIRED_MODELS = ("gemini-3.8-flash-medium",)
KEYRINGS = Path("/auth/keyrings")


def environment():
    # The same values as the benchmark worker, so AGY selects the same native token file.
    return os.environ | {"SSH_CONNECTION": "127.0.0.1 1 127.0.0.1 22", "AGY_CLI_DISABLE_AUTO_UPDATE": "true"}


def prepare_home():
    keyring = Path.home() / ".local/share/keyrings"
    keyring.parent.mkdir(parents=True, exist_ok=True)
    if not keyring.exists():
        keyring.symlink_to(KEYRINGS, target_is_directory=True)
    settings = Path.home() / ".gemini/antigravity-cli/settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"enableTelemetry": False, "useG1Credits": False, "showTips": False, "showFeedbackSurvey": False, "altScreenMode": "never"}) + "\n")


def unlock(command):
    exists = (KEYRINGS / "login.keyring").exists()
    if command == "models" and not exists:
        raise RuntimeError("No benchmark keyring exists yet; run agy-login first.")
    print("Enter the existing benchmark keyring passphrase (not your Google password)." if exists else "Choose a new passphrase for the dedicated benchmark keyring (not your Google password).", flush=True)
    password = getpass.getpass("Benchmark keyring passphrase: ")
    if not password:
        raise RuntimeError("An empty keyring passphrase is not supported.")
    if not exists and password != getpass.getpass("Repeat the new passphrase: "):
        raise RuntimeError("The passphrases did not match.")
    vault.unlock_keyring(password)


def catalog():
    result = subprocess.run(["agy", "models"], text=True, capture_output=True, timeout=60, env=environment())
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stdout + result.stderr)
    if result.returncode:
        return {"account_catalog_available": False, "sign_in_required": "Please sign in" in output}
    found = {model: bool(re.search(r"(?<![a-z0-9-])" + re.escape(model) + r"(?![a-z0-9-])", output)) for model in REQUIRED_MODELS}
    return {"account_catalog_available": True, "required_models_found": found}


def main():
    command = sys.argv[1]
    if command not in ("login", "models") or not sys.stdin.isatty():
        raise RuntimeError("Run agy-login or agy-models from an interactive terminal.")
    prepare_home()
    unlock(command)
    restored = vault.restore_native_token()
    if command == "login":
        print("Choose 'Login with Google', open the URL in your browser and paste the code here. Once AGY shows its prompt, press Ctrl+C without submitting a task.", flush=True)
        child = subprocess.Popen(["agy", "--log-file", os.devnull], env=environment())
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            child.wait()
        finally:
            signal.signal(signal.SIGINT, previous)
    sealed = False
    try:
        state = catalog()
    finally:
        # Seal even if the catalog check fails, so a refreshed login is not lost.
        sealed = vault.seal_native_token()
    report = state | {"auth_state": vault.native_state_metadata(restored=restored, sealed=sealed)}
    print(json.dumps(report, indent=2))
    ok = sealed and state.get("account_catalog_available") and all(state.get("required_models_found", {}).values())
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
