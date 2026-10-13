"""Use the established encrypted AGY keyring through a private credential channel."""
import base64
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


NATIVE_TOKEN_RELATIVE = ".gemini/antigravity-cli/antigravity-oauth-token"
VAULT_ATTRIBUTES = ["decompbench-credential", "agy-native-oauth-v1"]
MAX_TOKEN_BYTES = 1024 * 1024


def native_token_path() -> Path:
    return Path.home() / NATIVE_TOKEN_RELATIVE


def read_native_token() -> bytes | None:
    """Read only AGY's exact native token file into memory, never diagnostic output."""
    try:
        descriptor = os.open(native_token_path(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError("Native token path must be a regular file, not a symlink.") from exc
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or not 0 < metadata.st_size <= MAX_TOKEN_BYTES:
            raise RuntimeError("Native token file failed the size or regular-file check; existing encrypted state was preserved.")
        payload = stream.read(MAX_TOKEN_BYTES + 1)
        if not payload or len(payload) > MAX_TOKEN_BYTES:
            raise RuntimeError("Native token file has an invalid size; existing encrypted state was preserved.")
        return payload


def vault_lookup() -> str | None:
    result = subprocess.run(["secret-tool", "lookup", *VAULT_ATTRIBUTES], text=True, capture_output=True, timeout=10)
    if result.returncode == 1 and not result.stderr.strip():
        return None
    if result.returncode:
        raise RuntimeError("Could not read the encrypted AGY state; raw keyring output withheld.")
    return result.stdout.rstrip("\n")


def seal_native_token() -> bool:
    payload = read_native_token()
    if payload is None:
        return False
    encoded = base64.b64encode(payload).decode("ascii")
    stored = subprocess.run(["secret-tool", "store", "--label=DecompBench native AGY OAuth state v1", *VAULT_ATTRIBUTES], input=encoded, text=True, capture_output=True, timeout=10)
    if stored.returncode or vault_lookup() != encoded:
        raise RuntimeError("Could not verify encrypted AGY state persistence; raw keyring output withheld.")
    return True


def restore_native_token() -> bool:
    if read_native_token() is not None:
        return False
    encoded = vault_lookup()
    if encoded is None:
        return False
    if len(encoded) > MAX_TOKEN_BYTES * 2:
        raise RuntimeError("Encrypted AGY state exceeds the supported size; existing state was preserved.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise RuntimeError("Encrypted AGY state has an unsupported format; existing state was preserved.") from exc
    if not payload or len(payload) > MAX_TOKEN_BYTES:
        raise RuntimeError("Encrypted AGY state has an invalid size; existing state was preserved.")
    destination = native_token_path()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".agy-token-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True


def native_state_metadata(*, restored: bool, sealed: bool) -> dict[str, object]:
    return {"native_token_path": NATIVE_TOKEN_RELATIVE, "native_token_file_present": native_token_path().is_file(), "restored_from_encrypted_keyring": restored, "sealed_to_encrypted_keyring": sealed, "native_storage_mode": "file-in-tmpfs-with-encrypted-keyring-persistence", "inference_submitted": False}


def unlock_keyring(password: str) -> None:
    result = subprocess.run(
        ["gnome-keyring-daemon", "--unlock", "--components=secrets"],
        input=password,
        text=True,
        capture_output=True,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError("GNOME Keyring did not start; diagnostic output withheld.")
    locked = subprocess.run(
        ["dbus-send", "--print-reply", "--session", "--dest=org.freedesktop.secrets", "/org/freedesktop/secrets/collection/login", "org.freedesktop.DBus.Properties.Get", "string:org.freedesktop.Secret.Collection", "string:Locked"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    if locked.returncode or "boolean false" not in locked.stdout:
        raise RuntimeError("The benchmark keyring is locked. Check its passphrase.")

