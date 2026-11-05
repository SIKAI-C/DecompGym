"""Multi-file delivery entry of the trusted worker; shipped only in the multi-file agent image (PDPTW C++ track, protocol v4 §5).

The paper worker (`worker.py`) is imported unchanged and keeps doing the whole agent call; the paper image does not contain this
file and never runs it. Modes:

- ``agent <backend>``: the job must declare ``delivery = {"mode": "multi_file", "track": "python_only" | "cpp_allowed"}`` and
  target ``solver.py``. After the paper worker's call, the workspace is validated with the delivery rules below and the result
  (manifest or rejection reason) is written to ``/record/delivery_manifest.json``. A delivery that contains a credential is rejected.
- ``collect <names...>``: the paper collector's contract (``solver.py`` is taken only from a verified bundle); the name ``delivery_bundle.json`` returns the delivery record plus the
  manifest's files (base64), re-read from the read-only volumes after the agent container is gone and checked against the
  manifest's sizes and sha256 (a file that changed after validation rejects the delivery).
- ``preflight``: the paper preflight, plus this file's sha256 in ``worker.json``.

The delivery rules mirror ``src/experiments/xl_build.py`` (``collect_delivery(strict=False)`` and ``manifest_for``): ``solver.py``
plus, in the ``cpp_allowed`` track only, flat ``cpp/*.cpp`` / ``cpp/*.hpp`` (names ``[A-Za-z0-9_][A-Za-z0-9_.-]*``), at most 64 C++
files, 2 MiB per file, 8 MiB in total; no links, special files, nested directories or other paths. Other workspace entries are
recorded as rejected and not delivered; limit and track violations reject the whole delivery. ``tests/test_multifile_delivery.py``
checks that both implementations produce identical manifests.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

sys.path.insert(0, "/opt/decompbench")
import worker  # noqa: E402  (the unchanged paper worker)
from safe_artifacts import UnsafeArtifact, read_regular, read_regular_text  # noqa: E402

WORK, RECORD = worker.WORK, worker.RECORD
RECORD_NAME = "delivery_manifest.json"
COMPLETE_NAME = "multifile_complete.json"  # written last; without it the collector exports no delivery report or bundle
BUNDLE_NAME = "delivery_bundle.json"
RECORD_VERSION = "decompgym-multifile-worker-v1"
MANIFEST_VERSION = "decompgym-xl-delivery-v2"
TRACKS = ("python_only", "cpp_allowed")
SOURCE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}\.(cpp|hpp)")
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_SOURCES = 64
MAX_DIRECTORY_ENTRIES = 256


class DeliveryViolation(ValueError):
    def __init__(self, message, rejected=None):
        super().__init__(message)
        self.rejected = rejected or []


def path_problem(relative, track):
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        return "invalid path"
    if relative.startswith("/"):
        return "absolute path"
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "path traversal or empty component"
    if parts == ["solver.py"]:
        return None
    if len(parts) == 2 and parts[0] == "cpp":
        if track != "cpp_allowed":
            return f"C++ sources are not allowed in the {track} track"
        return None if SOURCE_NAME.fullmatch(parts[1]) else "cpp/ accepts only flat *.cpp and *.hpp names"
    return "not an allowed delivery path (solver.py" + (", cpp/*.cpp, cpp/*.hpp)" if track == "cpp_allowed" else " only)")


def check_files(files, track):
    if track not in TRACKS:
        raise ValueError("unknown track")
    problems = [{"path": path, "reason": reason} for path in files if (reason := path_problem(path, track))]
    if problems:
        raise DeliveryViolation("disallowed delivery paths", problems)
    if "solver.py" not in files:
        raise DeliveryViolation("solver.py is required")
    if sum(1 for path in files if path.startswith("cpp/")) > MAX_SOURCES:
        raise DeliveryViolation(f"at most {MAX_SOURCES} C++ files")
    for path, contents in files.items():
        if not isinstance(contents, (bytes, bytearray)) or len(contents) > MAX_FILE_BYTES:
            raise DeliveryViolation(f"{path}: not bytes or larger than {MAX_FILE_BYTES} bytes")
    if sum(len(contents) for contents in files.values()) > MAX_TOTAL_BYTES:
        raise DeliveryViolation(f"delivery larger than {MAX_TOTAL_BYTES} bytes")


def manifest_for(files, rejected, track):
    check_files(files, track)
    entries = [{"path": path, "size": len(files[path]), "sha256": hashlib.sha256(files[path]).hexdigest()} for path in sorted(files)]
    digest = hashlib.sha256(json.dumps({"track": track, "files": entries}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"version": MANIFEST_VERSION, "track": track, "files": entries, "rejected": list(rejected), "sha256": digest}


def collect_workspace(root, track):
    """Lenient collection of `root` with every limit enforced from lstat sizes before any read; returns (manifest, files)."""
    if track not in TRACKS:
        raise ValueError("unknown track")
    files, rejected, budget = {}, [], {"total": 0, "sources": 0}

    def reject(path, reason):
        rejected.append({"path": path, "reason": reason})

    def listing(path):
        entries = []
        with os.scandir(path) as iterator:
            for entry in iterator:
                entries.append(entry)
                if len(entries) > MAX_DIRECTORY_ENTRIES:
                    raise DeliveryViolation(f"more than {MAX_DIRECTORY_ENTRIES} entries in {Path(path).name or path}")
        return sorted(entries, key=lambda item: item.name)

    def read(relative, size):
        if size > MAX_FILE_BYTES:
            raise DeliveryViolation(f"{relative}: larger than {MAX_FILE_BYTES} bytes")
        if budget["total"] + size > MAX_TOTAL_BYTES:
            raise DeliveryViolation(f"delivery larger than {MAX_TOTAL_BYTES} bytes")
        try:
            contents = read_regular(Path(root), relative, maximum=min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - budget["total"]))
        except UnsafeArtifact as error:
            reject(relative, str(error))
            return
        budget["total"] += len(contents)
        files[relative] = contents

    for entry in listing(str(root)):
        info = entry.stat(follow_symlinks=False)
        if entry.name == "solver.py":
            if stat.S_ISREG(info.st_mode):
                read("solver.py", info.st_size)
            else:
                reject("solver.py", "not a regular file (symlink or special file)")
        elif entry.name == "cpp":
            if not stat.S_ISDIR(info.st_mode):
                reject("cpp", "cpp must be a real directory")
                continue
            for source in listing(entry.path):
                relative = "cpp/" + source.name
                details = source.stat(follow_symlinks=False)
                if track != "cpp_allowed":
                    raise DeliveryViolation(f"C++ sources are not allowed in the {track} track", [{"path": relative, "reason": "track"}])
                if stat.S_ISLNK(details.st_mode):
                    reject(relative, "symlink")
                elif stat.S_ISDIR(details.st_mode):
                    reject(relative, "nested directory")
                elif not stat.S_ISREG(details.st_mode):
                    reject(relative, "special file")
                elif (reason := path_problem(relative, track)) is not None:
                    reject(relative, reason)
                else:
                    budget["sources"] += 1
                    if budget["sources"] > MAX_SOURCES:
                        raise DeliveryViolation(f"at most {MAX_SOURCES} C++ files")
                    read(relative, details.st_size)
        else:
            reject(entry.name, "symlink" if stat.S_ISLNK(info.st_mode) else (path_problem(entry.name, track) or "not allowed"))
    return manifest_for(files, rejected, track), files


def strings(value):
    """Every decoded string in a JSON-like value (keys included), so escaping cannot hide a credential."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def has_secret(value, secrets):
    return any(secret in text for text in strings(value) for secret in secrets)


def scrub(value, secrets):
    """`value` with every string (keys included) redacted before serialization."""
    if isinstance(value, dict):
        return {worker.redact(str(key), secrets): scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, secrets) for item in value]
    return worker.redact(value, secrets) if isinstance(value, str) else value


def delivery_record(root, track, secrets):
    """The validated delivery of `root` as a JSON-safe record; a credential anywhere in it rejects the delivery."""
    record = {"version": RECORD_VERSION, "mode": "multi_file", "track": track, "manifest": None, "error": None, "error_rejected": []}
    try:
        manifest, files = collect_workspace(root, track)
        # Paths, including rejected names, are agent-chosen too.
        if any(secret.encode() in contents for secret in secrets for contents in files.values()) or has_secret(manifest, secrets):
            raise DeliveryViolation("credential_in_delivery")
        record["manifest"] = manifest
    except DeliveryViolation as error:
        record["error"] = worker.redact(str(error), secrets)
        record["error_rejected"] = scrub(error.rejected, secrets)
    except OSError as error:
        record["error"] = f"workspace cannot be read safely (errno {error.errno})"
    return record


def scrub_misplaced_report(secrets):
    """The paper worker redacts misplaced file contents but not their paths; drop entries whose path holds a credential."""
    text = read_regular_text(RECORD, "delivery.json", missing_ok=True)
    if text is None:
        return
    report = json.loads(text)
    misplaced = report.get("misplaced") or {}
    kept = {path: value for path, value in misplaced.items() if not has_secret(path, secrets)}
    if len(kept) != len(misplaced):
        report["misplaced"] = kept
        report["misplaced_dropped_for_credentials"] = len(misplaced) - len(kept)
        worker.write_json(RECORD / "delivery.json", report)


def run_agent(backend, job):
    delivery = job.get("delivery") or {}
    track = delivery.get("track")
    if delivery.get("mode") != "multi_file" or track not in TRACKS or job.get("target") != "solver.py" or set(delivery) != {"mode", "track"}:
        raise ValueError("Multi-file delivery requires solver.py and a declared track")
    captured = []
    original = worker.credentials

    def credentials(*args):
        environment, secrets, checkpoint = original(*args)
        captured.append(secrets)  # the paper worker keeps adding refreshed credentials to this same set
        return environment, secrets, checkpoint

    worker.credentials = credentials
    try:
        returncode = worker.run_agent(backend, job)
    finally:
        worker.credentials = original
        scrub_misplaced_report(set().union(*captured))
    secrets = set().union(*captured)
    worker.write_json(RECORD / RECORD_NAME, delivery_record(WORK, track, secrets))
    worker.write_json(RECORD / COMPLETE_NAME, {"version": RECORD_VERSION, "complete": True})
    return returncode


def complete():
    try:
        text = read_regular_text(RECORD, COMPLETE_NAME, missing_ok=True)
    except UnsafeArtifact:
        return False
    return text is not None and json.loads(text).get("complete") is True


def bundle():
    """The delivery record and its files, re-read now and checked against the record's manifest."""
    text = read_regular_text(RECORD, RECORD_NAME, missing_ok=True) if complete() else None
    if text is None:
        return None
    record = json.loads(text)
    record["files"] = {}
    manifest = record.get("manifest")
    if manifest is not None:
        try:
            for entry in manifest["files"]:
                if path_problem(entry["path"], manifest["track"]) is not None:
                    raise DeliveryViolation("manifest names a disallowed path")
                contents = read_regular(WORK, entry["path"], maximum=MAX_FILE_BYTES)
                if len(contents) != entry["size"] or hashlib.sha256(contents).hexdigest() != entry["sha256"]:
                    raise DeliveryViolation("delivered files changed after validation")
                record["files"][entry["path"]] = base64.b64encode(contents).decode("ascii")
        except (DeliveryViolation, UnsafeArtifact) as error:
            record.update(manifest=None, files={}, error=str(error), error_rejected=[])
    return json.dumps(record, sort_keys=True)


def collect(names):
    """The paper collector's contract; solver.py is exported only from a verified bundle (never re-read independently)."""
    artifacts, rejected = {}, []
    for name in names:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Invalid artifact name")
    packed = bundle() if BUNDLE_NAME in names or "solver.py" in names else None
    for name in names:
        if name in (BUNDLE_NAME, "solver.py"):
            if packed is None:
                continue
            if name == BUNDLE_NAME:
                artifacts[name] = packed
                continue
            encoded = json.loads(packed)["files"].get("solver.py")
            if encoded is not None:
                try:
                    artifacts[name] = base64.b64decode(encoded).decode("utf-8")
                except UnicodeError:
                    rejected.append(name)
            continue
        if name == "delivery.json" and not complete():
            continue  # the paper report holds unscrubbed misplaced paths until the extension has finished
        try:
            value = read_regular_text(RECORD if name in worker.RECORD_FILES | {RECORD_NAME} else WORK, name, missing_ok=True)
        except UnsafeArtifact:
            if name in worker.RECORD_FILES | {RECORD_NAME}:
                raise
            rejected.append(name)
            continue
        if value is not None:
            artifacts[name] = value
    print(json.dumps({"ok": True, "artifacts": artifacts, "rejected": rejected}))
    return 0


def preflight():
    returncode = worker.preflight()
    result = json.loads(read_regular_text(RECORD, "worker.json"))
    result["extension_files_sha256"] = {"multifile_worker.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    worker.write_json(RECORD / "worker.json", result)
    return returncode


def main():
    mode = sys.argv[1]
    if mode == "collect":
        return collect(sys.argv[2:])
    if mode == "preflight":
        return preflight()
    if mode == "agent":
        return run_agent(sys.argv[2], json.loads(sys.stdin.readline()))
    return worker.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Credential-plane exception values are intentionally never exported.
        print(json.dumps({"worker_error": type(error).__name__}))
        raise SystemExit(1)
