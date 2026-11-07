"""Multi-file delivery (PDPTW C++ track): the container validator matches the Phoenix validator, the worker rejects unsafe or
credential-bearing deliveries, the host re-validates every bundle, and only explicitly configured tasks can use the mode.

Pure Python: the agent CLI is a tiny fake that writes files; no Docker engine, model call, compiler, solver.py or optimizer runs.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import pytest

from src.agent import multifile
from src.agent.container import ContainerResult, DockerRuntime
from src.agent.docker_client import CLI_VERSIONS, DockerCliClient, worker_file_hashes
from src.agent.multifile import (BUNDLE_NAME, CPP_NOTE, MULTI_FILE, SINGLE_FILE_DELIVERY, DeliverySpec, MultiFileDockerCliClient, MultiFileDockerRuntime,
                                 delivery_spec, extension_file_hashes, load_delivery, verify_bundle)
from src.experiments import xl_build
from src.experiments.xl_build import DeliveryError

ROOT = Path(__file__).resolve().parents[1]
SECRET = "PRIVATE_CREDENTIAL_VALUE_0123456789"
CPP, PY = "cpp_allowed", "python_only"


@pytest.fixture(scope="module")
def mw():
    """The container module, imported on the host with the image's flat /opt/decompbench layout."""
    added = [str(ROOT / "docker/agents"), str(ROOT / "src/agent")]
    sys.path[:0] = added
    try:
        module = importlib.import_module("multifile_worker")
    finally:
        for path in added:
            sys.path.remove(path)
    return module


def build(root: Path, files: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, tuple) and value[0] == "symlink":
            path.symlink_to(value[1])
        elif isinstance(value, tuple) and value[0] == "dir":
            path.mkdir()
        elif isinstance(value, tuple) and value[0] == "fifo":
            os.mkfifo(path)
        else:
            path.write_bytes(value if isinstance(value, bytes) else value.encode())
    return root


def both(mw, root, track):
    """(worker outcome, xl_build outcome) as comparable values."""
    outcomes = []
    for collect in (lambda: mw.collect_workspace(root, track), lambda: xl_build.collect_delivery(root, track=track, strict=False)):
        try:
            manifest, files = collect()
            outcomes.append(("ok", manifest, files))
        except (mw.DeliveryViolation, DeliveryError) as error:
            outcomes.append(("error", str(error), error.rejected))
    return outcomes


BIG = b"x" * (2 * 1024 * 1024)
CASES = {
    "solver_only": ({"solver.py": "print(1)\n"}, (CPP, PY)),
    "sources": ({"solver.py": "import os\n", "cpp/core.cpp": "int f(){return 1;}\n", "cpp/core.hpp": "int f();\n", "cpp/a-b.c_d.cpp": "\n"}, (CPP, PY)),
    "empty_cpp": ({"solver.py": "x\n", "cpp": ("dir",)}, (CPP, PY)),
    "unsafe_entries": ({"solver.py": "x\n", "cpp/ok.cpp": "1", "cpp/link.cpp": ("symlink", "/etc/passwd"), "cpp/sub": ("dir",), "cpp/x.h": "h", "cpp/.hidden.cpp": "h",
                        "cpp/pipe.cpp": ("fifo",), "notes.md": "n", "escape": ("symlink", ".."), "build": ("dir",)}, (CPP,)),
    "solver_symlink": ({"solver.py": ("symlink", "other.py"), "other.py": "x"}, (CPP, PY)),
    "solver_missing": ({"cpp/core.cpp": "1"}, (CPP,)),
    "cpp_is_file": ({"solver.py": "x", "cpp": "not a dir"}, (CPP, PY)),
    "too_many_sources": ({"solver.py": "x", **{f"cpp/f{i:02d}.cpp": "1" for i in range(65)}}, (CPP,)),
    "file_too_large": ({"solver.py": "x", "cpp/big.cpp": BIG + b"y"}, (CPP,)),
    "total_too_large": ({"solver.py": "x", **{f"cpp/b{i}.cpp": BIG for i in range(4)}}, (CPP,)),
    "too_many_entries": ({"solver.py": "x", **{f"junk{i:03d}": "1" for i in range(300)}}, (CPP,)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_worker_validator_matches_the_phoenix_validator(mw, tmp_path, name):
    files, tracks = CASES[name]
    root = build(tmp_path / "work", files)
    for track in tracks:
        worker_side, phoenix_side = both(mw, root, track)
        assert worker_side == phoenix_side, (name, track)


def test_python_only_track_rejects_any_cpp_source(mw, tmp_path):
    root = build(tmp_path / "work", CASES["sources"][0])
    outcome, message, rejected = both(mw, root, PY)[0]
    assert outcome == "error" and "not allowed in the python_only track" in message and rejected[0]["reason"] == "track"


def test_hard_links_are_rejected_and_manifest_is_sorted(mw, tmp_path):
    root = build(tmp_path / "work", {"solver.py": "x", "cpp/z.cpp": "z", "cpp/a.cpp": "a", "cpp/m.hpp": "m"})
    os.link(root / "cpp/z.cpp", root / "cpp/zz.cpp")
    manifest, files = mw.collect_workspace(root, CPP)
    assert [entry["path"] for entry in manifest["files"]] == ["cpp/a.cpp", "cpp/m.hpp", "solver.py"]
    assert {item["path"] for item in manifest["rejected"]} == {"cpp/z.cpp", "cpp/zz.cpp"}
    assert manifest == xl_build.collect_delivery(root, track=CPP, strict=False)[0]


def test_container_constants_match_the_phoenix_constants(mw):
    for name in ("MANIFEST_VERSION", "TRACKS", "MAX_FILE_BYTES", "MAX_TOTAL_BYTES", "MAX_SOURCES", "MAX_DIRECTORY_ENTRIES"):
        assert getattr(mw, name) == getattr(xl_build, name), name
    assert mw.SOURCE_NAME.pattern == xl_build.SOURCE_NAME.pattern
    for path in ("solver.py", "cpp/a.cpp", "cpp/a.h", "../x", "/solver.py", "cpp/s/a.cpp", "cpp\\a.cpp", "", "a.py"):
        for track in (CPP, PY):
            assert mw.path_problem(path, track) == xl_build.path_problem(path, track)
    assert mw.RECORD_VERSION == multifile.RECORD_VERSION and mw.BUNDLE_NAME == multifile.BUNDLE_NAME


# ------------------------------------------------------------------------------------------- worker agent and collector

FAKE_CLI = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    plan = json.loads(os.environ["FAKE_PLAN"])
    for relative, text in plan.items():
        path = Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    sys.stdin.read()
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}), flush=True)
""")


@pytest.fixture
def container(mw, tmp_path, monkeypatch):
    """Run the worker's real agent/collect code against temporary /work and /record with a fake CLI and credential."""
    work, record = tmp_path / "work", tmp_path / "record"
    work.mkdir()
    record.mkdir()
    worker = mw.worker
    for module in (worker, mw):
        monkeypatch.setattr(module, "WORK", work)
        monkeypatch.setattr(module, "RECORD", record)
    script = tmp_path / "fake_cli.py"
    script.write_text(FAKE_CLI)
    monkeypatch.setattr(worker, "limits", lambda: None)
    monkeypatch.setattr(worker, "credentials", lambda backend, job: (os.environ.copy() | {"FAKE_PLAN": job["fake_plan"]}, {job.get("fake_secret", SECRET)}, lambda: None))
    monkeypatch.setattr(worker, "cli_command", lambda backend, job: [sys.executable, str(script)])

    def heartbeat(last_seen, stopped):
        while not stopped.wait(0.05):
            last_seen[0] = time.monotonic()

    monkeypatch.setattr(worker, "monitor_owner", heartbeat)

    def run(plan, track=CPP, secret=SECRET):
        job = {"fake_secret": secret, "target": "solver.py", "initial_content": "", "prompt": "fixture", "model": "fixture", "effort": "high", "role": "generation", "timeout": 30,
               "fake_plan": json.dumps(plan), "delivery": {"mode": MULTI_FILE, "track": track}}
        assert mw.run_agent("codex-cli", job) == 0
        return json.loads((record / mw.RECORD_NAME).read_text())

    def collect(capsys):
        capsys.readouterr()
        mw.collect(["solver.py", "worker.json", "delivery.json", BUNDLE_NAME])
        return json.loads(capsys.readouterr().out)["artifacts"]

    return type("Container", (), {"run": staticmethod(run), "collect": staticmethod(collect), "work": work, "record": record})


def test_worker_delivers_a_hashed_manifest_the_host_accepts(container, capsys):
    plan = {"solver.py": "import ctypes\n", "cpp/core.cpp": "extern \"C\" int f(){return 2;}\n", "cpp/core.hpp": "int f();\n", "scratch.txt": "notes"}
    record = container.run(plan)
    assert record["error"] is None and [entry["path"] for entry in record["manifest"]["files"]] == ["cpp/core.cpp", "cpp/core.hpp", "solver.py"]
    assert record["manifest"]["rejected"] == [{"path": "scratch.txt", "reason": "not an allowed delivery path (solver.py, cpp/*.cpp, cpp/*.hpp)"}]
    artifacts = container.collect(capsys)
    manifest, files = verify_bundle(artifacts[BUNDLE_NAME], artifacts["solver.py"], CPP)
    assert files == {path: text.encode() for path, text in plan.items() if path != "scratch.txt"}
    assert manifest == xl_build.collect_delivery(container.work, track=CPP, strict=False)[0]
    assert json.loads(artifacts["worker.json"])["candidate_present"]


@pytest.mark.parametrize("plan, track, reason", [
    ({"solver.py": "x\n", "cpp/core.cpp": "1"}, PY, "not allowed in the python_only track"),
    ({"solver.py": "x\n", "cpp/leak.cpp": "// " + SECRET}, CPP, "credential_in_delivery"),
    ({"solver.py": "x\n", "cpp/" + SECRET + ".cpp": "1"}, CPP, "credential_in_delivery"),
    ({"solver.py": "token = '" + SECRET + "'\n"}, CPP, "solver.py is required"),
    ({"solver.py": "x\n", **{f"cpp/f{i:02d}.cpp": "1" for i in range(65)}}, CPP, "at most 64 C++ files"),
])
def test_worker_rejects_contract_violations_without_exporting_files(container, capsys, plan, track, reason):
    record = container.run(plan, track)
    assert record["manifest"] is None and reason in record["error"]
    text = container.collect(capsys)[BUNDLE_NAME]
    assert SECRET not in text and json.loads(text)["files"] == {}
    with pytest.raises(DeliveryError, match="delivery rejected"):
        verify_bundle(text, "x\n", track)


def test_files_changed_after_validation_are_not_delivered(container, capsys):
    container.run({"solver.py": "x\n", "cpp/core.cpp": "int a;\n"})
    (container.work / "cpp/core.cpp").write_text("int b;\n")
    bundle = json.loads(container.collect(capsys)[BUNDLE_NAME])
    assert bundle["manifest"] is None and bundle["files"] == {} and "changed after validation" in bundle["error"]


def test_worker_requires_a_declared_multifile_job(mw):
    for delivery in (None, {"mode": "single_file", "track": CPP}, {"mode": MULTI_FILE, "track": "other"}, {"mode": MULTI_FILE, "track": CPP, "extra": 1}):
        with pytest.raises(ValueError):
            mw.run_agent("codex-cli", {"target": "solver.py", "delivery": delivery})
    with pytest.raises(ValueError):
        mw.run_agent("codex-cli", {"target": "debug.md", "delivery": {"mode": MULTI_FILE, "track": CPP}})


# ------------------------------------------------------------------------------------------------ host verification

def bundle_for(mw, files: dict[str, bytes], track=CPP, rejected=()):
    manifest = mw.manifest_for(files, list(rejected), track)
    record = {"version": mw.RECORD_VERSION, "mode": MULTI_FILE, "track": track, "manifest": manifest, "error": None, "error_rejected": [],
              "files": {path: base64.b64encode(contents).decode() for path, contents in files.items()}}
    return json.dumps(record), manifest


FILES = {"solver.py": b"import os\n", "cpp/core.cpp": b"int f(){return 1;}\n"}


def tamper(record, how):
    if how == "hash":
        record["manifest"]["files"][0]["sha256"] = "0" * 64
    elif how == "extra_file":
        record["files"]["cpp/extra.cpp"] = base64.b64encode(b"1").decode()
    elif how == "missing_file":
        del record["files"]["cpp/core.cpp"]
    elif how == "content":
        record["files"]["cpp/core.cpp"] = base64.b64encode(b"int g;\n").decode()
    elif how == "base64":
        record["files"]["cpp/core.cpp"] = "***"
    elif how == "track":
        record["track"] = record["manifest"]["track"] = PY
    elif how == "path":
        record["manifest"]["files"][0]["path"] = "../core.cpp"
        record["files"]["../core.cpp"] = record["files"].pop("cpp/core.cpp")
    elif how == "digest":
        record["manifest"]["sha256"] = "0" * 64
    elif how == "version":
        record["version"] = "other"
    elif how == "entry_without_path":
        record["manifest"]["files"].append({"size": 1})
    elif how == "files_not_list":
        record["manifest"]["files"] = "solver.py"
    elif how == "rejected_not_list":
        record["manifest"]["rejected"] = None
    return record


@pytest.mark.parametrize("how", ["hash", "extra_file", "missing_file", "content", "base64", "track", "path", "digest", "version", "entry_without_path", "files_not_list", "rejected_not_list"])
def test_host_rejects_any_bundle_that_differs_from_the_canonical_manifest(mw, how):
    text, _ = bundle_for(mw, FILES)
    with pytest.raises(DeliveryError):
        verify_bundle(json.dumps(tamper(json.loads(text), how)), "import os\n", CPP)


def test_host_accepts_the_canonical_bundle_and_xl_build_consumes_it(mw, tmp_path):
    text, manifest = bundle_for(mw, FILES, rejected=[{"path": "notes.md", "reason": "not allowed"}])
    verified, files = verify_bundle(text, "import os\n", CPP)
    assert verified == manifest == xl_build.manifest_for(FILES, [{"path": "notes.md", "reason": "not allowed"}], track=CPP)[0]
    xl_build.materialize(verified, files, tmp_path / "attempt")
    assert (tmp_path / "attempt/cpp/core.cpp").read_bytes() == FILES["cpp/core.cpp"]
    with pytest.raises(DeliveryError, match="solver.py differs"):
        verify_bundle(text, "import sys\n", CPP)


# ------------------------------------------------------------------------------------------------ host client

class FakeRuntime(MultiFileDockerRuntime):
    def __init__(self, root, backend, bundle, solver="import os\n"):
        super().__init__(seccomp_profile=root / "profile.json")
        self.seccomp_profile.write_text("{}")
        self.backend, self.bundle, self.solver, self.jobs = backend, bundle, solver, []

    def ensure_volume(self, *args, **kwargs):
        pass

    def volume_users(self, volume):
        return []  # the lock guard's probe; no Docker engine in this test

    def run(self, image, command, **kwargs):
        kwargs["state_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["state_path"].write_text("{}")
        if command == ["preflight"]:
            value = {"ok": True, "versions": {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]},
                     "worker_files_sha256": worker_file_hashes(), "extension_files_sha256": multifile.extension_file_hashes()}
            return ContainerResult(0, "", "", 0, artifacts={"worker.json": json.dumps(value)})
        self.jobs.append(kwargs["payload"])
        assert (BUNDLE_NAME in kwargs["artifacts"]) == ("delivery" in kwargs["payload"])
        worker = {"cli_version": CLI_VERSIONS[self.backend], "requested_model": "fixture", "target": "solver.py", "returncode": 0, "terminal_state": "completed", "error": None,
                  "usage": {}, "usage_available": True, "warnings": [], "observed_models": ["fixture"], "agent_seconds": 1.0}
        artifacts = {"solver.py": self.solver, "worker.json": json.dumps(worker), "delivery.json": json.dumps({"rejected_target": None, "misplaced": {}})}
        if self.bundle is not None:
            artifacts[BUNDLE_NAME] = self.bundle
        self.result = ContainerResult(0, "", "", 1, artifacts=artifacts, metadata={"cleanup_verified": True})
        return self.result

    def recover(self, state_path):
        return self.result


@pytest.fixture(autouse=True)
def clear_preflights():
    multifile._PREFLIGHTS.clear()
    yield
    multifile._PREFLIGHTS.clear()


def client_for(tmp_path, bundle, track=CPP, backend="codex-cli", solver="import os\n"):
    runtime = FakeRuntime(tmp_path, backend, bundle, solver)
    return MultiFileDockerCliClient("fixture", backend=backend, image_id="sha256:fixture", runtime=runtime, evidence_root=tmp_path / "evidence",
                                    delivery=DeliverySpec(MULTI_FILE, track), credential_provider=lambda _: {}), runtime


def test_client_commits_solver_sources_and_manifest(mw, tmp_path):
    text, manifest = bundle_for(mw, FILES)
    client, runtime = client_for(tmp_path, text)
    target = tmp_path / "attempt/solver.py"
    target.parent.mkdir()
    (target.parent / "cpp").mkdir()
    (target.parent / "cpp/stale.cpp").write_text("stale")
    result = client.edit_file(target, "Solve.", log_dir=tmp_path / "attempt/calls/one")
    assert result.ok and result.artifact_sha256 == hashlib.sha256(FILES["solver.py"]).hexdigest()
    assert runtime.jobs[0]["delivery"] == {"mode": MULTI_FILE, "track": CPP}
    assert runtime.jobs[0]["prompt"].startswith("Only edit the existing solver.py in the current working directory. " + CPP_NOTE)
    assert sorted(p.name for p in (target.parent / "cpp").iterdir()) == ["core.cpp"]
    loaded = load_delivery(target, tmp_path / "attempt/calls/one")
    assert loaded == (manifest, FILES)
    invocation = json.loads((tmp_path / "attempt/calls/one/agent_invocation.json").read_text())
    assert invocation["delivery_manifest_sha256"] == manifest["sha256"] and invocation["transport_policy"] == multifile.MULTIFILE_POLICY
    workspace = json.loads((tmp_path / "attempt/calls/one/agent_workspace.json").read_text())
    assert workspace["version"] == multifile.CPP_WORKSPACE_POLICY
    recovered = client.recover_file(target, log_dir=tmp_path / "attempt/calls/one", step="agent")
    assert recovered.ok and load_delivery(target, tmp_path / "attempt/calls/one") == (manifest, FILES)
    target.write_text("import sys\n")
    with pytest.raises(DeliveryError, match="committed solver.py differs"):
        load_delivery(target, tmp_path / "attempt/calls/one")


@pytest.mark.parametrize("bundle_kind", ["missing", "rejected", "tampered"])
def test_client_treats_an_invalid_delivery_as_a_completed_generation_without_a_file(mw, tmp_path, bundle_kind):
    text, _ = bundle_for(mw, FILES)
    bundle = {"missing": None, "rejected": json.dumps({"version": mw.RECORD_VERSION, "mode": MULTI_FILE, "track": CPP, "manifest": None, "error": "at most 64 C++ files", "error_rejected": [], "files": {}}),
              "tampered": json.dumps(tamper(json.loads(text), "content"))}[bundle_kind]
    client, _ = client_for(tmp_path, bundle)
    target = tmp_path / "attempt/solver.py"
    (target.parent / "cpp").mkdir(parents=True)
    target.write_text("print('previous attempt')\n")
    (target.parent / "cpp/stale.cpp").write_text("stale")
    result = client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    assert not result.ok and result.terminal_state == "completed" and result.returncode == 0 and result.artifact_sha256 is None
    assert "multi-file contract" in result.error
    assert target.read_text() == "" and not (target.parent / "cpp").exists() and not (tmp_path / "call/agent_delivery_files.json").exists()
    metrics = json.loads((tmp_path / "call/agent_metrics.json").read_text())
    assert metrics["error"] == result.error
    record = json.loads((tmp_path / "call/agent_delivery_manifest.json").read_text())
    assert record["manifest"] is None and record["failure"]["error"]
    with pytest.raises(DeliveryError):
        load_delivery(target, tmp_path / "call")


def test_python_only_track_sends_the_paper_prompt(mw, tmp_path):
    text, _ = bundle_for(mw, {"solver.py": b"import os\n"}, track=PY)
    (tmp_path / "multi").mkdir()
    client, runtime = client_for(tmp_path / "multi", text, track=PY)
    target = tmp_path / "multi/attempt/solver.py"
    target.parent.mkdir()
    assert client.edit_file(target, "Solve.", log_dir=tmp_path / "multi/call").ok

    class SingleRuntime:
        seccomp_profile = runtime.seccomp_profile

        def ensure_volume(self, *args, **kwargs):
            pass

        def run(self, image, command, **kwargs):
            return runtime.run(image, command, **kwargs)

    single = DockerCliClient("fixture", backend="codex-cli", image_id="sha256:other", runtime=SingleRuntime(), evidence_root=tmp_path / "evidence2", credential_provider=lambda _: {})
    (tmp_path / "single").mkdir()
    reference = tmp_path / "single/solver.py"
    single.edit_file(reference, "Solve.", log_dir=tmp_path / "single/call")
    assert runtime.jobs[0]["prompt"] == runtime.jobs[1]["prompt"]
    for name in ("agent_prompt.md", "agent_prompt_before_workspace.md", "agent_workspace.json"):
        assert (tmp_path / "multi/call" / name).read_text() == (tmp_path / "single/call" / name).read_text()


def test_preflight_requires_the_extension_hash(mw, tmp_path, monkeypatch):
    text, _ = bundle_for(mw, FILES)
    client, runtime = client_for(tmp_path, text)
    original = runtime.run

    def stale_image(image, command, **kwargs):
        result = original(image, command, **kwargs)
        if command == ["preflight"]:
            observed = json.loads(result.artifacts["worker.json"]) | {"extension_files_sha256": {"multifile_worker.py": "0" * 64}}
            result.artifacts["worker.json"] = json.dumps(observed)
        return result

    monkeypatch.setattr(runtime, "run", stale_image)
    with pytest.raises(multifile.RuntimeEnvironmentError):
        client.prepare()


# --------------------------------------------------------------------------------------------- mode selection

def test_multifile_is_opt_in_and_limited_to_large_scale_pdptw():
    assert delivery_spec("pdptw", "xl") is SINGLE_FILE_DELIVERY
    assert delivery_spec("pdptw", "xl", CPP) == DeliverySpec(MULTI_FILE, CPP)
    assert delivery_spec("pdptw", "xl", PY) == DeliverySpec(MULTI_FILE, PY)
    for problem, scale in (("pdptw", "small"), ("cvrp", "xl"), ("ufl", "small"), ("bp", "small")):
        assert delivery_spec(problem, scale) is SINGLE_FILE_DELIVERY
        with pytest.raises(ValueError):
            delivery_spec(problem, scale, CPP)
    with pytest.raises(ValueError):
        delivery_spec("pdptw", "xl", "rust_allowed")
    for mode, track in (("single_file", CPP), (MULTI_FILE, None), ("other", None)):
        with pytest.raises(ValueError):
            DeliverySpec(mode, track)


def test_multifile_client_refuses_the_single_file_configuration(tmp_path):
    runtime = FakeRuntime(tmp_path, "codex-cli", None)
    with pytest.raises(ValueError):
        MultiFileDockerCliClient("fixture", backend="codex-cli", image_id="sha256:x", runtime=runtime, evidence_root=tmp_path, delivery=SINGLE_FILE_DELIVERY)
    with pytest.raises(ValueError):
        MultiFileDockerCliClient("fixture", backend="codex-cli", image_id="sha256:x", runtime=DockerRuntime(seccomp_profile=tmp_path / "p"), evidence_root=tmp_path,
                                 delivery=DeliverySpec(MULTI_FILE, CPP))


def test_runtimes_enter_their_own_worker():
    assert DockerRuntime.WORKER == "/opt/decompbench/worker.py" and DockerRuntime.COLLECT_OUTPUT_BYTES == 8 * 1024 * 1024
    assert MultiFileDockerRuntime.WORKER == "/opt/decompbench/multifile_worker.py" and MultiFileDockerRuntime.COLLECT_OUTPUT_BYTES >= 2 * 8 * 1024 * 1024
    dockerfile = (ROOT / "docker/agents/Dockerfile.multifile").read_text()
    assert "COPY --chmod=0555 docker/agents/multifile_worker.py /opt/decompbench/" in dockerfile
    assert "multifile_worker" not in (ROOT / "docker/agents/Dockerfile").read_text()


def test_multifile_runtime_changes_only_the_entry_and_collector_ceiling(tmp_path, monkeypatch):
    from tests.test_single_file_golden import capture_runtime
    (tmp_path / "paper").mkdir()
    (tmp_path / "multi").mkdir()
    paper = capture_runtime(tmp_path.resolve() / "paper", monkeypatch)
    multi = capture_runtime(tmp_path.resolve() / "multi", monkeypatch, cls=MultiFileDockerRuntime)
    swap = lambda value: json.loads(json.dumps(value).replace("/opt/decompbench/worker.py", "/opt/decompbench/multifile_worker.py"))
    assert multi["commands"] == swap(paper["commands"]) and "/opt/decompbench/multifile_worker.py" in sum(multi["commands"], [])
    assert [entry["args"] for entry in multi["processes"]] == [entry["args"] for entry in swap(paper["processes"])]
    assert [entry["max_output_bytes"] for entry in multi["processes"]] == [8 * 1024 * 1024, 128 * 1024 * 1024]
    assert [entry["max_output_bytes"] for entry in paper["processes"]] == [8 * 1024 * 1024, 8 * 1024 * 1024]
    assert multi["records"] == swap(paper["records"])


UNICODE_SECRET = 'mot-de-passe-\u00e9-"quoted"-0123456789'


@pytest.mark.parametrize("plan", [
    {"solver.py": "x\n", "notes-" + UNICODE_SECRET.replace('"', "") : "rejected name"},
    {"solver.py": "x\n", "cpp/" + "k" * 10 + ".cpp": "// " + UNICODE_SECRET.replace('"', "")},
])
def test_escaped_credentials_are_found_in_names_and_contents(container, capsys, plan):
    secret = UNICODE_SECRET.replace('"', "")
    record = container.run(plan, secret=secret)
    assert record["manifest"] is None and record["error"] == "credential_in_delivery"
    text = container.collect(capsys)[BUNDLE_NAME]
    assert secret not in json.loads(text)["error"] and secret not in json.dumps(json.loads(text), ensure_ascii=False)


def test_misplaced_paths_holding_credentials_are_dropped(container, capsys):
    record = container.run({"solver.py": "", SECRET + "/solver.py": "print(1)\n", "nested/solver.py": "print(2)\n"})
    assert record["manifest"] is None
    report = json.loads((container.record / "delivery.json").read_text())
    assert list(report["misplaced"]) == ["nested/solver.py"] and report["misplaced_dropped_for_credentials"] == 1
    capsys.readouterr()
    container.collect(capsys)
    assert SECRET not in (container.record / "delivery.json").read_text()


def test_solver_is_exported_only_from_a_verified_bundle(container, capsys):
    container.run({"solver.py": "x = 1\n"})
    assert container.collect(capsys)["solver.py"] == "x = 1\n"
    (container.work / "solver.py").write_text("x = 2\n")
    artifacts = container.collect(capsys)
    assert "solver.py" not in artifacts and "changed after validation" in json.loads(artifacts[BUNDLE_NAME])["error"]


def test_case_colliding_sources_stay_canonical_without_an_inspection_copy(mw, tmp_path):
    files = {"solver.py": b"x\n", "cpp/Core.cpp": b"1", "cpp/core.cpp": b"2"}
    text, manifest = bundle_for(mw, files)
    client, _ = client_for(tmp_path, text, solver="x\n")
    target = tmp_path / "attempt/solver.py"
    target.parent.mkdir()
    assert client.edit_file(target, "Solve.", log_dir=tmp_path / "call").ok
    assert not (target.parent / "cpp").exists()
    assert json.loads((tmp_path / "call/agent_delivery_manifest.json").read_text())["sources_materialized"] is False
    assert load_delivery(target, tmp_path / "call") == (manifest, files)


def test_a_crash_before_the_paper_bookkeeping_leaves_no_success_and_recovers(mw, tmp_path, monkeypatch):
    text, manifest = bundle_for(mw, FILES)
    client, runtime = client_for(tmp_path, text)
    target = tmp_path / "attempt/solver.py"
    target.parent.mkdir()
    original = DockerCliClient._record
    monkeypatch.setattr(DockerCliClient, "_record", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    assert not (tmp_path / "call/agent_metrics.json").exists()
    monkeypatch.setattr(DockerCliClient, "_record", original)
    assert client.recover_file(target, log_dir=tmp_path / "call", step="agent").ok
    assert load_delivery(target, tmp_path / "call") == (manifest, FILES) and len(runtime.jobs) == 1


def test_preflight_is_repeated_when_the_expected_sources_change(mw, tmp_path, monkeypatch):
    text, _ = bundle_for(mw, FILES)
    client, runtime = client_for(tmp_path, text)
    calls = []
    original = runtime.run
    monkeypatch.setattr(runtime, "run", lambda image, command, **kwargs: (calls.append(command), original(image, command, **kwargs))[1])
    client.prepare()
    client.prepare()
    monkeypatch.setattr(multifile, "extension_file_hashes", lambda: {"multifile_worker.py": "1" * 64})
    client.prepare()
    assert calls == [["preflight"], ["preflight"]]


def test_an_exceptional_worker_exit_scrubs_the_report_and_exports_nothing_unfinished(container, capsys, monkeypatch, mw):
    import subprocess as subprocesses
    original = subprocesses.run

    def failing_version(command, *args, **kwargs):
        if command[-1:] == ["--version"]:
            raise subprocesses.TimeoutExpired(command, 10)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(mw.worker.subprocess, "run", failing_version)
    with pytest.raises(subprocesses.TimeoutExpired):
        container.run({"solver.py": "", SECRET + "/solver.py": "print(1)\n"})
    assert SECRET not in (container.record / "delivery.json").read_text()
    artifacts = container.collect(capsys)
    assert not {"delivery.json", BUNDLE_NAME, "solver.py"} & set(artifacts)


def test_unfinished_volumes_export_no_report_bundle_or_solver(container, capsys, mw):
    container.run({"solver.py": "x = 1\n", "cpp/a.cpp": "1"})
    assert {"delivery.json", BUNDLE_NAME, "solver.py"} <= set(container.collect(capsys))
    (container.record / mw.COMPLETE_NAME).unlink()
    assert not {"delivery.json", BUNDLE_NAME, "solver.py"} & set(container.collect(capsys))


@pytest.mark.parametrize("parent_ok", [True, False])
def test_a_crash_after_the_paper_bookkeeping_keeps_metrics_as_the_commit_marker(mw, tmp_path, monkeypatch, parent_ok):
    text, manifest = bundle_for(mw, FILES)
    client, runtime = client_for(tmp_path, text)
    if not parent_ok:  # a container limit makes the paper bookkeeping report an infrastructure failure
        run = runtime.run
        runtime.run = lambda image, command, **kwargs: (lambda result: (setattr(result, "timed_out", command != ["preflight"]), result)[1])(run(image, command, **kwargs))
    target = tmp_path / "attempt/solver.py"
    target.parent.mkdir()
    original = DockerCliClient._record

    def crash_after(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("crash after metrics")

    monkeypatch.setattr(DockerCliClient, "_record", crash_after)
    with pytest.raises(OSError):
        client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    record = json.loads((tmp_path / "call/agent_delivery_manifest.json").read_text())
    if parent_ok:
        assert record["manifest"] == manifest and "sources_materialized" in record
        assert load_delivery(target, tmp_path / "call") == (manifest, FILES)
    else:  # the pre-written partial delivery is withdrawn: no manifest, no committed files, no sources
        assert record["manifest"] is None and record["failure"] and record["sources_materialized"] is False
        assert not (tmp_path / "call/agent_delivery_files.json").exists() and not (target.parent / "cpp").exists()
        with pytest.raises(DeliveryError, match="no committed successful result"):
            load_delivery(target, tmp_path / "call")


def test_durable_preflight_evidence_is_keyed_by_the_whole_preflight_key(mw, tmp_path):
    text, _ = bundle_for(mw, FILES)
    client, runtime = client_for(tmp_path, text)
    paths = []
    original = runtime.run
    runtime.run = lambda image, command, **kwargs: (paths.append(kwargs["state_path"]), original(image, command, **kwargs))[1]
    client.prepare()
    runtime.seccomp_profile.write_text('{"changed": true}')
    client.prepare()
    assert len(paths) == 2 and paths[0] != paths[1] and paths[0].parent == paths[1].parent.parent / paths[0].parent.name


def test_a_terminal_credential_failure_withdraws_the_partial_delivery(mw, tmp_path):
    from src.agent.docker_client import AuthenticationRequired
    text, _ = bundle_for(mw, FILES)
    client, runtime = client_for(tmp_path, text)
    run = runtime.run

    def failing(image, command, **kwargs):
        result = run(image, command, **kwargs)
        if command != ["preflight"]:
            message = "Your access token could not be refreshed. Please log out and sign in again."
            worker = json.loads(result.artifacts["worker.json"]) | {"returncode": 1, "terminal_state": "failed", "error": message}
            result.artifacts.update({"worker.json": json.dumps(worker), "events.jsonl": json.dumps({"type": "turn.failed", "error": {"message": message}}) + "\n"})
            result.returncode = 1
        return result

    runtime.run = failing
    target = tmp_path / "attempt/solver.py"
    target.parent.mkdir()
    with client.bind_auth_slot(1), pytest.raises(AuthenticationRequired):
        client.edit_file(target, "Solve.", log_dir=tmp_path / "call")
    record = json.loads((tmp_path / "call/agent_delivery_manifest.json").read_text())
    assert record["manifest"] is None and record["sources_materialized"] is False
    assert not (tmp_path / "call/agent_delivery_files.json").exists() and not (target.parent / "cpp").exists()
    assert json.loads((tmp_path / "call/agent_metrics.json").read_text())["ok"] is False and target.read_text() == ""
    assert json.loads((tmp_path / "call/agent_auth_invalid.json").read_text())["auth_volume"] == "decompbench-v3-codex-auth-slot1"
