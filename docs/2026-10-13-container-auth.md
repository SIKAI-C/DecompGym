# Container authentication for the fresh Docker workflow

The new image and its credential-free isolation preflight are ready. Codex has completed one real, unscored Docker-to-native file handoff: the CLI wrote the supplied synthetic program, and the native sandbox solved its small Gurobi model with objective 1.0 and one thread. The saved record is [the Codex readiness report](../outputs/diagnostics/hybrid_runtime_20261013/readiness/codex-cli/9135ba9ef4af4210ae3edd3d4904ceb5/report.json). No production campaign or hourly review was started.

Claude and AGY require owner-supplied authentication in the local terminal. Do not paste a token or passphrase into this conversation. The commands below accept hidden input, transmit credentials through a private channel, and save only nonsecret diagnostic records. `--verify` makes one unscored editing call for the named backend and runs the supplied synthetic Gurobi program; it does not run any of the fixed 81 benchmark cases.

Run the commands from the implementation worktree:

```sh
cd /Users/scheng326/Desktop/GT_ML_PhD/z_oo/code/DecompBench-docker-v2
```

## Claude

Use the existing Claude subscription. If there is no dedicated OAuth token already available locally, run `claude setup-token` in the local terminal. The official [CLI reference](https://code.claude.com/docs/en/cli-reference) describes this command as generating a token for scripts and printing it locally without saving it. Then run the command below and paste that token at its hidden prompt. An API key is not required for this route.

```sh
claude setup-token
../DecompBench/.venv/bin/python -m src.hybrid_runtime auth --backend claude --verify --evidence-root ../DecompBench/outputs/diagnostics/hybrid_runtime_20261013
```

The dedicated Docker authentication volume stores the token privately for later fresh containers. If the volume is already configured, Enter at the token prompt reuses it. The diagnostic uses the approved `claude-sonnet-5-5` model and preserves the existing returned-model check.

## AGY

Run this command and enter the passphrase previously chosen for the benchmark's encrypted AGY keyring. This is the benchmark keyring passphrase, not the Google account password. The historical encrypted volume is retained; the helper restores only its native authentication state into the fresh container.

```sh
../DecompBench/.venv/bin/python -m src.hybrid_runtime auth --backend agy --verify --evidence-root ../DecompBench/outputs/diagnostics/hybrid_runtime_20261013
```

The diagnostic uses `gemini-3.8-flash-medium`. The passphrase is not saved as a plaintext file. The same passphrase will be needed privately in the launch environment when the new campaign is explicitly started; controller/watchdog subprocesses can inherit that private environment, while the native solver's environment excludes it.

## Completion signal

A successful check prints `passed: true` and a report path under `outputs/diagnostics/hybrid_runtime_20261013/readiness/`. Reply only with whether the commands succeeded or the nonsecret error message. Credentials should remain in the local terminal/private store. Authentication loading alone does not prove file-editing readiness; the separately recorded handoff check supplies that evidence.
