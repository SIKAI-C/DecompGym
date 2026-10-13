# Container authentication for the fresh Docker workflow

**Validated October 13 at 13:04 EDT:** Codex, Claude Code and AGY have all passed real, unscored Docker-to-native file handoffs. Each CLI wrote the supplied synthetic program, and the native sandbox returned objective 1.0 with one thread. The current records are [Codex](../outputs/diagnostics/hybrid_runtime_20261013/readiness/codex-cli/9135ba9ef4af4210ae3edd3d4904ceb5/report.json), [Claude with its replacement token](../outputs/diagnostics/hybrid_runtime_20261013/readiness/claude-code-cli/5d2c0e79a1384ca7ae8c14df1c860d29/report.json), and [AGY](../outputs/diagnostics/hybrid_runtime_20261013/readiness/agy-cli/2aa7cdd1b4f3426085b96847b974844b/report.json). The frozen controller accepted all three evidence sets; see [final acceptance](../outputs/diagnostics/hybrid_runtime_20261013/prelaunch-acceptance.json). No production campaign or hourly review was started.

The setup below is retained for future authentication renewal; the completed checks need not be repeated unless credentials or the validated environment change. The commands accept hidden input in the owner's local terminal, transmit credentials through a private channel, and save only nonsecret diagnostic records. `--verify` makes one unscored editing call for the named backend and runs the supplied synthetic Gurobi program; it does not run any of the fixed 81 benchmark cases. Keep tokens and passphrases in the local terminal/private store.

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

When replacing a token, paste the new token at the hidden prompt; Enter would reuse the stored one. The prior token was explicitly revoked and a request using only that stored token received a server-side 401/revoked response at 12:57 EDT on October 13. The replacement's real file handoff began at 13:00:56 EDT and passed. Generating another token alone is not evidence that a previous token has been revoked; [the saved revocation check](../outputs/diagnostics/hybrid_runtime_20261013/credential-revocation/claude-code-cli/ac1efaf183214e54ae109f1f347b4d1b/report.json) records only the outcome, not the credential value.

## AGY

Run this command and enter the passphrase previously chosen for the benchmark's encrypted AGY keyring. This is the benchmark keyring passphrase, not the Google account password. The historical encrypted volume is retained; the helper restores only its native authentication state into the fresh container.

```sh
../DecompBench/.venv/bin/python -m src.hybrid_runtime auth --backend agy --verify --evidence-root ../DecompBench/outputs/diagnostics/hybrid_runtime_20261013
```

The diagnostic uses `gemini-3.8-flash-medium`. The passphrase is not saved as a plaintext file. The same passphrase will be needed privately in the launch environment when the new campaign is explicitly started; controller/watchdog subprocesses can inherit that private environment, while the native solver's environment excludes it.

## Completion signal

A successful check prints `passed: true` and a report path under `outputs/diagnostics/hybrid_runtime_20261013/readiness/`. Reply only with whether the commands succeeded or the nonsecret error message. Credentials should remain in the local terminal/private store. Authentication loading alone does not prove file-editing readiness; the separately recorded handoff check supplies that evidence.
