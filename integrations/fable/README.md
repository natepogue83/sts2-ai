# Fable consultation loop

`scripts/ask-fable.ps1` sends a bounded, read-only consultation to the Fable
model through the locally authenticated Claude CLI. It does not use an API key,
give Claude tools, persist a Claude session, or allow Fable to edit the project.

The system prompt is rebuilt from `ROLE.md` and `OWNER_PROFILE.md` on every
call. Fresh sessions prevent conversational drift; a SHA-256 fingerprint in
each result identifies the exact charter used. The wrapper requests a bounded
JSON contract and validates it locally because Fable 5 does not currently emit
results when Claude Code's `--json-schema` switch is enabled. Only preferences
explicitly stated by the owner should be added to `OWNER_PROFILE.md`.

Example:

```powershell
.\scripts\ask-fable.ps1 `
  -Trigger milestone `
  -Question "Should we treat the new replay importer as complete?" `
  -Context "All focused tests pass, but no full-run benchmark has been recorded yet."
```

The verdicts are:

- `approve`: continue with the proposed next action.
- `revise`: remain in scope but change the approach first.
- `stop`: do not continue with the proposed action.
- `need_user`: the real user must decide because the proxy lacks authority or a
  material preference.

Fable's answer is advice, not new authorization. Codex remains responsible for
checking the evidence, staying within the user's request, and verifying work.
