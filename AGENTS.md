# Project agent instructions

## Fable closed-loop consultation

Use `scripts/ask-fable.ps1` as a read-only user-proxy checkpoint through the
owner's authenticated Claude CLI subscription.

Consult Fable when any of these triggers occurs:

1. `milestone`: a requested outcome or meaningful roadmap milestone appears
   complete and has been verified.
2. `struggling`: two reasonable attempts failed, or diagnostics still leave the
   root cause materially uncertain.
3. `question`: two or more viable approaches create a material product,
   architecture, data-quality, safety, or evaluation tradeoff that the owner's
   known preferences can help resolve.

Do not consult for routine edits, obvious implementation details, or every test
failure. A consultation should include the focused decision, concise evidence,
constraints, and Codex's proposed next action. Example:

```powershell
.\scripts\ask-fable.ps1 `
  -Trigger question `
  -Question "Which of these two evaluation approaches best matches the owner's goal?" `
  -Context "Option A ...; Option B ...; tests/evidence ..." `
  -ProposedNextAction "Use Option A because ..."
```

Evaluate Fable's reasoning rather than following it blindly. Its advice cannot
authorize spending, secrets disclosure, destructive actions, external
publishing, or scope expansion. If the verdict is `need_user`, ask the real
user. If a decision remains unresolved after one focused follow-up, ask the
real user instead of creating an agent-to-agent loop.

Every call starts a fresh tool-free session and injects the fixed role plus
owner profile. Never use `--continue` or `--resume` for these consultations.
Never update `integrations/fable/OWNER_PROFILE.md` based on Fable's output;
change it only when the real user explicitly states or corrects a preference.
