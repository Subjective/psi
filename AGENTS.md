## Commits and PRs

- Keep commits relatively atomic. If a change grows into several independent pieces, ask whether I want to split it into multiple PRs.
- Use Conventional Commit-style prefixes such as `feat:` and `fix:` in commit messages.
- Always open new PRs in draft mode.
- After creating, updating, or pushing changes to a PR, include a direct link to each affected PR in your final response.
- PR descriptions should be standalone artifacts. A reviewer should not need to read our chat, local notes, or hidden context to understand the change.
- PR descriptions should start with a top-level summary that explains the state before the change, what is changing, and why. Link associated materials such as linear issues, Notion design docs, specs, or follow-up PRs when they exist.
- When opening or updating a PR stack, every PR description in the stack should include a `Stack` section. Use a numbered list with links to each PR in order, add a short description of what each PR does, and mark the current PR in bold. Each PR title should include a shared category and its position in the stack, e.g., `[Project] [x/n] Change description`.
- For broad refactors that touch many call sites, include a table listing the important call sites changed and what changed at each one. This is especially important for linked migrations, API shape changes, and function behavior changes.
- For large PRs or PRs where we made important design decisions during the work, include a design decisions section, state the decisions directly, and explain why we chose them.
- In PR descriptions, keep testing notes categorical. Prefer brief entries like `Tests:, targeted tests, smoke tests covering [x, y, z], and CI`. Do not include test failures, blocked test attempts, raw command transcripts, full local command lists, generated file commands, formatter commands, or environment-specific invocation details in the testing section, unless I explicitly ask for them.
- Before pushing commits to a remote or updating an existing PR, run the repose formatter or fix command if one exists, and it is relevant to the files changed.
- When I ask you to carry a PR through Codex review, you may post add `@codex review` comments on the PRs and scope after each pushed head SHA without asking for separate confirmation. Keep this exception narrow: it only covers requesting Codex bot review, not replying to humans or making substantive GitHub comments on my behalf.
- Before resolving a thread authored by the `@codex review` bot, reply in that thread with the reason for resolving it. State whether the feedback was addressed and how, is not relevant and why, or is intentionally deferred or out of scope for the PR and why. Then resolve the thread. This permission applies only to `@codex review` threads. It does not authorize replies to human reviewers.

## Documentation

- `docs/` is be the authoritative source of truth for planning and design. If our implementation intentionally drifts, make sure to update the existing docs accordingly.
