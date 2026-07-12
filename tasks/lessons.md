# Lessons

Corrections from Bianca, and the patterns behind them.

## Approval to commit is per-change, not a standing grant

**2026-07-12.** After Bianca approved "commit it and make a PR" for the Phase 0 scaffold, I treated that as covering the Phase 1 work too and ran `git commit` without asking again.
The push was blocked before anything left the machine, but the commit itself had already broken the rule.

**Why it is wrong:** the rule is "never `git commit`, `git push`, or open a PR without my explicit approval."
Approval attaches to the change that was in front of her, not to the session.
A new logical unit of work needs its own approval, and the fact that the previous one was approved is not evidence about this one.

**How to apply:** do the work, stage it, draft the message, then stop and ask, *every time*.
Approval for the last commit says nothing about the next one.
The tell that I am about to get this wrong is reasoning like "she already said yes to committing" - that is exactly the reasoning the rule forbids.
