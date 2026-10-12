# Model release audits

One completed audit form per model family, named `<family>.md` where
`<family>` is the `family` value from
`tooling/publish-model/models-core.toml`. Start every form by copying
[TEMPLATE.md](TEMPLATE.md).

## Why

The form exists so every model release ships in its best known state --
maximum performance, complete backend coverage, verified correctness -- and so
every consciously skipped optimization is on the record with a justification
and an unlock condition instead of silently forgotten. It is a means to peak
release quality, not paperwork: filling one should take an afternoon.

## The rules

- **When to fill:** before a new model family enters the release flow, i.e.
  before its first catalog entry flips `public:true`. Effective 2026-07, this
  is mandatory for every new family.
- **Enforcement:** the publish pipeline fails closed.
  `tooling/publish-model/scripts/_manifest.py --public` (also reached via
  `regenerate_all.sh --public`) refuses to write a public catalog entry when
  `docs/model-audits/<family>.md` is missing, still contains `<!-- TODO:fill -->`
  markers, or lost any of the ten numbered sections. See
  `tooling/publish-model/scripts/audit_form.py`.
- **Ships with the release:** the completed form is part of the release; keep
  it updated when a dimension's status changes (e.g. a Deferred item lands).
- **Existing families:** `PRE_AUDIT_FAMILIES` is retained only as a migration
  ledger for source-tree reporting. The form is required whenever a family's
  public set would grow (its first public entry), and a missing form fails
  closed there. Re-emitting entries of a family that is ALREADY public in the
  committed catalog (metadata or download-URL changes) is never blocked by the
  gate -- an unfinished form must not hold unrelated catalog work hostage to
  the backfill. The gate instead validates the form when one exists and logs
  the result as owed backfill (form missing, or the exact validation failure),
  so the debt stays visible on every re-emission. For every public
  family/provider lane that the architecture
  inventory marks Auto-eligible or explicitly selectable, the structured backend
  table must contain current positive Supported, Golden-verified, and utilization
  answers (CPU may use `N/A` for utilization). `Untested`, `Deferred`, `No`,
  missing, or stale cells never approve a lane. An explicit `No` remains useful
  only when the inventory does not advertise that provider, and must include the
  non-activation reason and unlock condition.

## Filling guidance

Each item is `Supported` / `Not applicable` / `Deferred`. Should-support items
MUST be `Supported`; `Not applicable` and `Deferred` both require a detailed
justification, and `Deferred` additionally requires the unlock condition that
flips it. Cite evidence (test names, bench runs, gate scripts) rather than
asserting -- the form is only as useful as its audit trail.
