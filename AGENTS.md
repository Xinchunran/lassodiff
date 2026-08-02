# Repository Agent Instructions

1. Before every task in this repository, read `README.md` completely before mutations, training, downloads, or process control.
2. For OpenDDE V3, also read `reasoning.md`, `metrics.md`, and `scoring.md` completely, then the active files under `lasso_instruction/`.
3. Treat every `MUST` as fail-closed. Do not claim completion from configuration, logging, module construction, or checkpoint keys alone.
4. Do not start V3 training until its required tests and architecture preflight pass. Never silently fall back from OpenDDE reasoning.
5. Preserve active runs and user data unless the user explicitly authorizes stopping or deletion.
