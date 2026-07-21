# Suggested Commands

- Inspect a narrow working slice: `git status --short -- <path>`.
- Tensile runner scripts live in `shared/tensile/next-tuning`; use their `--dry-run` and `--status` modes before expensive runs.
- Use `python3 shared/tensile/next-tuning/merge_gfx1201_logic.py --status` to inspect benchmark-output merge coverage.
- Use CMake presets from repository root for project builds; individual project docs provide target-specific commands.