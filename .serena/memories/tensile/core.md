# Tensile Gfx1201 Tuning

- Tuning configs/runners/merge tooling are in `shared/tensile/next-tuning`.
- Benchmark outputs are intentionally external: `/home/iamthecage/out` and `/home/iamthecage/out-v2`; per-config `3_LibraryLogic/*.yaml` enables re-merging without rerunning benchmarks.
- `merge_gfx1201_logic.py` currently reads `/home/iamthecage/out-v2` and targets `asm_full/gfx1201` on `--install`.
- Recovery priorities: final merged YAML in rocBLAS source; runner/config/merge scripts in git; external raw benchmark output backup; installed `/opt/rocm` gfx1201 library backup.
- Existing merge snapshots are under `shared/tensile/next-tuning/gfx1201-merged-logic*`.