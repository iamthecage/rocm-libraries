# Conventions

- Preserve project-local style and avoid unrelated formatting/refactors.
- Do not delete or revert user changes without explicit permission.
- Keep tuning provenance: generated configs, result JSONL ledgers, merge scripts, and final merged logic YAML are distinct recovery artifacts.
- For custom gfx1201 rocBLAS kernels, source-of-truth logic belongs under rocBLAS `asm_full/gfx1201`; benchmark outputs are external, larger reproducibility artifacts.