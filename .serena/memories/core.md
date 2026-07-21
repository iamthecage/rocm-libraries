# Core

- ROCm monorepo; top-level CMake coordinates independently buildable libraries under `projects/` plus shared tooling under `shared/`.
- Custom gfx1201 Tensile tuning work lives in `shared/tensile/next-tuning`.
- For the tuning/recovery pipeline, read `mem:tensile/core`.
- Build and validation conventions are covered by `mem:tech_stack`, `mem:suggested_commands`, and `mem:task_completion`.