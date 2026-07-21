# Tech Stack

- Linux ROCm C++/HIP monorepo, coordinated with CMake presets/toolchains.
- Python and shell drive Tensile generation, tuning, merging, and library creation.
- Tensile includes Python tooling; rocBLAS consumes pregenerated logic YAML under its Tensile logic tree.
- Local environment state (ROCm compiler/runtime) is documented outside the repo in VS Code system-baseline instructions.