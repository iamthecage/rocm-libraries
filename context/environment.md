# Agent Environment Notes

- Use terminal runs, not MCP Python execution, for tests because live output matters and MCP runs can hide or delay stderr/stdout.
- Use the master uv environment: `source ~/master/bin/activate`. Do not use `~/aiter/list/bin/activate`.
- Python 3.12 via uv for all Tensile work.
- ROCm 7.2.1 on Ubuntu 24.04 (Noble).
- PyTorch 2.8.0+rocm7.2.1 from AMD manylinux repo.
- **CRITICAL**: Do NOT run GPU kernel execution (tensile-client, benchmarks) without explicit user approval. The kernel may hang.
