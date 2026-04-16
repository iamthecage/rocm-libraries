#!/usr/bin/env bash
# =============================================================================
# run_gfx1201_validate.sh — gfx1201 (Navi48) code-path validation sweep
#
# Runs every *.yaml config in gfx1201-tall/ sequentially.
# Each config exercises a specific dimension of gfx12 code generation:
#   - GSU algorithms (SingleBuffer / MultipleBuffer workspace stores)
#   - GSU AtomicAdd (buffer_atomic_add_f32)
#   - NonTemporal cache modifiers (sc0/sc1/nt flags)
#   - Buffer vs Global load/store paths
#   - Scheduling algorithms (SIA 0–3, PGR, PLR)
#   - StreamK (workspace stores, atomic accumulation)
#   - Data types (all WMMA V2 + VALU combos)
#   - Store paths (VectorWidth, StoreRemap, VectorStore)
#
# All configs use NumElementsToValidate=-1 (full correctness) with tiny
# problem sizes for fast compile+validate. Assembly failures are highlighted.
#
# Usage:
#   ./run_gfx1201_validate.sh                    # run all validate_* configs
#   ./run_gfx1201_validate.sh --rerun-failed      # only re-run FAILED configs
#   ./run_gfx1201_validate.sh --filter "gsu"      # only configs matching "gsu"
#   ./run_gfx1201_validate.sh --dry-run            # list what would run
#
# Results:
#   $RESULTS_JSONL  — JSONL file, one record per run
#   gfx1201-tall/*.log  — per-config Tensile stdout/stderr logs
#
# =============================================================================
set -uo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/gfx1201-tall"
RESULTS_JSONL="${SCRIPT_DIR}/gfx1201_validate_results.jsonl"

# Tensile client — override with TENSILE_CLIENT env var if needed
TENSILE_CLIENT="${TENSILE_CLIENT:-${SCRIPT_DIR}/../next-cmake/build/tensile-client}"

# Arch for this run
ARCH="gfx1201"

# Output base dir for compiled+benchmarked kernels
OUTPUT_BASE="${OUTPUT_BASE:-/home/iamthecage/out_validate}"

# ── Argument parsing ─────────────────────────────────────────────────────────
RERUN_FAILED=0
FILTER=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rerun-failed)  RERUN_FAILED=1 ; shift ;;
        --filter)        FILTER="$2"   ; shift 2 ;;
        --dry-run)       DRY_RUN=1     ; shift ;;
        -h|--help)
            sed -n '2,30p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
log_result() {
    # log_result NAME STATUS ELAPSED_S YAML OUTPUT_DIR ERROR_MSG
    local name="$1" status="$2" elapsed="$3" yaml="$4" outdir="$5" err="$6"
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '{"name": "%s", "status": "%s", "elapsed_s": %d, "arch": "%s", "yaml": "%s", "output_dir": "%s", "error": %s, "ts": "%s"}\n' \
        "$name" "$status" "$elapsed" "$ARCH" "$yaml" "$outdir" \
        "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$err")" \
        "$ts" \
        >> "$RESULTS_JSONL"
}

already_passed() {
    local name="$1"
    [[ -f "$RESULTS_JSONL" ]] || return 1
    python3 -c "
import json, sys
name = sys.argv[1]
with open('${RESULTS_JSONL}') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('name') == name and r.get('status') == 'PASSED':
                sys.exit(0)
        except Exception:
            pass
sys.exit(1)
" "$name" 2>/dev/null
}

previously_failed() {
    local name="$1"
    [[ -f "$RESULTS_JSONL" ]] || return 1
    python3 -c "
import json, sys
name = sys.argv[1]
last = None
with open('${RESULTS_JSONL}') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('name') == name:
                last = r.get('status')
        except Exception:
            pass
sys.exit(0 if last == 'FAILED' else 1)
" "$name" 2>/dev/null
}

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "${TENSILE_CLIENT}" ]]; then
    echo "[ERROR] tensile-client not found at: ${TENSILE_CLIENT}"
    echo "        Set TENSILE_CLIENT=/path/to/tensile-client to override."
    exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "[ERROR] Config dir not found: ${CONFIG_DIR}"
    exit 1
fi

# ── Build file list (all *.yaml files in the config dir) ─────────────────────
mapfile -t ALL_YAMLS < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.yaml' | sort)

YAMLS_TO_RUN=()
for yaml in "${ALL_YAMLS[@]}"; do
    name=$(basename "$yaml" .yaml)

    # Apply --filter if set
    if [[ -n "$FILTER" ]] && [[ "$name" != *"${FILTER}"* ]]; then
        continue
    fi

    # --rerun-failed: skip configs that already passed
    if [[ $RERUN_FAILED -eq 1 ]]; then
        if already_passed "$name"; then
            echo "[SKIP] $name — already PASSED"
            continue
        fi
        if ! previously_failed "$name"; then
            echo "[SKIP] $name — no prior FAILED record"
            continue
        fi
    fi

    YAMLS_TO_RUN+=("$yaml")
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  gfx1201 code-path VALIDATION sweep"
echo "  arch          : ${ARCH}"
echo "  configs total : ${#YAMLS_TO_RUN[@]}"
echo "  client        : ${TENSILE_CLIENT}"
echo "  output base   : ${OUTPUT_BASE}"
echo "  results file  : ${RESULTS_JSONL}"
[[ -n "$FILTER" ]]        && echo "  filter        : ${FILTER}"
[[ $RERUN_FAILED -eq 1 ]] && echo "  mode          : --rerun-failed"
[[ $DRY_RUN -eq 1 ]]      && echo "  mode          : --dry-run (no execution)"
echo "════════════════════════════════════════════════════════"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would run:"
    for yaml in "${YAMLS_TO_RUN[@]}"; do
        echo "  $(basename "$yaml")"
    done
    exit 0
fi

if [[ ${#YAMLS_TO_RUN[@]} -eq 0 ]]; then
    echo "[INFO] Nothing to run."
    exit 0
fi

mkdir -p "${OUTPUT_BASE}"

# ── Main validation loop ─────────────────────────────────────────────────────
PASSED=0
FAILED=0

for yaml in "${YAMLS_TO_RUN[@]}"; do
    name=$(basename "$yaml" .yaml)
    output_dir="${OUTPUT_BASE}/${name}"
    log_file="${CONFIG_DIR}/${name}.log"

    echo "────────────────────────────────────────────────────────"
    echo "[$(date '+%H:%M:%S')] START: ${name}"
    echo "  yaml   : ${yaml}"
    echo "  output : ${output_dir}"
    echo "  log    : ${log_file}"
    echo "────────────────────────────────────────────────────────"

    start_ts=$(date +%s)
    error_msg=""
    exit_code=0

    if (
        set -e
        cd "${SCRIPT_DIR}"
        Tensile -v \
            "${yaml}" \
            "${output_dir}" \
            --prebuilt-client "${TENSILE_CLIENT}" \
            --arch="${ARCH}" \
            2>&1 | tee "${log_file}"
    ); then
        exit_code=0
        status="PASSED"
        ((PASSED++))
    else
        exit_code=$?
        status="FAILED"
        error_msg="Tensile exited with code ${exit_code}"
        ((FAILED++))
        echo ""
        echo "[WARN] ${name} FAILED (exit ${exit_code})"
    fi

    # ── Check for assembly errors in the log ─────────────────────────────
    if [[ -f "${log_file}" ]]; then
        asm_errors=$(grep -c 'error: instruction not supported\|error: invalid operand\|error: expected' "${log_file}" 2>/dev/null || true)
        if [[ "$asm_errors" -gt 0 ]]; then
            echo ""
            echo "  ╔══════════════════════════════════════════════════╗"
            echo "  ║  ASSEMBLY ERRORS DETECTED: ${asm_errors} occurrence(s)     ║"
            echo "  ╚══════════════════════════════════════════════════╝"
            echo "  Relevant lines:"
            grep -n 'error: instruction not supported\|error: invalid operand\|error: expected' "${log_file}" | head -10 | sed 's/^/    /'
            echo ""
            # Override status if we saw asm errors even if Tensile exited 0
            if [[ "$status" == "PASSED" ]]; then
                status="ASM_ERRORS"
                error_msg="Assembly errors found (${asm_errors} occurrences) despite exit 0"
                ((PASSED--))
                ((FAILED++))
            fi
        fi
    fi

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    log_result "$name" "$status" "$elapsed" "$yaml" "$output_dir" "$error_msg"

    echo "[$(date '+%H:%M:%S')] ${status}: ${name}  (${elapsed}s)"
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  VALIDATION SUMMARY"
echo "  PASSED : ${PASSED}"
echo "  FAILED : ${FAILED}"
echo "  Total  : ${#YAMLS_TO_RUN[@]}"
echo "  Results: ${RESULTS_JSONL}"
echo "════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed configs:"
    python3 -c "
import json
with open('${RESULTS_JSONL}') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('status') in ('FAILED', 'ASM_ERRORS'):
                print(f\"  {r['name']}: {r['error']}\")
        except Exception:
            pass
" 2>/dev/null || true
    exit 1
fi
