#!/usr/bin/env bash
# =============================================================================
# run_gfx1201_wide.sh — gfx1201 (Navi48) wide tuning run
#
# Runs every YAML in gfx1201-wide/ sequentially through Tensile.
# Each config is isolated: a failure is logged to the JSONL results file
# and the script continues with the next config.
#
# Usage:
#   ./run_gfx1201_wide.sh                       # run all 52 configs
#   ./run_gfx1201_wide.sh --rerun-failed         # only re-run configs marked FAILED
#   ./run_gfx1201_wide.sh --filter "wmma"        # only run configs whose name contains "wmma"
#   ./run_gfx1201_wide.sh --dry-run              # list what would run, don't execute
#
# Results:
#   $RESULTS_JSONL  — JSONL file, one record per run (append-safe, timestamped)
#   $WIDE_DIR/*.log — per-config Tensile stdout/stderr logs
#
# =============================================================================
set -uo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIDE_DIR="${SCRIPT_DIR}/gfx1201-wide-v2"
RESULTS_JSONL="${SCRIPT_DIR}/gfx1201_wide_results.jsonl"

# Tensile client — override with TENSILE_CLIENT env var if needed
TENSILE_CLIENT="${TENSILE_CLIENT:-${SCRIPT_DIR}/../next-cmake/build/tensile-client}"

# Arch for this run
ARCH="gfx1201"

# Output base dir for compiled+benchmarked kernels
OUTPUT_BASE="${OUTPUT_BASE:-/home/iamthecage/out-v2}"

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
    # Use printf to safely escape the error message for JSON
    printf '{"name": "%s", "status": "%s", "elapsed_s": %d, "arch": "%s", "yaml": "%s", "output_dir": "%s", "error": %s, "ts": "%s"}\n' \
        "$name" "$status" "$elapsed" "$ARCH" "$yaml" "$outdir" \
        "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$err")" \
        "$ts" \
        >> "$RESULTS_JSONL"
}

already_passed() {
    # Returns 0 (true) if the name has a PASSED entry in the results file
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
    # Returns 0 (true) if the name's most recent entry in the results is FAILED
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

if [[ ! -d "${WIDE_DIR}" ]]; then
    echo "[ERROR] Wide config dir not found: ${WIDE_DIR}"
    exit 1
fi

# Generate YAML files if fewer than 10 exist (first run)
yaml_count=$(find "${WIDE_DIR}" -maxdepth 1 -name '*.yaml' | wc -l)
if [[ ${yaml_count} -lt 10 ]]; then
    echo "[INFO] Only ${yaml_count} YAML(s) found — running generator..."
    python3 "${WIDE_DIR}/generate_configs.py"
fi

# ── Build file list ───────────────────────────────────────────────────────────
mapfile -t ALL_YAMLS < <(find "${WIDE_DIR}" -maxdepth 1 -name '*.yaml' | sort)

YAMLS_TO_RUN=()
for yaml in "${ALL_YAMLS[@]}"; do
    name=$(basename "$yaml" .yaml)

    # Apply --filter if set
    if [[ -n "$FILTER" ]] && [[ "$name" != *"${FILTER}"* ]]; then
        continue
    fi

    # --rerun-failed: skip configs that already passed, only run previously failed
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
echo "  gfx1201 wide tuning run"
echo "  arch          : ${ARCH}"
echo "  configs total : ${#YAMLS_TO_RUN[@]}"
echo "  client        : ${TENSILE_CLIENT}"
echo "  output base   : ${OUTPUT_BASE}"
echo "  results file  : ${RESULTS_JSONL}"
[[ -n "$FILTER" ]]   && echo "  filter        : ${FILTER}"
[[ $RERUN_FAILED -eq 1 ]] && echo "  mode          : --rerun-failed"
[[ $DRY_RUN -eq 1 ]] && echo "  mode          : --dry-run (no execution)"
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

# ── Main benchmark loop ───────────────────────────────────────────────────────
PASSED=0
FAILED=0

for yaml in "${YAMLS_TO_RUN[@]}"; do
    name=$(basename "$yaml" .yaml)
    output_dir="${OUTPUT_BASE}/${name}"
    log_file="${WIDE_DIR}/${name}.log"

    echo "────────────────────────────────────────────────────────"
    echo "[$(date '+%H:%M:%S')] START: ${name}"
    echo "  yaml   : ${yaml}"
    echo "  output : ${output_dir}"
    echo "  log    : ${log_file}"
    echo "────────────────────────────────────────────────────────"

    start_ts=$(date +%s)
    error_msg=""
    exit_code=0

    # Run Tensile in a subshell so a non-zero exit is caught by `if`, not
    # `set -e`, which would abort the entire script.
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
        echo "[WARN] ${name} FAILED (exit ${exit_code}) — captured in results, continuing..."
    fi

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    log_result "${name}" "${status}" "${elapsed}" "${yaml}" "${output_dir}" "${error_msg}"

    echo "[${status}] ${name} finished in ${elapsed}s"
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  WIDE RUN COMPLETE"
echo "  PASSED : ${PASSED}"
echo "  FAILED : ${FAILED}"
echo "  TOTAL  : $((PASSED + FAILED))"
echo "  Results: ${RESULTS_JSONL}"
echo "════════════════════════════════════════════════════════"

# Print list of failed configs for easy re-running
if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "FAILED CONFIGS (re-run with --rerun-failed or copy these lines):"
    python3 - <<'PYEOF'
import json, os, sys

results_file = os.environ.get('RESULTS_JSONL', '')
if not results_file:
    results_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.',
        'gfx1201_wide_results.jsonl'
    )

# Track most-recent status per config name
latest = {}
try:
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                latest[r['name']] = r
            except Exception:
                pass
except FileNotFoundError:
    pass

for name, r in sorted(latest.items()):
    if r.get('status') == 'FAILED':
        print(f"  {name}")
        print(f"    yaml: {r.get('yaml', '?')}")
        print(f"    err : {r.get('error', '?')}")
        print()
PYEOF
fi

# Non-zero exit if any configs failed (useful for CI)
exit $FAILED
