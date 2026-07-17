#!/bin/bash
# Runs the multivariate CP point-forecasting grid MODEL-BY-MODEL, ONE SEED ONLY:
# it finishes every setting + dataset for one model before moving to the next.
# This is the single-seed variant of per-model-mv-forecasting.sh - identical
# traversal, but every run passes --quick so only the first seed (69421) trains.
# Same 60 cells as conformal-prediction-grid.sh (5 models x 3 datasets x
# L{192,336} x O{96,336}), only the traversal order and seed count change:
#
#   DLinear      -> all 12 (dataset, L, O) cells
#   PatchTST     -> all 12 cells
#   TSMixer      -> all 12 cells
#   Transformer  -> all 12 cells
#   EMForecaster -> all 12 cells
#
# Use this when you want a fast, one-model-at-a-time sweep (e.g. to sanity-check
# DLinear end-to-end before the slower models) without paying for the second
# seed, rather than the dataset-major order of conformal-prediction-grid.sh.
#
# CP is enabled (alpha=0.1) on every run; the same run yields both the point
# metrics and the coverage numbers, so there is no separate _pf pass. Each run
# trains only the first seed (69421) - this variant always passes --quick.
#
# Prerequisite: activate the venv before running. No folder swap is needed -
# `python main.py rf_emf_mv/<model>` auto-selects the forecasting.multivariate setup.
#
# Logs -> logs/multivariate/{Model}/{dataset_slug}/L{L}_O{O}_cp/{seed}/log.json
# (identical log layout to conformal-prediction-grid.sh - the scripts share
# results and resume state, since a cell's log dir is keyed by model + window,
# not by traversal order. This single-seed variant only writes/expects seed
# 69421, so a cell it marks done is not the same as the two-seed runners' done.)
#
# --------------------------------------------------------------------------
# Robustness (same as conformal-prediction-grid.sh):
#   1. Survive per-run failures. A single failed cell (e.g. an MPS OOM on
#      EMForecaster, or the known EMF_2023 data issue) is recorded and the
#      remaining cells still run.
#   2. Auto-resume. Cells whose seed already has a fully finished log.json are
#      skipped automatically - just run it again to pick up where it left off.
#   3. End-of-run summary + non-zero exit if any cell failed.
#
# Env knobs (optional):
#   DRYRUN=1   print the python command for each cell instead of running it
#              (still edits the configs, so only use on an idle repo).
#
# NOTE: like conformal-prediction-grid.sh, this rewrites the shared model configs
# as it steps through cells, so only run it when no other grid is currently
# running against this repo. It leaves seed_list at [69421, 69422] and relies on
# --quick, so it does not disturb the two-seed runners' configs.

set -o pipefail
# Deliberately NOT `set -e`: run() handles per-cell failures itself so one bad
# cell does not abort the grid. Also NOT `set -u`: macOS /bin/bash is 3.2, where
# expanding an empty array under nounset ("${PY_FLAGS[@]}") is a fatal error.
cd "$(dirname "$0")/../../.." || { echo "cannot cd to repo root" >&2; exit 1; }

# --- knobs ------------------------------------------------------------------
DRYRUN="${DRYRUN:-0}"

# This runner is inherently single-seed: always --quick, always expect seed one.
PY_FLAGS=(--quick)
EXPECTED_SEEDS=(69421)

if [[ ! -d forecasting/multivariate/jobs/exp/rf_emf_mv ]]; then
    echo "No rf_emf_mv job configs at forecasting/multivariate/jobs/exp/rf_emf_mv." >&2
    echo "Is the forecasting/multivariate setup present at the repo root?" >&2
    exit 1
fi

CONFIGS=forecasting/multivariate/jobs/exp/rf_emf_mv
CFGS=(
    "$CONFIGS/dlinear/args.yaml"
    "$CONFIGS/tsmixer/args.yaml"
    "$CONFIGS/patchtst/args.yaml"
    "$CONFIGS/transformer/args.yaml"
    "$CONFIGS/emforecaster/args.yaml"
)

# One-time safety edit: these are already the checked-in values, re-asserted so
# a stale working tree cannot silently change the protocol. The indented `conf:`
# is the on/off flag (the leading space keeps the column-0 `conf:` section
# header untouched). seed_list stays two-seeded; --quick selects seed one.
sed -i '' -E \
    -e 's/^([[:space:]]*)neptune: .*/\1neptune: False/' \
    -e 's/^([[:space:]]+)conf: .*/\1conf: True/' \
    -e 's/^([[:space:]]*)alpha: .*/\1alpha: 0.1/' \
    -e 's/^([[:space:]]*)seed_list:.*/\1seed_list: [69421, 69422]/' \
    -e 's/^([[:space:]]*)id: .*/\1id: ""/' \
    "${CFGS[@]}" || { echo "one-time config edit failed" >&2; exit 1; }

# --- bookkeeping ------------------------------------------------------------
CUR_DATASET=""; CUR_L=""; CUR_O=""   # set by set_window; consumed by cell_done/run
PASSED=()
SKIPPED=()
FAILED=()

# set_window <dataset> <L> <O>
# Point all model configs at one (dataset, L, O) row and remember it so the
# resume check can rebuild the derived run dir. exp.id is no longer used.
set_window() {
    local dataset=$1 L=$2 O=$3
    CUR_DATASET="$dataset"; CUR_L="$L"; CUR_O="$O"
    sed -i '' -E \
        -e "s/^([[:space:]]*)dataset: .*/\1dataset: \"${dataset}\"/" \
        -e "s/^([[:space:]]*)seq_len: .*/\1seq_len: ${L}/" \
        -e "s/^([[:space:]]*)pred_len: .*/\1pred_len: ${O}/" \
        "${CFGS[@]}" || { echo "config edit failed for ${dataset} L${L} O${O}" >&2; exit 1; }
}

# run_dir_for <model_id> -> new-tree run dir for the current window (cp on).
run_dir_for() {
    python3 -c "from forecasting.run_path import build_run_dir; print(build_run_dir('multivariate', '$1', '$CUR_DATASET', $CUR_L, $CUR_O, True))"
}

# cell_done <model_id>
# True (exit 0) when every expected seed for the current window has a fully
# finished log.json. log.json is created at training start, so its existence is
# not enough; we require the last-written conformal metric (sl_test/
# interval_width_mean). Every rf_emf_mv job runs with conf.conf + intervals on,
# so that key is present iff training + test + conformal recalibration completed.
cell_done() {
    local model_id=$1
    DONE_DIR="$(run_dir_for "$model_id")" python3 - "${EXPECTED_SEEDS[@]}" <<'PY'
import json, os, sys
d = os.environ["DONE_DIR"]
marker = "sl_test/interval_width_mean"
for seed in sys.argv[1:]:
    p = os.path.join(d, seed, "log.json")
    try:
        with open(p) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        sys.exit(1)
    if marker not in data:
        sys.exit(1)
sys.exit(0)
PY
}

# run <label> <model_id> <job>
# Resume-aware, failure-tolerant single cell. Assumes set_window ran first.
run() {
    local label=$1 model_id=$2 job=$3
    if cell_done "$model_id"; then
        echo "[SKIP] ${label}  ($(run_dir_for "$model_id") already complete)"
        SKIPPED+=("$label")
        return 0
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "[DRY ] ${label}  -> python main.py ${job} ${PY_FLAGS[*]}"
        PASSED+=("$label")
        return 0
    fi
    echo "[RUN ] ${label}"
    if python main.py "$job" "${PY_FLAGS[@]}"; then
        echo "[ OK ] ${label}"
        PASSED+=("$label")
    else
        local code=$?
        echo "[FAIL] ${label}  (exit ${code}) - continuing with the rest of the grid" >&2
        FAILED+=("$label")
    fi
}

# =====================================================================
# Grid definition. The traversal below is model-major: the outer loop is the
# model, so every (dataset, L, O) cell for one model finishes before the next
# model starts. Datasets and windows keep the same inner order as
# conformal-prediction-grid.sh.
# =====================================================================

# Model order: DLinear, PatchTST, TSMixer, Transformer, EMForecaster.
# Each entry: "<display label>|<model_id>|<job>". The Transformer's on-disk
# model_id is RF_EMF_Transformer, hence the separate display label.
MODELS=(
    "DLinear|DLinear|rf_emf_mv/dlinear"
    "PatchTST|PatchTST|rf_emf_mv/patchtst"
    "TSMixer|TSMixer|rf_emf_mv/tsmixer"
    "Transformer|RF_EMF_Transformer|rf_emf_mv/transformer"
    "EMForecaster|EMForecaster|rf_emf_mv/emforecaster"
)

# Each entry: "<dscode>|<dataset>|<display label>".
# EMF_2023 (University 2023, 29 channels) has a known data issue and runs as-is
# per the 2026-07-02 decision: 1835 MHz spike + ~13% synthetic (interpolated)
# test gap; footnote it in the results table.
DATASETS=(
    "ed2022|ED2022|ED2022"
    "uh2022|UH2022|UH2022"
    "emf2023|EMF_2023_university_5_with_PTV|EMF_2023"
)

# Each entry: "<seq_len> <pred_len>".
WINDOWS=(
    "192 96"
    "192 336"
    "336 96"
    "336 336"
)

n=0
for m in "${MODELS[@]}"; do
    IFS='|' read -r mlabel model_id job <<< "$m"
    echo ""
    echo "====================================================================="
    echo "${mlabel} - all settings and datasets (12 cells)"
    echo "====================================================================="
    for d in "${DATASETS[@]}"; do
        IFS='|' read -r dscode dataset dslabel <<< "$d"
        for w in "${WINDOWS[@]}"; do
            read -r L O <<< "$w"
            n=$((n + 1))
            set_window "$dataset" "$L" "$O"
            label=$(printf '[%2d/60] %-12s %s/L%s/O%s' "$n" "$mlabel" "$dslabel" "$L" "$O")
            run "$label" "$model_id" "$job"
        done
    done
    echo ">>> ${mlabel} complete (12 cells)."
done

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "============ multivariate CP grid summary (1 seed, per model) ============"
echo "  passed:  ${#PASSED[@]}"
echo "  skipped: ${#SKIPPED[@]}  (already complete)"
echo "  failed:  ${#FAILED[@]}"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "  failed cells:"
    for c in "${FAILED[@]}"; do
        echo "    - $c"
    done
    echo "Re-run this script to retry only the failed/incomplete cells."
    exit 1
fi
echo ">>> All 60 multivariate CP cells complete."
