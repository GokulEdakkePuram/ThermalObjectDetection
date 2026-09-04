#!/usr/bin/env bash
# Unattended sweep runner.
#
#   tmux new -s train
#   ./scripts/run_sweep.sh
#
# Runs each config in turn, then the cross-run comparison. Deliberately NOT
# `set -e`: a config that fails must not cancel the ones after it, because the
# whole point of running unattended is to come back to as much finished work as
# possible.
set -uo pipefail

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    # Control first, then the transfer axis, then radiometry. Ordered so that
    # an early stop still leaves a complete ablation rather than half of two.
    CONFIGS=(pretrained scratch frozen_stem frozen_backbone global_map p1p99_map rgb)
fi

TRACK="${TRACK:-wandb}"
LOG_DIR="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

if [ "${TRACK}" = "wandb" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: TRACK=wandb but WANDB_API_KEY is unset." >&2
    echo "         Runs may fail at startup. Export it, or run with TRACK=none." >&2
    sleep 5
fi

echo "=============================================================="
echo " sweep started $(date)"
echo " configs : ${CONFIGS[*]}"
echo " tracking: ${TRACK}"
echo " logs    : ${LOG_DIR}"
echo "=============================================================="

declare -a SUCCEEDED=() FAILED=()

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "---- ${cfg} : started $(date +%H:%M:%S) ----"
    started=${SECONDS}

    # tee, not `>`: a silent terminal for an hour is indistinguishable from a
    # hang, and the first thing anyone does about a hang is kill it. `pipefail`
    # is set above, so the exit status is still the trainer's, not tee's.
    if uv run thermaldet train "${cfg}" --track "${TRACK}" 2>&1 | tee "${LOG_DIR}/${cfg}.log"; then
        SUCCEEDED+=("${cfg}")
        printf -- "---- %s : OK in %dm%02ds ----\n" "${cfg}" $(( (SECONDS-started)/60 )) $(( (SECONDS-started)%60 ))
    else
        FAILED+=("${cfg}")
        echo "---- ${cfg} : FAILED (see ${LOG_DIR}/${cfg}.log) ----"
        tail -15 "${LOG_DIR}/${cfg}.log" | sed 's/^/     /'
    fi
done

# Compare only the runs that actually produced weights. Guarded because
# expanding an empty array is an error under `set -u` in bash < 4.4, and the
# all-runs-failed case is exactly when the summary below matters most.
WEIGHTS=()
if [ ${#SUCCEEDED[@]} -gt 0 ]; then
    for cfg in "${SUCCEEDED[@]}"; do
        w="runs/train/${cfg}/weights/best.pt"
        [ -f "${w}" ] && WEIGHTS+=("${w}")
    done
fi

if [ ${#WEIGHTS[@]} -gt 0 ]; then
    echo ""
    echo "==> Comparing ${#WEIGHTS[@]} checkpoints"
    uv run thermaldet eval "${WEIGHTS[@]}" --split test 2>&1 | tee "${LOG_DIR}/eval.log"
fi

echo ""
echo "=============================================================="
printf " finished %s after %dh%02dm\n" "$(date)" $(( SECONDS/3600 )) $(( (SECONDS%3600)/60 ))
echo " succeeded: $([ ${#SUCCEEDED[@]} -gt 0 ] && echo "${SUCCEEDED[*]}" || echo none)"
echo " failed   : $([ ${#FAILED[@]} -gt 0 ] && echo "${FAILED[*]}" || echo none)"
echo ""
echo " Pull runs/ and reports/ before destroying the instance --"
echo " a stopped instance still bills for disk."
echo "=============================================================="
