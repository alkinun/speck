#!/bin/sh

set -eu

usage() {
    cat <<'EOF'
Usage: ./scripts/run_search_v3.sh [options]

Run or resume the version three calibration search and dashboard.

Options:
  --study NAME          Study name (default: speck00-v3-search)
  --experiment PATH     Experiment directory (default: experiments/speck00-200m)
  --config PATH         Search configuration (default: <experiment>/search-v3.json)
  --host HOST           Dashboard host (default: 127.0.0.1)
  --port PORT           Dashboard port (default: 8000)
  --no-dashboard        Do not start the dashboard
  -h, --help            Show this help

Environment overrides:
  SPECK_QUALITY_TOKENS_PER_COST     Default: 10000
  SPECK_EVALUATION_TOKENS_PER_COST  Default: 30000
  SPECK_PROFILE_COST                Default: 600
  SPECK_LEASE_SECONDS               Default: 300
  SPECK_IDLE_SECONDS                Default: 30
EOF
}

STUDY=speck00-v3-search
EXPERIMENT=experiments/speck00-200m
CONFIG=
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8000
DASHBOARD_ENABLED=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --study)
            [ "$#" -ge 2 ] || { printf '%s\n' "missing value for --study" >&2; exit 2; }
            STUDY=$2
            shift 2
            ;;
        --experiment)
            [ "$#" -ge 2 ] || { printf '%s\n' "missing value for --experiment" >&2; exit 2; }
            EXPERIMENT=$2
            shift 2
            ;;
        --config)
            [ "$#" -ge 2 ] || { printf '%s\n' "missing value for --config" >&2; exit 2; }
            CONFIG=$2
            shift 2
            ;;
        --host)
            [ "$#" -ge 2 ] || { printf '%s\n' "missing value for --host" >&2; exit 2; }
            DASHBOARD_HOST=$2
            shift 2
            ;;
        --port)
            [ "$#" -ge 2 ] || { printf '%s\n' "missing value for --port" >&2; exit 2; }
            DASHBOARD_PORT=$2
            shift 2
            ;;
        --no-dashboard)
            DASHBOARD_ENABLED=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

if [ -z "$CONFIG" ]; then
    CONFIG=$EXPERIMENT/search-v3.json
fi

QUALITY_TOKENS_PER_COST=${SPECK_QUALITY_TOKENS_PER_COST:-10000}
EVALUATION_TOKENS_PER_COST=${SPECK_EVALUATION_TOKENS_PER_COST:-30000}
PROFILE_COST=${SPECK_PROFILE_COST:-600}
LEASE_SECONDS=${SPECK_LEASE_SECONDS:-300}
IDLE_SECONDS=${SPECK_IDLE_SECONDS:-30}
OWNER=${SPECK_OWNER:-v3-runner-$$}
UV=${UV:-uv}
PYTHONPATH_VALUE=$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}
DASHBOARD_PID=
ACTIVE_PID=
EMPTY_TICKS=0
TEMP_DIR=${TMPDIR:-/tmp}/speck-v3-runner-$$
umask 077
if ! mkdir "$TEMP_DIR"; then
    printf 'unable to create temporary directory: %s\n' "$TEMP_DIR" >&2
    exit 1
fi
COORDINATOR_FILE=$TEMP_DIR/coordinator.json
STATUS_FILE=$TEMP_DIR/status.json
PYTHON_FILE=$TEMP_DIR/python.txt
INITIALIZATION_FILE=$TEMP_DIR/initialization.json
: > "$COORDINATOR_FILE"
: > "$STATUS_FILE"
: > "$PYTHON_FILE"
: > "$INITIALIZATION_FILE"

stop_process() {
    process=$1
    if [ -n "$process" ] && kill -0 "$process" 2>/dev/null; then
        kill "$process" 2>/dev/null || true
        wait "$process" 2>/dev/null || true
    fi
}

stop_dashboard() {
    stop_process "$DASHBOARD_PID"
}

cleanup() {
    status=$?
    trap - 0 INT TERM
    stop_process "$ACTIVE_PID"
    stop_dashboard
    rm -rf "$TEMP_DIR"
    if [ "$status" -ne 0 ]; then
        printf '\nSearch stopped. Resume with the same command; committed study state is preserved.\n' >&2
    fi
    exit "$status"
}

trap cleanup 0
trap 'exit 130' INT
trap 'exit 143' TERM

run_uv() {
    env PYTHONPATH="$PYTHONPATH_VALUE" "$UV" run --project "$PROJECT_ROOT" --extra gpu "$@" &
    ACTIVE_PID=$!
    wait "$ACTIVE_PID"
    status=$?
    ACTIVE_PID=
    return "$status"
}

printf 'Preparing CUDA environment and study %s...\n' "$STUDY"
run_uv python -m scripts.architecture_search_v3 init "$EXPERIMENT" \
    --study "$STUDY" \
    --config "$CONFIG" > "$INITIALIZATION_FILE"
cat "$INITIALIZATION_FILE"

run_uv python -c 'import sys; print(sys.executable)' > "$PYTHON_FILE"
PYTHON=$(cat "$PYTHON_FILE")
if [ ! -x "$PYTHON" ]; then
    printf 'uv environment Python not found: %s\n' "$PYTHON" >&2
    exit 1
fi

run_python() {
    env PYTHONPATH="$PYTHONPATH_VALUE" PYTHONUNBUFFERED=1 "$PYTHON" "$@"
}

run_managed() {
    run_python "$@" &
    ACTIVE_PID=$!
    wait "$ACTIVE_PID"
    status=$?
    ACTIVE_PID=
    return "$status"
}

MAX_ACTIONS=$(run_python -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["planner"]["max_actions_per_event"])' \
    "$CONFIG")

case "$MAX_ACTIONS" in
    ''|*[!0-9]*)
        printf 'planner.max_actions_per_event must be a positive integer\n' >&2
        exit 1
        ;;
    0)
        printf 'planner.max_actions_per_event must be positive\n' >&2
        exit 1
        ;;
esac

run_python -c '
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
quality_rate = float(sys.argv[2])
profile_cost = float(sys.argv[3])
evaluation_rate = float(sys.argv[4])
evaluation_tokens = int(sys.argv[5])
calibration = config["calibration"]
crossed_runs = (
    calibration["initialization_seeds"]
    * calibration["data_seeds"]
    * calibration["numerical_repeats"]
)
quality_tokens = (
    calibration["broad_architectures"] * calibration["broad_tokens"]
    + calibration["noise_architectures"]
    * (crossed_runs - 1)
    * calibration["noise_tokens"]
    + calibration["anchor_architectures"]
    * (calibration["anchor_tokens"] - calibration["broad_tokens"])
)
objective_names = {item["name"] for item in config["objective_sets"]}
profile_repetitions = calibration["broad_architectures"] * sum(
    item["process_repetitions"]
    for item in config["profiles"]
    if item["name"] in objective_names
)
checkpoints = config["quality"]["checkpoint_tokens"]
noise_checkpoints = sum(tokens <= calibration["noise_tokens"] for tokens in checkpoints)
broad_checkpoints = sum(tokens <= calibration["broad_tokens"] for tokens in checkpoints)
anchor_checkpoints = sum(tokens <= calibration["anchor_tokens"] for tokens in checkpoints)
evaluation_actions = (
    calibration["broad_architectures"] * broad_checkpoints
    + calibration["noise_architectures"] * (crossed_runs - 1) * noise_checkpoints
    + calibration["anchor_architectures"] * (anchor_checkpoints - broad_checkpoints)
)
minimum = (
    quality_tokens / quality_rate
    + evaluation_actions * evaluation_tokens / evaluation_rate
    + profile_repetitions * profile_cost
)
required = minimum * 1.1
budget = config["planner"]["total_cost"]
if budget < required:
    raise SystemExit(
        f"planner budget {budget:.0f} is below the {required:.0f} preflight minimum "
        "for the selected rates"
    )
print(f"Budget preflight: {budget:.0f} available / {required:.0f} required with reserve")
' "$CONFIG" "$QUALITY_TOKENS_PER_COST" "$PROFILE_COST" \
    "$EVALUATION_TOKENS_PER_COST" \
    "$(run_python -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["evaluation_tokens"])' \
        "$INITIALIZATION_FILE")"

if [ "$DASHBOARD_ENABLED" -eq 1 ]; then
    run_python -m scripts.search_dashboard "$STUDY" \
        --host "$DASHBOARD_HOST" \
        --port "$DASHBOARD_PORT" &
    DASHBOARD_PID=$!
    sleep 1
    if ! kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        wait "$DASHBOARD_PID" 2>/dev/null || true
        DASHBOARD_PID=
        printf 'dashboard failed to start on %s:%s\n' "$DASHBOARD_HOST" "$DASHBOARD_PORT" >&2
        exit 1
    fi
    printf 'Dashboard: http://%s:%s\n' "$DASHBOARD_HOST" "$DASHBOARD_PORT"
fi

printf 'Search rates: quality=%s tokens/s, evaluation=%s tokens/s, profile=%s s\n' \
    "$QUALITY_TOKENS_PER_COST" "$EVALUATION_TOKENS_PER_COST" "$PROFILE_COST"
printf 'Press Ctrl-C to stop safely. Run this command again to resume.\n'

while :; do
    printf '\nCoordinating next actions...\n'
    run_managed -m scripts.architecture_search_v3 coordinate "$STUDY" \
        --quality-tokens-per-cost "$QUALITY_TOKENS_PER_COST" \
        --evaluation-tokens-per-cost "$EVALUATION_TOKENS_PER_COST" \
        --profile-cost "$PROFILE_COST" > "$COORDINATOR_FILE"
    COORDINATOR_OUTPUT=$(cat "$COORDINATOR_FILE")
    printf '%s\n' "$COORDINATOR_OUTPUT"

    PHASE=$(run_python -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["phase"])' \
        "$COORDINATOR_FILE")
    if [ "$PHASE" = anchor_complete ]; then
        printf '\nVersion three calibration is complete.\n'
        break
    fi

    run_managed -m scripts.architecture_search_v3 status "$STUDY" > "$STATUS_FILE"
    RUNNING_ACTIONS=$(run_python -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["actions"].get("running", 0))' \
        "$STATUS_FILE")
    PENDING_ACTIONS=$(run_python -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["actions"].get("pending", 0))' \
        "$STATUS_FILE")
    SCHEDULED_ACTIONS=$(run_python -c \
        'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["scheduled_actions"]))' \
        "$COORDINATOR_FILE")

    if [ "$RUNNING_ACTIONS" -gt 0 ]; then
        printf '%s action(s) still hold leases; waiting %s seconds before reclamation checks...\n' \
            "$RUNNING_ACTIONS" "$IDLE_SECONDS"
        sleep "$IDLE_SECONDS"
    fi

    if [ "$RUNNING_ACTIONS" -eq 0 ] && [ "$PENDING_ACTIONS" -eq 0 ] && [ "$SCHEDULED_ACTIONS" -eq 0 ]; then
        EMPTY_TICKS=$((EMPTY_TICKS + 1))
        if [ "$EMPTY_TICKS" -gt 1 ]; then
            printf '%s\n' \
                'search stalled before anchor_complete; inspect planner budget and failed actions' >&2
            exit 1
        fi
        printf '%s\n' 'coordinator completed a phase transition; ticking once more...'
        continue
    fi
    EMPTY_TICKS=0

    run_managed -m scripts.architecture_search_v3 worker "$STUDY" \
        --device cuda \
        --owner "$OWNER-quality" \
        --lease-seconds "$LEASE_SECONDS"

    action=1
    while [ "$action" -le "$MAX_ACTIONS" ]; do
        run_managed -m scripts.architecture_search_v3 evaluation-worker "$STUDY" \
            --device cuda \
            --owner "$OWNER-evaluation" \
            --lease-seconds "$LEASE_SECONDS"
        action=$((action + 1))
    done

    action=1
    while [ "$action" -le "$MAX_ACTIONS" ]; do
        run_managed -m scripts.architecture_search_v3 profile-worker "$STUDY" \
            --backend torch_native \
            --device cuda \
            --owner "$OWNER-profile-gpu" \
            --lease-seconds "$LEASE_SECONDS"
        run_managed -m scripts.architecture_search_v3 profile-worker "$STUDY" \
            --backend torch_native \
            --device cpu \
            --owner "$OWNER-profile-cpu" \
            --lease-seconds "$LEASE_SECONDS"
        action=$((action + 1))
    done
done
