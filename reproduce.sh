#!/usr/bin/env bash
#
# Reproduce the experiments of thesis Chapter 7 (and Appendix A).
#
# Stages, in order (every stage can be rerun; the sweeps resume):
#   sweep     560 runs (70 instances x 8 configurations), 900 s each, into
#             results/thesis-eval-900.jsonl; every returned equilibrium is
#             checked by the certifier as it is found
#   certify   re-check every equilibrium of the log with `rcmas certify`
#             -> results/thesis-eval-900.cert.jsonl
#   encoding  time the construction of the formula F per instance
#             -> results/encoding-build.jsonl
#   repeat    the same sweep with a limit of 300 s (for the variation
#             between runs, Section 7.2) -> results/thesis-eval-300.jsonl,
#             and its certificates
#   assets    Table 7.1, Figures 7.2-7.3 and Table A.1
#             -> results/thesis-assets/{chapters,appendices}/experiment/*.tex
#   numbers   a report with the new value of every \rerun{...} of the thesis
#             -> results/thesis-numbers.md (and .json)
#
# Usage:
#   ./reproduce.sh [--workers N] [--no-repeat] [--thesis-dir DIR]
#                  [--only STAGE[,STAGE...]] [--results-dir DIR] [--archive]
#   ./reproduce.sh --smoke          # 4 small instances, 30 s limit, minutes
#
# Options:
#   --workers N       runs in parallel (default: physical cores minus one;
#                     on Apple silicon, performance cores minus one)
#   --no-repeat       skip the 300 s repeat sweep
#   --thesis-dir DIR  also write the assets into the thesis checkout DIR and
#                     scan DIR for \rerun markers (default scan: ../../thesis,
#                     read-only)
#   --only STAGES     run only these stages (comma-separated)
#   --results-dir DIR where outputs go (default: results, or results/smoke,
#                     in the tool directory); a relative DIR of either option
#                     is taken relative to the current directory
#   --archive         after the stages, pack the committed tool and the
#                     committed results into dist/ for upload to Zenodo
#   --smoke           the smoke test: experiments/smoke.toml, 30 s limit,
#                     repeat with 10 s, outputs in results/smoke/
#
set -euo pipefail

CALLER_DIR="$PWD"
TOOL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TOOL_DIR"

WORKERS=""
SMOKE=0
REPEAT=1
THESIS_DIR=""
ONLY=""
RESULTS_DIR=""
ARCHIVE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --workers=*) WORKERS="${1#*=}"; shift ;;
        --smoke) SMOKE=1; shift ;;
        --no-repeat) REPEAT=0; shift ;;
        --thesis-dir) THESIS_DIR="$2"; shift 2 ;;
        --thesis-dir=*) THESIS_DIR="${1#*=}"; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --only=*) ONLY="${1#*=}"; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --results-dir=*) RESULTS_DIR="${1#*=}"; shift ;;
        --archive) ARCHIVE=1; shift ;;
        -h|--help) sed -n '2,44p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see ./reproduce.sh --help)" >&2; exit 2 ;;
    esac
done

# Relative --thesis-dir and --results-dir values are relative to the directory
# from which the script was called, not to the tool directory.
from_caller() {
    case "$1" in
        ""|/*) printf '%s' "$1" ;;
        *) printf '%s/%s' "$CALLER_DIR" "$1" ;;
    esac
}
THESIS_DIR=$(from_caller "$THESIS_DIR")
RESULTS_DIR=$(from_caller "$RESULTS_DIR")
if [ -n "$THESIS_DIR" ] && [ ! -f "$THESIS_DIR/chapters/experiment/main.tex" ]; then
    echo "error: --thesis-dir $THESIS_DIR is not a thesis checkout (no chapters/experiment/main.tex there)" >&2
    exit 2
fi

if [ "$SMOKE" = "1" ]; then
    CONFIG="experiments/smoke.toml"
    SUITE="smoke"
    REPEAT_LIMIT=10
    MARK_TIME=10
    RESULTS_DIR="${RESULTS_DIR:-results/smoke}"
    if [ -n "$THESIS_DIR" ]; then
        echo "error: --smoke never writes into a thesis checkout; drop --thesis-dir" >&2
        exit 2
    fi
else
    CONFIG="experiments/thesis-eval.toml"
    SUITE="thesis-eval"
    REPEAT_LIMIT=300
    MARK_TIME=300
    RESULTS_DIR="${RESULTS_DIR:-results}"
fi

run_stage() {
    # run_stage NAME: true if NAME is selected by --only (or no --only given)
    [ -z "$ONLY" ] && return 0
    case ",$ONLY," in *",$1,"*) return 0 ;; esac
    return 1
}

say() { printf '\n== %s\n' "$*"; }

# ── Signals ─────────────────────────────────────────────────────────
# Each long stage runs in the background while this script waits for it, so
# that a signal reaches the script at once (bash runs a trap only after a
# foreground command has finished).  On Ctrl-C, `kill <pid of this script>`
# or a closed terminal, the running stage gets SIGTERM; the sweep then kills
# its runs and writes the end of its session to the log before it exits.
STAGE_PID=""
stop_stage() {
    # stop_stage SIGNAL EXIT_CODE
    trap '' INT TERM HUP
    if [ -n "$STAGE_PID" ]; then
        kill -TERM "$STAGE_PID" 2>/dev/null || true
        wait "$STAGE_PID" 2>/dev/null || true
    fi
    echo "reproduce.sh: stopped by SIG$1; run the same command again to resume." >&2
    exit "$2"
}
trap 'stop_stage INT 130' INT
trap 'stop_stage TERM 143' TERM
trap 'stop_stage HUP 129' HUP

run_py() {
    # run_py ARGS...: "$PY" ARGS in the background; returns its exit code
    "$PY" "$@" &
    STAGE_PID=$!
    local rc=0
    wait "$STAGE_PID" || rc=$?
    STAGE_PID=""
    return $rc
}

# ── Environment ─────────────────────────────────────────────────────
PY="$TOOL_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    say "No .venv/ yet: running ./setup.sh --no-tests"
    ./setup.sh --no-tests
fi
Z3_PIN=$(sed -n 's/.*"z3-solver==\([^"]*\)".*/\1/p' pyproject.toml)
Z3_FOUND=$("$PY" -c 'import importlib.metadata as m; print(m.version("z3-solver"))')
if [ "$Z3_FOUND" != "$Z3_PIN" ]; then
    echo "error: .venv has z3-solver $Z3_FOUND but pyproject.toml pins $Z3_PIN; rerun ./setup.sh" >&2
    exit 1
fi
export PYTHONHASHSEED=0
export PYTHONPATH="$TOOL_DIR${PYTHONPATH:+:$PYTHONPATH}"

# The limit of the main sweep comes from the [protocol] table of the config.
LIMIT=$("$PY" -c 'import sys; from experiments.config import load_protocol; print(load_protocol(sys.argv[1]).time_limit_s)' "$CONFIG")
LOG="$RESULTS_DIR/$SUITE-$LIMIT.jsonl"
CERTS="$RESULTS_DIR/$SUITE-$LIMIT.cert.jsonl"
REPEAT_LOG="$RESULTS_DIR/$SUITE-$REPEAT_LIMIT.jsonl"
REPEAT_CERTS="$RESULTS_DIR/$SUITE-$REPEAT_LIMIT.cert.jsonl"
ENCODING="$RESULTS_DIR/encoding-build.jsonl"
ASSETS="$RESULTS_DIR/thesis-assets"
NUMBERS="$RESULTS_DIR/thesis-numbers.md"
NUMBERS_JSON="$RESULTS_DIR/thesis-numbers.json"
mkdir -p "$RESULTS_DIR"

if [ -z "$WORKERS" ]; then
    WORKERS=$("$PY" -c 'from experiments.provenance import default_workers; print(default_workers())')
fi

SCAN_DIR="$THESIS_DIR"
if [ -z "$SCAN_DIR" ] && [ -f "$TOOL_DIR/../../thesis/chapters/experiment/main.tex" ]; then
    SCAN_DIR="$(cd "$TOOL_DIR/../../thesis" && pwd)"
fi

# Only a git checkout of the tool itself counts (not an enclosing repository).
COMMIT="unknown"
TOP=$(git -C "$TOOL_DIR" rev-parse --show-toplevel 2>/dev/null || true)
if [ -n "$TOP" ] && [ "$(cd "$TOP" && pwd -P)" = "$(cd "$TOOL_DIR" && pwd -P)" ]; then
    COMMIT=$(git -C "$TOOL_DIR" rev-parse --short HEAD)
    if [ -n "$(git -C "$TOOL_DIR" status --porcelain --untracked-files=no)" ]; then
        echo "WARNING: the tool has uncommitted changes; the logs will record tool_dirty=true." >&2
    fi
elif [ -f "$TOOL_DIR/.git-commit" ]; then
    COMMIT=$(cut -c1-7 "$TOOL_DIR/.git-commit")
fi
say "rcmas at $COMMIT, $("$PY" -V), z3-solver $Z3_FOUND, $WORKERS workers, config $CONFIG"

STATUS=0

# ── Stages ──────────────────────────────────────────────────────────
if run_stage sweep; then
    say "sweep: $CONFIG, limit $LIMIT s -> $LOG"
    run_py -m experiments.run --config "$CONFIG" \
        --output "$LOG" --workers "$WORKERS" --resume
fi

certify_log() {
    # certify_log LOG CERTS
    say "certify: $1 -> $2"
    set +e
    run_py -c 'import sys; from rcmas.certify import main; sys.exit(main(sys.argv[1:]))' \
        --results "$1" --output "$2"
    local rc=$?
    set -e
    if [ $rc -eq 1 ]; then
        echo "WARNING: the certifier REJECTED an equilibrium of $1 (see above)." >&2
        STATUS=1
    elif [ $rc -eq 2 ]; then
        echo "WARNING: some equilibria of $1 could not be checked (see above)." >&2
        STATUS=1
    elif [ $rc -ne 0 ]; then
        echo "error: rcmas certify failed with exit code $rc" >&2
        exit $rc
    fi
}

if run_stage certify; then
    if [ -f "$LOG" ]; then certify_log "$LOG" "$CERTS"; else echo "no $LOG; run the sweep first" >&2; exit 1; fi
fi

if run_stage encoding; then
    say "encoding: time to build F -> $ENCODING"
    run_py -m experiments.encoding_times --config "$CONFIG" --output "$ENCODING"
fi

if [ "$REPEAT" = "1" ] && run_stage repeat; then
    say "repeat: $CONFIG, limit $REPEAT_LIMIT s -> $REPEAT_LOG"
    run_py -m experiments.run --config "$CONFIG" --timeout "$REPEAT_LIMIT" \
        --output "$REPEAT_LOG" --workers "$WORKERS" --resume
    certify_log "$REPEAT_LOG" "$REPEAT_CERTS"
fi

if run_stage assets; then
    say "assets -> $ASSETS"
    CERT_ARG=()
    [ -f "$CERTS" ] && CERT_ARG=(--certificates "$CERTS")
    THESIS_ARG=()
    [ -n "$THESIS_DIR" ] && THESIS_ARG=(--thesis-dir "$THESIS_DIR")
    run_py -m experiments.thesis_assets --results "$LOG" ${CERT_ARG[@]+"${CERT_ARG[@]}"} \
        --out-dir "$ASSETS" --mark-time "$MARK_TIME" ${THESIS_ARG[@]+"${THESIS_ARG[@]}"}
fi

if run_stage numbers; then
    say "numbers -> $NUMBERS"
    ARGS=(--results "$LOG" --output "$NUMBERS" --json "$NUMBERS_JSON")
    [ -f "$CERTS" ] && ARGS+=(--certificates "$CERTS")
    [ -f "$REPEAT_LOG" ] && ARGS+=(--repeat "$REPEAT_LOG")
    [ -f "$REPEAT_CERTS" ] && ARGS+=(--repeat-certificates "$REPEAT_CERTS")
    [ -f "$ENCODING" ] && ARGS+=(--encoding "$ENCODING")
    [ -n "$SCAN_DIR" ] && ARGS+=(--thesis-dir "$SCAN_DIR")
    run_py -m experiments.thesis_numbers "${ARGS[@]}"
fi

if [ "$ARCHIVE" = "1" ]; then
    VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
    NAME="rcmas-$VERSION-$COMMIT"
    say "archive -> dist/$NAME.tar.gz"
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "WARNING: uncommitted changes are NOT in the archive (it holds HEAD $COMMIT)." >&2
    fi
    STAGE_DIR=$(mktemp -d)
    mkdir -p "$STAGE_DIR/$NAME" dist
    git archive --format=tar HEAD | tar -x -C "$STAGE_DIR/$NAME"
    git rev-parse HEAD > "$STAGE_DIR/$NAME/.git-commit"
    # Only committed files go into the archive: the results that README step 1
    # commits are part of HEAD, and the smoke outputs and other untracked files
    # under results/ stay out.
    for f in "$LOG" "$CERTS" "$REPEAT_LOG" "$REPEAT_CERTS" "$ENCODING" "$NUMBERS"; do
        [ -f "$f" ] || continue
        if ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 \
            || ! git diff --quiet HEAD -- "$f" 2>/dev/null; then
            echo "WARNING: $f has uncommitted changes, and the archive holds only what HEAD has of it; commit it first (README, Archiving on Zenodo)." >&2
        fi
    done
    tar -czf "dist/$NAME.tar.gz" -C "$STAGE_DIR" "$NAME"
    rm -rf "$STAGE_DIR"
    echo "wrote dist/$NAME.tar.gz"
fi

say "outputs in $RESULTS_DIR/"
for f in "$LOG" "$CERTS" "$REPEAT_LOG" "$REPEAT_CERTS" "$ENCODING" "$NUMBERS"; do
    [ -f "$f" ] && echo "  $f"
done
[ -d "$ASSETS" ] && echo "  $ASSETS/"
exit $STATUS
