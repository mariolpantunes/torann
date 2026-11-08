#!/usr/bin/env bash
# Launch one or all Slurm benchmark campaigns on `playstation` and exit.
#
# Designed so you can connect to the IT network / VPN, queue the entire
# optimization validation suite in seconds, and close the laptop while
# Slurm runs build -> sweep arrays -> merge -> standalone ablations.
#
# Usage:
#   cluster/campaigns.sh [CAMPAIGN ...] [options]
#
# Campaigns (default: all). Each chain link isolates one core idea:
#   smoke       2 pairs of a few cases per link: proves every arm builds and runs
#   aa          A/A noise calibration on `NODES` (s3fix vs s3fix_aa, every suite)
#   core1       Batched radius join          (c0 vs c1)
#   core2       u8 prefilter + auto-off      (c1 vs c2)
#   core34      recall knob + L^p LSH        (c2 vs c3)
#   core5       Crossover thresholds, S3     (c3 vs s3)
#   s3-fix      S3 fix, needs the fix on HEAD (s3 vs s3fix)
#   retune      Drift/promote retuning       (s3fix vs retune)
#   s7          S7 2-wall multi-probe        (retune vs s7)
#   c0-s7       Cumulative baseline vs S7    (c0 vs s7)
#   s7-aa       S7 A/A noise calibration     (s7 vs s7_aa)
#   w8-cutoff   S0 prefilter cutoff study    (s0_off vs s0_1, s0_off vs s0_17, A/A)
#   s5-soa      S5 SoA tiles on their own base (c2 vs s5_soa)
#   experiments 7-task array on arm s7 (see experiments.sbatch)
#   all         Everything above but smoke, sharing one arm-build job
#
# Options:
#   --dry-run   Print the remote commands and sbatch submissions without SSH
#   --no-sync   Skip rsync to playstation
#   --no-build  Skip build_arms.sbatch (assume arms in cluster/arms.spec are built)
#   --pairs N   Override paired ABBA repetitions per case (default: 8)
#   -h, --help  Show this help and exit
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster/config.sh
source "$DIR/config.sh"

usage() {
    sed -n '2,25s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
}

DRY_RUN=0
DO_SYNC=1
DO_BUILD=1
PAIRS_ARG="${PAIRS:-8}"
CAMPAIGNS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-sync) DO_SYNC=0; shift ;;
        --no-build) DO_BUILD=0; shift ;;
        --pairs) PAIRS_ARG="${2:?--pairs needs an even integer}"; shift 2 ;;
        smoke|aa|core1|core2|core34|core5|s3-fix|retune|s7|c0-s7|s7-aa|w8-cutoff|s5-soa|experiments|all)
            CAMPAIGNS+=("$1"); shift ;;
        *)
            echo "error: unknown argument '$1' (see --help)" >&2
            exit 2 ;;
    esac
done

[[ ${#CAMPAIGNS[@]} -eq 0 ]] && CAMPAIGNS=(all)
for c in "${CAMPAIGNS[@]}"; do
    if [[ "$c" == "all" ]]; then
        CAMPAIGNS=(aa core1 core2 core34 core5 s3-fix retune s7 c0-s7 s7-aa w8-cutoff s5-soa experiments)
        break
    fi
done

run_submit() {
    local arm_a="$1" arm_b="$2" tag="$3" suites="$4" after_id="$5" pairs="${6:-$PAIRS_ARG}" cases="${7:-}"
    local cmd=("$DIR/submit.sh" "$arm_a" "$arm_b"
        --name "$tag"
        --suites "$suites"
        --pairs "$pairs"
        --no-sync)
    [[ -n "$after_id" ]] && cmd+=(--after "$after_id")
    [[ -n "$cases" ]] && cmd+=(--cases "$cases")
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] ${cmd[*]}"
    else
        echo "== queueing campaign $tag ($arm_a vs $arm_b on $suites)"
        "${cmd[@]}"
    fi
}

if [[ "$DO_SYNC" -eq 1 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] $DIR/sync.sh"
    else
        "$DIR/sync.sh"
    fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    ROOT="\$HOME/bench"
else
    ROOT=$(ssh_remote "eval echo $REMOTE_ROOT")
fi

BUILD_ID=""
if [[ "$DO_BUILD" -eq 1 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] scp $DIR/arms.spec $REMOTE:$ROOT/arms.spec"
        echo "[dry-run] ssh $REMOTE sbatch -p $PARTITION -w ${NODES%%,*} -c 8 -t $TIME_BUILD build_arms.sbatch"
        BUILD_ID="10000"
    else
        ssh_remote "mkdir -p '$ROOT/logs'"
        scp -q "$DIR/arms.spec" "$REMOTE:$ROOT/arms.spec"
        BUILD_ID=$(ssh_remote "cd '$ROOT/logs' && sbatch --parsable -p '$PARTITION' -w '${NODES%%,*}' \
            -c 8 -t '$TIME_BUILD' --export=ALL,BENCH_ROOT='$ROOT',ARMS_SPEC='$ROOT/arms.spec' \
            '$ROOT/torann/cluster/build_arms.sbatch'")
        echo "== queued shared arm build job $BUILD_ID ($DIR/arms.spec)"
    fi
fi

SUITES_BASE="W1 W2 W3 W4 W5 W7"

for c in "${CAMPAIGNS[@]}"; do
    case "$c" in
        smoke)
            local_cases='/d8-|/d8/|/d32-|n4096-d8'
            for link in "c0 c1" "c1 c2" "c2 c3" "c3 s3" "s3 s3fix" "c2 s5_soa"; do
                read -r arm_a arm_b <<<"$link"
                run_submit "$arm_a" "$arm_b" "smoke_${arm_a}_${arm_b}" "$SUITES_BASE" "$BUILD_ID" 2 "$local_cases"
            done
            ;;
        aa)
            run_submit s3fix s3fix_aa "aa" "$SUITES_BASE W8 W9" "$BUILD_ID"
            ;;
        core1)
            run_submit c0 c1 "core1" "$SUITES_BASE" "$BUILD_ID"
            ;;
        core2)
            run_submit c1 c2 "core2" "$SUITES_BASE W8" "$BUILD_ID"
            ;;
        core34)
            run_submit c2 c3 "core34" "$SUITES_BASE W8" "$BUILD_ID"
            ;;
        core5)
            run_submit c3 s3 "core5" "$SUITES_BASE W8 W9" "$BUILD_ID"
            ;;
        s3-fix)
            run_submit s3 s3fix "s3fix" "$SUITES_BASE W8 W9" "$BUILD_ID"
            ;;
        retune)
            run_submit s3fix retune "retune" "$SUITES_BASE W8 W9" "$BUILD_ID"
            ;;
        s7)
            run_submit retune s7 "s7" "$SUITES_BASE W8 W9" "$BUILD_ID"
            ;;
        c0-s7)
            run_submit c0 s7 "c0_s7" "$SUITES_BASE W8 W9" "$BUILD_ID"
            ;;
        s7-aa)
            run_submit s7 s7_aa "s7_aa" "$SUITES_BASE W8 W9" "$BUILD_ID"
            ;;
        w8-cutoff)
            run_submit s0_off s0_1  "w8_off_vs_1"  "W8" "$BUILD_ID"
            run_submit s0_off s0_17 "w8_off_vs_17" "W8" "$BUILD_ID"
            run_submit s0_17  s0_17b "w8_aa_17"    "W8" "$BUILD_ID"
            ;;
        s5-soa)
            run_submit c2 s5_soa "s5_soa" "$SUITES_BASE W8" "$BUILD_ID"
            ;;
        experiments)
            # Named like the sweeps and singleton: these share the node, so
            # they must wait for every sweep or both timings are void.
            DEP_EXP="--dependency=${BUILD_ID:+afterok:$BUILD_ID,}singleton --job-name=torann-sweep"
            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[dry-run] ssh $REMOTE sbatch $DEP_EXP -p $PARTITION -w ${NODES%%,*} --array=0-6%1 experiments.sbatch"
            else
                EXP_ID=$(ssh_remote "cd '$ROOT/logs' && sbatch --parsable $DEP_EXP -p '$PARTITION' \
                    -w '${NODES%%,*}' --array=0-6%1 \
                    --export=ALL,BENCH_ROOT='$ROOT',ARM='${EXP_ARM:-s7}' \
                    '$ROOT/torann/cluster/experiments.sbatch'")
                echo "== queued standalone experiments array $EXP_ID (recall-ablation, crossover, refine-rounds, exploration, lp-recall, tune-model, tune-cost)"
            fi
            ;;
    esac
done

cat <<EOF

All requested campaigns queued. You can now disconnect or end the session.
Monitor when on VPN/IT network:
  ~/.agents/skills/slurm/scripts/status.sh
  ssh $REMOTE "ls -lh $ROOT/torann/examples/out/"
Fetch A/B reports once complete:
  cluster/fetch.sh aa
  for t in core1 core2 core34 core5 s3fix retune s7 s7_aa w8_off_vs_1 w8_off_vs_17 s5_soa; do
      cluster/fetch.sh \$t --aa examples/out/cluster/ab_aa.json; done
EOF
