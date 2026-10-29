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
# Campaigns (default: all):
#   aa          A/A noise calibration on `NODES` (main vs main_aa, W1-W5 W7 W8)
#   w8-cutoff   S0/W8 8-bit prefilter cutoff study (s0_off vs s0_1, s0_off vs s0_17)
#   s5-soa      S5 transposed float32 SoA tile filter (main vs s5_soa, W1-W5 W7 W8)
#   n1-full     Upstream origin/main vs local main (origin_main vs main, W1-W5 W7 WQ)
#   experiments Standalone 4-task array (recall-ablation, crossover, refine-rounds, exploration)
#   all         Queue every campaign above sharing a single arm-build job
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
        aa|w8-cutoff|s5-soa|n1-full|experiments|all)
            CAMPAIGNS+=("$1"); shift ;;
        *)
            echo "error: unknown argument '$1' (see --help)" >&2
            exit 2 ;;
    esac
done

[[ ${#CAMPAIGNS[@]} -eq 0 ]] && CAMPAIGNS=(all)
for c in "${CAMPAIGNS[@]}"; do
    if [[ "$c" == "all" ]]; then
        CAMPAIGNS=(aa w8-cutoff s5-soa n1-full experiments)
        break
    fi
done

run_submit() {
    local arm_a="$1" arm_b="$2" tag="$3" suites="$4" after_id="$5"
    local cmd=("$DIR/submit.sh" "$arm_a" "$arm_b"
        --name "$tag"
        --suites "$suites"
        --pairs "$PAIRS_ARG"
        --no-sync)
    [[ -n "$after_id" ]] && cmd+=(--after "$after_id")
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

for c in "${CAMPAIGNS[@]}"; do
    case "$c" in
        aa)
            run_submit main main_aa "aa_main" "W1 W2 W3 W4 W5 W7 W8" "$BUILD_ID"
            ;;
        w8-cutoff)
            run_submit s0_off s0_1  "w8_off_vs_1"  "W8" "$BUILD_ID"
            run_submit s0_off s0_17 "w8_off_vs_17" "W8" "$BUILD_ID"
            run_submit s0_17  s0_17b "w8_aa_17"    "W8" "$BUILD_ID"
            ;;
        s5-soa)
            run_submit main s5_soa "s5_soa" "W1 W2 W3 W4 W5 W7 W8" "$BUILD_ID"
            ;;
        n1-full)
            run_submit origin_main main "n1_origin_vs_main" "W1 W2 W3 W4 W5 W7 WQ" "$BUILD_ID"
            ;;
        experiments)
            DEP_EXP="${BUILD_ID:+--dependency=afterok:$BUILD_ID}"
            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[dry-run] ssh $REMOTE sbatch $DEP_EXP -p $PARTITION -w ${NODES%%,*} --array=0-4 experiments.sbatch"
            else
                EXP_ID=$(ssh_remote "cd '$ROOT/logs' && sbatch --parsable $DEP_EXP -p '$PARTITION' \
                    -w '${NODES%%,*}' --array=0-4 \
                    --export=ALL,BENCH_ROOT='$ROOT',ARM=main \
                    '$ROOT/torann/cluster/experiments.sbatch'")
                echo "== queued standalone experiments array $EXP_ID (recall-ablation, crossover, refine-rounds, exploration, lp-recall)"
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
  cluster/fetch.sh aa_main
  cluster/fetch.sh w8_off_vs_1 --aa aa_main
  cluster/fetch.sh w8_off_vs_17 --aa aa_main
  cluster/fetch.sh s5_soa --aa aa_main
  cluster/fetch.sh n1_origin_vs_main --aa aa_main
EOF
