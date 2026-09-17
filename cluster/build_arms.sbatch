#!/usr/bin/env bash
#SBATCH --job-name=torann-build
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out
# Build every arm listed in $ARMS_SPEC into the shared arms directory.
#
# One job, one node, all arms: they share a cargo target directory, so
# building them together is both faster and the only way to be sure they came
# out of the same toolchain. The arms live on /slurm_shared, so the sweep
# array tasks -- possibly on other nodes of the same CPU model -- read the
# very binaries this job produced, and each worker checks the `.so` hash.
#
# $ARMS_SPEC is a file, one arm per line, `#` comments allowed:
#     name  rev            [extra bench_ab build flags]
#     base  b06c7df
#     p1e   372d8e0
#     p1e_al 372d8e0 --rustflags "-C target-feature=+avx2,+fma -C llvm-args=..."
set -euo pipefail

ROOT="${BENCH_ROOT:?BENCH_ROOT not set}"
SPEC="${ARMS_SPEC:?ARMS_SPEC not set}"

export CARGO_HOME="$ROOT/cargo" RUSTUP_HOME="$ROOT/rustup"
export PATH="$CARGO_HOME/bin:$ROOT/venv/bin:$PATH"
export TORANN_ARMS="$ROOT/arms"
export CARGO_TARGET_DIR="$ROOT/arms/_target"

cd "$ROOT/torann"
echo "node $(hostname) | $(rustc --version) | $(python -V)"

while read -r name rev flags; do
    case "$name" in ''|\#*) continue;; esac
    echo "== arm $name ($rev) $flags"
    eval python examples/bench_ab.py build "$name" --rev "$rev" $flags
done < "$SPEC"

echo "== arms in $TORANN_ARMS"
for a in "$TORANN_ARMS"/*/arm.json; do
    python - "$a" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  {m['name']:10s} {m['head'][:9]} dirty={m['dirty']} "
      f"features={m['features']} rustflags={m['rustflags']}")
PY
done
