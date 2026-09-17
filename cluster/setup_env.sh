#!/usr/bin/env bash
# Create the toolchain and the virtualenv on the cluster. Idempotent: run it
# again after a sync and it only does what is missing.
#
# It runs *on a compute node* (through srun), not on the entry node, because
# the wheels and the Rust toolchain have to match the glibc and the CPU of
# the machine that will run them. The entry node is not in the measurement.
#
# Usage: cluster/setup_env.sh [--force-venv]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

FORCE=""
[ "${1:-}" = "--force-venv" ] && FORCE=1

NODE="${NODES%%,*}"
DEST_ROOT=$(ssh_remote "eval echo $REMOTE_ROOT")

# shellcheck disable=SC2087  # deliberately expanded locally into the heredoc
ssh_remote "srun -p '$PARTITION' -w '$NODE' -n1 -c8 -t 60 bash -s" <<EOF
set -euo pipefail
ROOT="$DEST_ROOT"
export CARGO_HOME="\$ROOT/cargo" RUSTUP_HOME="\$ROOT/rustup"

echo "== rust ($RUST_TOOLCHAIN) on \$(hostname)"
if [ ! -x "\$CARGO_HOME/bin/cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
else
    "\$RUSTUP_HOME/../cargo/bin/rustup" toolchain install "$RUST_TOOLCHAIN" 2>/dev/null || true
fi
export PATH="\$CARGO_HOME/bin:\$PATH"
rustc --version

echo "== venv"
[ -n "$FORCE" ] && rm -rf "\$ROOT/venv"
if [ ! -x "\$ROOT/venv/bin/python" ]; then
    python3 -m venv "\$ROOT/venv"
fi
"\$ROOT/venv/bin/pip" install -q --upgrade pip wheel
"\$ROOT/venv/bin/pip" install -q maturin numpy pytest
# ess is the workload, torann the thing measured: ess is installed editable,
# torann never is -- each arm is an unpacked wheel that the runner puts on
# PYTHONPATH, and an editable torann in the venv would shadow it.
[ -d "\$ROOT/ess" ] && "\$ROOT/venv/bin/pip" install -q -e "\$ROOT/ess"
"\$ROOT/venv/bin/python" - <<'PY'
import numpy, sys
print("python", sys.version.split()[0], "numpy", numpy.__version__)
try:
    import ess; print("ess", ess.__file__)
except ImportError as e:
    print("ess MISSING:", e)
PY
echo "== ok: \$ROOT/venv + \$CARGO_HOME"
EOF
