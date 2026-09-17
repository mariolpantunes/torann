#!/usr/bin/env bash
# Push torann (and ess, which the ESS workloads import) to the cluster.
#
# `.git` goes too: `bench_ab.py build --rev REV` checks a revision out into a
# temporary worktree, so the arms can only be built there if the history is
# there -- but `.git/worktrees/` is excluded from the delete, because a sync
# during a build would otherwise unregister the checkout that build is using
# (it did, once: "fatal: '/tmp/arm-base-...' is not a working tree").
# What is deliberately *not* synced is everything a build produces --
# `target/`, `examples/out/` -- because those are per-machine: an arm built
# on this laptop is a different CPU's binary and must never be measured on a
# cluster node.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

DEST_ROOT=$(ssh_remote "eval echo $REMOTE_ROOT")
ssh_remote "mkdir -p '$DEST_ROOT'"

sync_one() {
    local src="$1" name="$2"
    [ -d "$src" ] || { echo "skip $name: $src not found"; return; }
    echo "== $name -> $REMOTE:$DEST_ROOT/$name"
    rsync -az --delete --info=stats1 \
        --exclude 'target/' --exclude 'examples/out/' --exclude 'exploration/out/' \
        --exclude '.git/worktrees/' \
        --exclude '.venv/' --exclude '__pycache__/' --exclude '*.so' \
        --exclude 'dist/' --exclude 'build/' --exclude '*.egg-info/' \
        "$src/" "$REMOTE:$DEST_ROOT/$name/"
}

sync_one "$LOCAL_TORANN" torann
sync_one "$LOCAL_ESS" ess

# The results directory is the one thing that must survive a sync.
ssh_remote "mkdir -p '$DEST_ROOT/torann/examples/out' '$DEST_ROOT/arms' '$DEST_ROOT/logs'"
echo "synced to $REMOTE:$DEST_ROOT"
