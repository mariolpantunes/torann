# Shared settings for the Slurm benchmark scripts. Sourced, never run.
#
# Every value can be overridden from the environment, so a one-off run needs
# no edit here: `NODES=xbox,atari cluster/submit.sh ...`.

# Entry node (an ssh alias in ~/.ssh/config) and where the repos live there.
# Home is on /slurm_shared, so one copy is visible from every compute node.
REMOTE="${REMOTE:-playstation}"
REMOTE_ROOT="${REMOTE_ROOT:-\$HOME/bench}"     # expanded remotely, not here
LOCAL_TORANN="${LOCAL_TORANN:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOCAL_ESS="${LOCAL_ESS:-$(cd "$LOCAL_TORANN/../ess" 2>/dev/null && pwd)}"

# Where the measuring runs.
#
# Node choice is a measurement decision, not a scheduling one. As of
# 2026-09-17 case-hpc, tars-hpc and xbox carry another user's long jobs at
# load 3000-6200 with 8-16 CPUs free: timing there measures the neighbours.
# `atari` (EPYC 9354, Zen 4, 32 cores x 2 threads) and `sega` (Threadripper
# 2920X, 12 x 2) are idle, and `bilbo-hpc` is aarch64 (no AVX: a separate
# experiment, never an arm in an x86 comparison).
#
# Slurm here refuses `--exclusive` -- the nodes advertise 127 of 128 CPUs as
# usable, so an exclusive request can never be satisfied -- so a job asks for
# CPUS_PER_NODE cores instead, which on an idle node is the same thing in
# practice, with `--mem=0` (the whole node's memory): the default of 4 GB per
# CPU times a node's worth of CPUs exceeds what the node has, and Slurm then
# rejects the job as "Requested node configuration is not available".
# Check with `sinfo -N -o "%N %C %O %t"` before a long run.
PARTITION="${PARTITION:-cpuPartition}"
NODES="${NODES:-atari}"
CPUS_PER_NODE="${CPUS_PER_NODE:-62}"          # atari: CPUEfctv (sega: 22)
TIME_SWEEP="${TIME_SWEEP:-12:00:00}"
TIME_BUILD="${TIME_BUILD:-2:00:00}"

# The toolchain the arms are built with. Pinned so a rebuilt arm is the same
# compiler as the one it is compared against; the kernel needs 1.98+.
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.98.0}"

# Threads per run. Default: the node's *physical* cores, not its threads --
# two hyperthreads of one core share the execution units this kernel is
# limited by, and the laptop baseline was taken with one thread per core.
# Empty means "every CPU Slurm gave the job".
RAYON_THREADS="${RAYON_THREADS:-32}"          # atari: 32 cores (sega: 12)

ssh_remote() { ssh -o BatchMode=yes "$REMOTE" "$@"; }
