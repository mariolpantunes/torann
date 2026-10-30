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
# practice, with MEM_PER_NODE: the default of 4 GB per CPU times a node's worth
# of CPUs exceeds what the node has, and Slurm then rejects the job as
# "Requested node configuration is not available". Never `--mem=0` on a shared
# node: it needs all of the memory, so a neighbour's job blocks it for as long
# as it runs (xbox, 2026-10-30: estimated start in 2027).
# Check with `sinfo -N -o "%N %C %O %t"` before a long run.
#
# As of 2026-10-30 atari, case-hpc and tars-hpc hold another user's 180-day
# jobs. `xbox` has the same CPU as atari (EPYC 9354, 32 x 2) and 60 CPUs free.
# 50 CPUs keeps the shared-cluster limit and is 25 physical cores, hence 25
# threads. Atari A/A files do not calibrate it: run the `aa` campaign first.
PARTITION="${PARTITION:-cpuPartition}"
NODES="${NODES:-xbox}"
CPUS_PER_NODE="${CPUS_PER_NODE:-50}"          # xbox (atari: 62, sega: 22)
MEM_PER_NODE="${MEM_PER_NODE:-75G}"           # 1500M per CPU
TIME_SWEEP="${TIME_SWEEP:-12:00:00}"
TIME_BUILD="${TIME_BUILD:-2:00:00}"

# The toolchain the arms are built with. Pinned so a rebuilt arm is the same
# compiler as the one it is compared against; matches pyproject.toml / CI.
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.99.1}"

# Threads per run. Default: the node's *physical* cores, not its threads --
# two hyperthreads of one core share the execution units this kernel is
# limited by, and the laptop baseline was taken with one thread per core.
# Empty means "every CPU Slurm gave the job".
RAYON_THREADS="${RAYON_THREADS:-25}"          # xbox at 50 CPUs (atari: 32, sega: 12)

ssh_remote() { ssh -o BatchMode=yes "$REMOTE" "$@"; }
