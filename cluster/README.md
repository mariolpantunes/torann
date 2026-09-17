# Running the A/B benchmarks on Slurm

`examples/bench_ab.py` measures one machine. These scripts move that onto the
`playstation` cluster and run it there, without weakening the thing that makes
its numbers mean anything: **both arms of a case are always measured by one
process, on one node, interleaved ABBA.** The cluster buys parallelism across
*cases* and jobs that survive a laptop reboot, never parallelism across arms.

## Layout

Everything lives under `$HOME/bench` on the cluster (`/slurm_shared`, so every
node sees the same copy):

    bench/
      torann/  ess/     the synced repositories (`.git` included: arms are
                        built from revisions via `git worktree`)
      venv/             python 3.14 + maturin, numpy, pytest, ess (editable)
      cargo/ rustup/    the pinned Rust toolchain (config.sh: RUST_TOOLCHAIN)
      arms/             built arms + the shared cargo target
      logs/             Slurm job output
      torann/examples/out/ab_<tag>.shard*.json and the merged ab_<tag>.json

## One-time setup

    cluster/sync.sh          # push both repos
    cluster/setup_env.sh     # rustup + venv, on a compute node, idempotent

## A run

    # build arms and sweep them, sharded across the nodes in config.sh
    cluster/submit.sh base p1e --spec cluster/arms.spec \
        --suites "W1 W2 W3 W4 W5" --pairs 7 --name p1-final

    # later
    cluster/fetch.sh p1-final --aa examples/out/cluster/ab_aa.json

`submit.sh` chains three jobs: build (optional, `--spec`), a sweep array with
one task per shard, and a merge that runs `afterok` and **refuses any case
whose two arms did not meet on one machine**. Results are fetched into
`examples/out/cluster/`, deliberately separate from the laptop's own runs.

Useful options: `--cases REGEX`, `--shards N`, `--no-sync`, `--wait`.

## What has to hold for the numbers to mean anything

* **Calibrate per node type.** The verdict rule needs an A/A run of a build
  against a copy of itself *on the same CPU*; a report without one prints
  `(uncal.)`. Run `cluster/submit.sh base base2 --name aa ...` once per node
  model, and pass that file to `fetch.sh --aa`.
* **Never compare across machines.** A laptop number and a cluster number are
  different CPUs; the alignment finding of 2026-09-17 was a 10% swing from
  code moving inside one binary, so a different micro-architecture is not a
  detail. `merge` enforces this within a run; between runs it is on you.
* **Only idle nodes.** As of 2026-09-17 `case-hpc`, `tars-hpc` and `xbox`
  carry another user's long jobs at load 3000-6200 with 8-16 CPUs free.
  Timing there measures the neighbours. Check first:

      ssh playstation 'sinfo -N -o "%N %C %O %t"'

* **`--exclusive` does not work here** (nodes advertise 127 of 128 CPUs), so
  jobs ask for `CPUS_PER_NODE` cores and `--mem=0` instead. On an idle node
  that is equivalent; on a busy one it only queues.
* **Threads are pinned** to physical cores (`RAYON_THREADS`, default 32 for
  `atari`), because two hyperthreads of a core share the units this kernel is
  limited by.
* **aarch64 (`bilbo-hpc`) is a separate experiment**, never an arm in an x86
  comparison: no AVX2, and `.cargo/config.toml`'s flags are x86-only.
* **Don't sync during a build.** `sync.sh` excludes `.git/worktrees/` for
  exactly this reason, but a `--delete` sync of a file a running job reads is
  still a bad idea.

## Nodes (2026-09-17)

| node | CPU | cores | state |
|---|---|---|---|
| atari | EPYC 9354 (Zen 4) | 32 x 2 threads | idle — the default |
| sega | Threadripper 2920X | 12 x 2 | idle, older core, `CPUS_PER_NODE=22 RAYON_THREADS=12` |
| case-hpc, tars-hpc | EPYC 9534 (Zen 4) | 128, no SMT | busy (another user) |
| xbox | EPYC 9354 | 32 x 2 | busy |
| bilbo-hpc | TaiShan-v110 | 120 | idle, aarch64 |
