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

    # One-command fire-and-forget launcher for all active campaigns
    # (builds arms once, then queues A/A, W8 cutoff, S5 SoA, N1 full, and
    # standalone ablation/crossover jobs so you can disconnect immediately):
    cluster/campaigns.sh --dry-run
    cluster/campaigns.sh all

    # Or launch individual campaigns:
    cluster/campaigns.sh aa w8-cutoff s5-soa
    cluster/campaigns.sh experiments --no-sync --no-build

    # Manual single A/B comparison:
    cluster/submit.sh main s5_soa --spec cluster/arms.spec \
        --suites "W1 W2 W3 W4 W5 W7 W8" --pairs 8 --name s5_soa

    # Later (once jobs finish):
    cluster/fetch.sh aa_main
    cluster/fetch.sh s5_soa --aa examples/out/cluster/ab_aa_main.json

`submit.sh` chains three jobs: build (optional, `--spec` or `--after JOB_ID`), a
sweep array with one task per shard, and a merge that runs `afterok` and
**refuses any case whose two arms did not meet on one machine**. Results are
fetched into `examples/out/cluster/`, deliberately separate from the laptop's
own runs.

`experiments.sbatch` runs standalone non-A/B workloads (`recall-ablation`,
`crossover`, `refine-rounds`, `exploration`) on a compute node using `/tmp`
tmpfs staging and copies outputs back to `examples/out/exp_<task>_<arm>/`.

Useful options: `--cases REGEX`, `--shards N`, `--after JOB_ID`, `--no-sync`, `--wait`.

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
