# Arms built by `build_arms.sbatch`.
# Columns: name  rev  [KEY=VAL ...] [extra `bench_ab.py build` flags]
# Revisions must exist in the synced history.

# Upstream & current main baselines (plus A/A noise calibration arm):
origin_main  37cbcec
main         0d566d1
main_aa      0d566d1

# S0 / W8 8-bit prefilter dimension-cutoff study (branch `cutoff-evidence`):
s0_off       fc42da3  TORANN_PREFILTER_MIN_D=9999 --features bench-knobs
s0_1         fc42da3  TORANN_PREFILTER_MIN_D=1 --features bench-knobs
s0_17        fc42da3
s0_17b       fc42da3

# S5 transposed float32 (SoA) tile filter (branch `s5-soa-tile`):
s5_soa       b54cd88

# Historical Phase-1 arms (kept for reference):
# base       b06c7df
# base2      b06c7df
# p1e        372d8e0
# p1e_al     372d8e0 --rustflags "-C target-feature=+avx2,+fma -C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6"
# base_al    b06c7df --rustflags "-C target-feature=+avx2,+fma -C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6"
