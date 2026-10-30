# Arms built by `build_arms.sbatch`.
# Columns: name  rev  [KEY=VAL ...] [extra `bench_ab.py build` flags]
# Revisions must exist in the synced history.

# One link per core idea: each arm adds one idea to the one before it.
c0           b06c7df
c1           dfde4cf
c2           0d566d1
c3           fc611bb
s3           726c836

# The S3 fix, and its copy for A/A noise calibration. HEAD must hold the fix.
s3fix        HEAD
s3fix_aa     HEAD

# S0 / W8 8-bit prefilter dimension-cutoff study:
s0_off       fc42da3  TORANN_PREFILTER_MIN_D=9999 --features bench-knobs
s0_1         fc42da3  TORANN_PREFILTER_MIN_D=1 --features bench-knobs
s0_17        fc42da3
s0_17b       fc42da3

# S5 transposed float32 (SoA) tile filter. Built on c2 and compared with c2;
# the commit is kept by refs/archive/s5-soa-tile.
s5_soa       b54cd88
