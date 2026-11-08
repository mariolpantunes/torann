# Arms built by `build_arms.sbatch`.
# Columns: name  rev  [KEY=VAL ...] [extra `bench_ab.py build` flags]
# Revisions must exist in the synced history.

# One link per core idea: each arm adds one idea to the one before it.
c0           b06c7df
c1           dfde4cf
c2           0d566d1
c3           fc611bb
s3           726c836

# The S3 fix, drift/promote retune, and S7 2-wall probing (with A/A copy).
s3fix        e108e98
s3fix_aa     e108e98
retune       b70775b
s7           -
s7_aa        -

# S0 / W8 8-bit prefilter dimension-cutoff study:
s0_off       fc42da3  TORANN_PREFILTER_MIN_D=9999 --features bench-knobs
s0_1         fc42da3  TORANN_PREFILTER_MIN_D=1 --features bench-knobs
s0_17        fc42da3
s0_17b       fc42da3

# S5 transposed float32 (SoA) tile filter. Built on c2 and compared with c2;
# the commit is kept by refs/archive/s5-soa-tile.
s5_soa       b54cd88
