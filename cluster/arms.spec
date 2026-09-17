# Arms for the current Phase 1 campaign. Columns: name rev [build flags].
# Revisions must exist in the synced history (push local branches first).
base    b06c7df
base2   b06c7df
p1e     372d8e0
p1e_al  372d8e0 --rustflags "-C target-feature=+avx2,+fma -C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6"
base_al b06c7df --rustflags "-C target-feature=+avx2,+fma -C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6"
