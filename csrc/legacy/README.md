# Legacy learning experiments

These files preserve the original early learning stages and profiling history.
They are deliberately excluded from `setup.py` because their input contracts
are inconsistent and some assume sequence lengths that are exact tile
multiples.

Use them as implementation notes, not as correctness or performance baselines.
When promoting an idea back into the active kernels, first add tail masking,
the unified BHND contract, current-stream launching, and tests.

Names use `fwd_fp32_lNN_*` for chronological learning stages and
`fwd_fp32_xNN_*` for incomplete experiments. Legacy numbering is intentionally
separate from the active `kernels/simt/fwd_fp32_NN_*` sequence.
