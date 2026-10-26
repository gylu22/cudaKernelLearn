from cuda_kernel_lab import flash_attention, sgemm


def test_sgemm_registry_describes_tile_constraints():
    assert sgemm.DEFAULT_IMPLEMENTATION in sgemm.BACKENDS
    assert sgemm.BACKENDS["simt_fp32_00_naive"].tile_multiple is None
    assert sgemm.BACKENDS["simt_fp32_04_double_buffer"].tile_multiple == (
        128,
        128,
        8,
    )


def test_flash_attention_registry_preserves_existing_backends():
    assert flash_attention.DEFAULT_IMPLEMENTATION in flash_attention.BACKENDS
    assert "tensor_core_tf32_wmma" in flash_attention.IMPLEMENTATIONS

