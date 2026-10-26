from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    """User-visible capabilities of a native teaching backend."""

    name: str
    family: str
    dtype: str
    minimum_compute_capability: tuple[int, int] | None = None
    tile_multiple: tuple[int, ...] | None = None
    atol: float = 1e-3
    rtol: float = 1e-3

