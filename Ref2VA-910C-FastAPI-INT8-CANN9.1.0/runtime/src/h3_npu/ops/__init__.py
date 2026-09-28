"""H3 NPU ops — lazy imports to avoid clashing with custom OpCommand extensions."""

__all__ = [
    "all_to_all_heads",
    "build_hadamard",
    "rotate_activation",
    "int8_convrot_linear",
    "dequantize_nvfp4",
    "nvfp4_linear",
    "safe_cast_fp8",
    "fusion_attention",
]


def __getattr__(name: str):
    if name == "all_to_all_heads":
        from .ulysses import all_to_all_heads

        return all_to_all_heads
    if name in ("build_hadamard", "rotate_activation"):
        from . import hadamard

        return getattr(hadamard, name)
    if name == "int8_convrot_linear":
        from .int8_convrot import int8_convrot_linear

        return int8_convrot_linear
    if name in ("dequantize_nvfp4", "nvfp4_linear"):
        from . import nvfp4

        return getattr(nvfp4, name)
    if name == "safe_cast_fp8":
        from .float8_cast import safe_cast_fp8

        return safe_cast_fp8
    if name == "fusion_attention":
        from .attention import fusion_attention

        return fusion_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
