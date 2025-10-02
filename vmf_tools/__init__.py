"""Public API for the Source 1 VMF toolkit."""

from .builder import VMFBuilder, Entity, Brush, BrushFace
from .parser import load_vmf, parse_vmf, VMFMap
from .geometry import Vector3
from .compiler import CompileOptions, ToolchainPaths, compile_map


def preview_vmf(vmf: VMFMap) -> None:
    from .viewer import preview_vmf as _preview_vmf

    _preview_vmf(vmf)


def preview_file(path: str) -> None:
    from .viewer import preview_file as _preview_file

    _preview_file(path)

__all__ = [
    "VMFBuilder",
    "Entity",
    "Brush",
    "BrushFace",
    "Vector3",
    "VMFMap",
    "load_vmf",
    "parse_vmf",
    "preview_vmf",
    "preview_file",
    "compile_map",
    "CompileOptions",
    "ToolchainPaths",
]
