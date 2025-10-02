"""Utilities for compiling VMF files with the Source SDK toolchain."""
from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class ToolchainPaths:
    """Locations of the Source SDK compilation tools."""

    vbsp: str | Path = "vbsp"
    vvis: str | Path = "vvis"
    vrad: str | Path = "vrad"
    game_dir: Optional[Path] = None


@dataclass
class CompileOptions:
    """Options that control VBSP/VVIS/VRAD invocation."""

    fast_vis: bool = False
    fast_vrad: bool = False
    final_vrad: bool = False
    lighting: str = "both"
    threads: Optional[int] = None
    extra_vbsp: Sequence[str] = field(default_factory=tuple)
    extra_vvis: Sequence[str] = field(default_factory=tuple)
    extra_vrad: Sequence[str] = field(default_factory=tuple)


def _resolve_tool(tool: str | Path) -> str:
    path = Path(tool)
    if path.is_file():
        return str(path)
    located = shutil.which(str(tool))
    if located:
        return located
    raise FileNotFoundError(f"Unable to locate tool '{tool}'. Provide an absolute path or add it to PATH.")


def _run_command(name: str, args: Sequence[str]) -> None:
    print(f"[{name}] Executing: {' '.join(shlex.quote(arg) for arg in args)}")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{name}] {line.rstrip()}")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} failed with exit code {return_code}.")


def compile_map(
    vmf_path: str | Path,
    paths: ToolchainPaths | None = None,
    options: CompileOptions | None = None,
) -> Path:
    """Compile the provided VMF file into a BSP using VBSP/VVIS/VRAD."""

    vmf_path = Path(vmf_path)
    if not vmf_path.exists():
        raise FileNotFoundError(f"VMF file '{vmf_path}' does not exist.")

    paths = paths or ToolchainPaths()
    options = options or CompileOptions()

    if options.fast_vrad and options.final_vrad:
        raise ValueError("Cannot use both fast_vrad and final_vrad options at the same time.")
    lighting = options.lighting.lower()
    if lighting not in {"both", "hdr", "ldr"}:
        raise ValueError("lighting must be one of 'both', 'hdr', or 'ldr'.")

    vbsp_exe = _resolve_tool(paths.vbsp)
    vvis_exe = _resolve_tool(paths.vvis)
    vrad_exe = _resolve_tool(paths.vrad)

    bsp_path = vmf_path.with_suffix(".bsp")

    vbsp_args: list[str] = [vbsp_exe, str(vmf_path)]
    if paths.game_dir is not None:
        vbsp_args.extend(["-game", str(paths.game_dir)])
    if options.threads:
        vbsp_args.extend(["-threads", str(options.threads)])
    vbsp_args.extend(options.extra_vbsp)
    _run_command("VBSP", vbsp_args)

    vvis_args: list[str] = [vvis_exe, str(bsp_path)]
    if options.fast_vis:
        vvis_args.append("-fast")
    if options.threads:
        vvis_args.extend(["-threads", str(options.threads)])
    vvis_args.extend(options.extra_vvis)
    _run_command("VVIS", vvis_args)

    vrad_args: list[str] = [vrad_exe, str(bsp_path)]
    if options.fast_vrad:
        vrad_args.append("-fast")
    if options.final_vrad:
        vrad_args.append("-final")
    if lighting == "hdr":
        vrad_args.append("-hdr")
    elif lighting == "ldr":
        vrad_args.append("-ldr")
    else:
        vrad_args.append("-both")
    if options.threads:
        vrad_args.extend(["-threads", str(options.threads)])
    vrad_args.extend(options.extra_vrad)
    _run_command("VRAD", vrad_args)

    print(f"[compile] BSP written to {bsp_path}")
    return bsp_path
