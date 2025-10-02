"""Command-line entry point for VMF utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmf_tools import (
    CompileOptions,
    Entity,
    ToolchainPaths,
    VMFBuilder,
    Vector3,
    compile_map,
    preview_file,
)


def build_from_json(path: Path, output: Path) -> None:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    builder = VMFBuilder()
    for brush in data.get('brushes', []):
        min_corner = Vector3(*brush['min'])
        max_corner = Vector3(*brush['max'])
        material = brush.get('material', 'DEV/DEV_MEASUREWALL01C')
        builder.add_axis_aligned_block(min_corner, max_corner, material)
    for entity_data in data.get('entities', []):
        classname = entity_data['classname']
        props = {key: str(value) for key, value in entity_data.get('properties', {}).items()}
        brushes = []
        for brush_data in entity_data.get('brushes', []):
            min_corner = Vector3(*brush_data['min'])
            max_corner = Vector3(*brush_data['max'])
            material = brush_data.get('material', 'DEV/DEV_MEASUREWALL01C')
            color = brush_data.get('editor_color', '220 30 220 255')
            brushes.append(
                VMFBuilder.axis_aligned_brush(min_corner, max_corner, material=material, editor_color=color)
            )
        builder.add_entity(Entity(classname=classname, properties=props, brushes=brushes))
    builder.save(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="VMF toolkit")
    sub = parser.add_subparsers(dest='command')

    preview_cmd = sub.add_parser('preview', help='Preview a VMF file in 3D')
    preview_cmd.add_argument('path', type=Path, help='Path to the VMF file')

    build_cmd = sub.add_parser('build', help='Generate a VMF file from JSON description')
    build_cmd.add_argument('description', type=Path, help='Input JSON description')
    build_cmd.add_argument('output', type=Path, help='Output VMF path')

    compile_cmd = sub.add_parser('compile', help='Compile a VMF file with VBSP/VVIS/VRAD')
    compile_cmd.add_argument('vmf', type=Path, help='VMF file to compile')
    compile_cmd.add_argument('--game', type=Path, help='Path passed to -game for VBSP')
    compile_cmd.add_argument('--vbsp', default='vbsp', help='Path to vbsp executable')
    compile_cmd.add_argument('--vvis', default='vvis', help='Path to vvis executable')
    compile_cmd.add_argument('--vrad', default='vrad', help='Path to vrad executable')
    compile_cmd.add_argument('--threads', type=int, help='Number of threads for the tools')
    compile_cmd.add_argument('--fast-vis', action='store_true', help='Run VVIS with -fast')
    compile_cmd.add_argument('--fast-vrad', action='store_true', help='Run VRAD with -fast')
    compile_cmd.add_argument('--final', action='store_true', help='Run VRAD with -final (high quality)')
    compile_cmd.add_argument(
        '--lighting', choices=['both', 'hdr', 'ldr'], default='both', help='Which lightmaps VRAD should bake'
    )
    compile_cmd.add_argument('--extra-vbsp', nargs='*', default=[], help='Additional arguments for VBSP')
    compile_cmd.add_argument('--extra-vvis', nargs='*', default=[], help='Additional arguments for VVIS')
    compile_cmd.add_argument('--extra-vrad', nargs='*', default=[], help='Additional arguments for VRAD')

    args = parser.parse_args()
    if args.command == 'preview':
        preview_file(str(args.path))
    elif args.command == 'build':
        build_from_json(args.description, args.output)
    elif args.command == 'compile':
        paths = ToolchainPaths(vbsp=args.vbsp, vvis=args.vvis, vrad=args.vrad, game_dir=args.game)
        options = CompileOptions(
            fast_vis=args.fast_vis,
            fast_vrad=args.fast_vrad and not args.final,
            final_vrad=args.final,
            lighting=args.lighting,
            threads=args.threads,
            extra_vbsp=tuple(args.extra_vbsp),
            extra_vvis=tuple(args.extra_vvis),
            extra_vrad=tuple(args.extra_vrad),
        )
        compile_map(args.vmf, paths, options)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
