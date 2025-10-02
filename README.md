# VMF Toolkit

This repository provides a pure Python toolkit for working with Source 1
Valve Map Format (VMF) files. It contains two major pieces of functionality:

* **Interactive viewer** – open an existing VMF and inspect all brush geometry
  in real time using a simple OpenGL viewer. Geometry is rendered with flat
  white shading, which makes it easy to review proportions and lighting without
  textures.
* **Programmatic builder** – create new VMF maps from Python code. The builder
  focuses on axis-aligned brush construction but can be extended for more
  complex scenarios. It generates valid VMF documents that can be compiled with
  the Source SDK toolchain.

## Requirements

* Python 3.10+
* [pyglet](https://pyglet.org) 2.0 or newer for the viewer

Install the dependencies with `pip install -r requirements.txt` (see below).

## Installation

Create a virtual environment and install the toolkit in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs `pyglet` for the previewer. The toolkit itself is pure Python and
has no compiled extensions.

## Command-line usage

The repository includes the `vmf_tool.py` helper script:

```bash
python vmf_tool.py preview path/to/map.vmf
```

Opens the VMF file in the viewer window. Use the mouse to orbit the camera
(left button) and zoom (wheel or right-button drag). `WASDQE` pans the focus
point.

```bash
python vmf_tool.py build description.json output.vmf
```

Generates a VMF from a minimal JSON description. The JSON schema is:

```json
{
  "brushes": [
    {"min": [x0, y0, z0], "max": [x1, y1, z1], "material": "DEV/DEV_MEASUREWALL01C"}
  ],
  "entities": [
    {
      "classname": "light",
      "properties": {"origin": "0 0 64", "_light": "255 255 255 200"}
    },
    {
      "classname": "func_detail",
      "properties": {"origin": "0 0 0"},
      "brushes": [
        {"min": [-64, -64, 0], "max": [64, 64, 128], "material": "TOOLS/TOOLSNODRAW"}
      ]
    }
  ]
}
```

All coordinates use Hammer Units. The builder currently supports axis-aligned
rectangular solids. Entity-attached brushes may also include an optional
`editor_color` field to influence how Hammer displays them.

```bash
python vmf_tool.py compile output.vmf --game /path/to/game --threads 8 --final
```

Runs VBSP, VVIS and VRAD on the specified VMF. Use the `--vbsp`, `--vvis` and
`--vrad` options to point at custom tool locations, `--fast-vis` / `--fast-vrad`
for quick builds, and `--final` plus `--lighting` (`both`, `hdr`, or `ldr`) for
production-quality lighting.

## AI-assisted VMF generation

The repository now ships with a geometry-focused workflow that learns directly
from brush layouts instead of raw VMF text. A variational autoencoder (VAE)
ingests existing maps, analyses the distribution of axis-aligned brushes and
materials, and then samples entirely new arrangements that are converted back
into VMF solids.

### Training a model

Collect a directory of `.vmf` files and launch training with:

```bash
python scripts/train_vmf_language_model.py /path/to/vmfs checkpoints/vmf-generator \
    --epochs 25 --batch-size 8 --latent-dim 96
```

Shell and Windows batch helpers are available if you prefer:

```bash
scripts/train_vmf_language_model.sh /path/to/vmfs checkpoints/vmf-generator
```

```bat
scripts\train_vmf_language_model.bat C:\path\to\vmfs checkpoints\vmf-generator
```

During training the script automatically extracts brush bounding boxes, builds
a material vocabulary, normalises spatial features, and saves the learned
statistics next to each checkpoint.

### Generating a map

After training, generate a new VMF by sampling from a checkpoint:

```bash
python scripts/generate_vmf.py checkpoints/vmf-generator/epoch_025.pt \
    generated/map.vmf --presence-threshold 0.55
```

Wrapper scripts run the same command and automatically open the result in the
included viewer (falls back to a warning when a display is not available):

```bash
scripts/generate_vmf_and_preview.sh checkpoints/vmf-generator/epoch_025.pt generated/map.vmf
```

```bat
scripts\generate_vmf_and_preview.bat checkpoints\vmf-generator\epoch_025.pt generated\map.vmf
```

The generator predicts which brushes should exist, their materials, and their
axis-aligned extents. The helper script converts the sampled brushes back into
VMF solids that Hammer and the included tooling can consume.

## Library usage

Import the toolkit in your own Python code:

```python
from vmf_tools import VMFBuilder, Vector3

builder = VMFBuilder()
builder.add_axis_aligned_block(Vector3(-256, -256, 0), Vector3(256, 256, 128))
builder.save("example.vmf")
```

You can also load and preview VMFs programmatically:

```python
from vmf_tools import load_vmf, preview_vmf

vmf = load_vmf("example.vmf")
preview_vmf(vmf)
```

To compile a map from code:

```python
from pathlib import Path

from vmf_tools import CompileOptions, ToolchainPaths, compile_map

paths = ToolchainPaths(game_dir=Path(r"C:\\Steam\\steamapps\\common\\Half-Life 2\\hl2"))
options = CompileOptions(final_vrad=True, lighting="hdr", threads=8)
compile_map("example.vmf", paths, options)
```

## Module overview

* `vmf_tools/geometry.py` – vector math utilities and polygon helpers.
* `vmf_tools/parser.py` – tokenises VMF text into Python objects and rebuilds
  brush polygons for rendering.
* `vmf_tools/builder.py` – high-level API for constructing VMF files via code.
* `vmf_tools/viewer.py` – pyglet-based OpenGL viewer with simple lighting.
* `vmf_tool.py` – CLI wrapper combining the parser, viewer and builder.

## Notes

* The parser currently focuses on brush geometry (solids). Entities without
  brushes are preserved but not visualised.
* Displacements and other advanced VMF features are not supported yet.
* Viewer shading uses a single directional light and renders all faces with
  white material, which makes geometry inspection easier without textures.
