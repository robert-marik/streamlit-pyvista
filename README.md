# streamlit-pyvista

Fast PyVista viewer for Streamlit, a drop-in replacement for `stpyvista(plotter)`.

Why it's faster than stpyvista:

- the scene is sent as binary glTF (GLB, zlib-compressed), not as vtk.js JSON/base64 embedded in HTML
- the GLB is a static file named by a content hash, so the browser caches it and reruns transfer ~0 bytes
- duplicated vertices (the OBJ reader splits them per face corner) are welded, which makes the file ~3-5x smaller
- positions are quantized to uint16 (`KHR_mesh_quantization`)
- three.js renders it on the GPU

## Installation

```bash
pip install git+https://github.com/robert-marik/streamlit-pyvista
```

## Usage

```python
import pyvista as pv
import streamlit as st
from st_pv import pv_view

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(pv.Sphere(), color="red", label="Sphere")
plotter.add_axes()
plotter.show_grid()

picked = pv_view(plotter, height=600, key="scene")
st.write(picked)  # points the user clicked in "add point" mode

# or a single dataset, add_mesh kwargs are passed through
sphere = pv.Sphere()
sphere["z"] = sphere.points[:, 2]
pv_view(sphere, scalars="z", cmap="coolwarm", height=400)
```

### `pv_view(obj, height=600, background="#ffffff", brightness=1.0, key=None, **add_mesh_kwargs)`

- `obj`: a `pv.Plotter` (actors, labels, camera, `add_axes`, `show_grid` are used) or a `pv.DataSet`
- `brightness`: texture multiplier (0.2-3), applied on the GPU
- returns the list of picked points `[[x, y, z], ...]`

Supported from the plotter: surfaces, lines, point clouds, textures, scalars + colormaps
(point and cell data), color, opacity, style, point size, line width, `render_points_as_spheres`,
lighting, labels.

### Viewer toolbar

Surface / wireframe / points, pan mode, add points by clicking, undo last point,
texture brightness, layer visibility (by plotter labels), reset view, maximize.
Mouse: left = rotate, right/middle = pan, wheel = zoom; arrow keys pan.

### Cache

Scene files are stored in `st_pv/frontend/cache/` and the least recently used ones are deleted
above `st_pv.CACHE_LIMIT_MB` (default 1024). `st_pv.clear_cache()` deletes everything.
The package directory must be writable.

## Example and tests

```bash
uv sync
uv run streamlit run examples/app.py   # expects a scan in examples/scans/ (not in the repo)
uv run pytest
```

## License

MIT
