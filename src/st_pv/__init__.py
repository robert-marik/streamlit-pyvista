"""Fast PyVista viewer for Streamlit, drop-in for ``stpyvista(plotter)``.

Why it's faster than stpyvista:
- scene is sent as binary glTF (GLB), not vtk.js JSON/base64 embedded in HTML
- GLB is a static file named by content hash -> browser caches it, reruns transfer ~0 bytes
- duplicated vertices (OBJ reader splits per face corner) are welded -> ~3-5x smaller
- three.js renders it on the GPU
"""

import atexit
import hashlib
import io
import json
import os
import shutil
import struct
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pyvista as pv
import streamlit.components.v1 as components
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

_FRONTEND = Path(__file__).parent / "frontend"
# GLBs live in a private tmp dir, served by a second (file-only) component; removed on exit
_CACHE = Path(tempfile.mkdtemp(prefix="st_pv-"))
atexit.register(shutil.rmtree, _CACHE, True)
_VERSION = hashlib.blake2b(Path(__file__).read_bytes()).digest()  # code change -> new keys, old files age out
CACHE_LIMIT_MB = 1024  # oldest-used files are deleted above this
_component = components.declare_component("st_pv", path=str(_FRONTEND))
_cache_server = components.declare_component("st_pv_cache", path=str(_CACHE))

_STYLE = {0: "points", 1: "wireframe", 2: "surface"}


def _srgb_to_linear(rgba_u8):
    c = rgba_u8.astype(np.float32) / 255
    c[:, :3] **= 2.2  # glTF vertex colors are linear
    return (c * 255 + 0.5).astype(np.uint8)


def _weld(points, *attrs):
    """Merge vertices identical in position and all attributes -> (kept indices, old->new index map)."""
    data = np.hstack([points, *[a.reshape(len(points), -1).view(np.float32) for a in attrs]])
    rows = np.ascontiguousarray(data).view(np.dtype((np.void, data.shape[1] * 4))).ravel()
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    return first, inverse.ravel().astype(np.uint32)


def _layer(actor, labels):
    """Extract geometry + look of one pv.Actor into plain numpy/dict."""
    mapper, prop = actor.mapper, actor.prop
    ds = mapper.dataset
    if not isinstance(ds, pv.PolyData):
        ds = ds.extract_surface(algorithm=None)
    if ds.n_points == 0:
        return None

    rgba = None
    if mapper.scalar_visibility and mapper.GetArrayName():
        cell = mapper.GetScalarMode() in (2, 4)
        arr = (ds.cell_data if cell else ds.point_data).get(mapper.GetArrayName())
        if arr is not None:
            rgba = vtk_to_numpy(mapper.lookup_table.MapScalars(numpy_to_vtk(np.asarray(arr)), 1, -1))
            if cell:  # ponytail: cell colors -> point colors by averaging; split faces if sharp cells matter
                rgba = ds.cell_data_to_point_data().point_data.get(mapper.GetArrayName())
                rgba = vtk_to_numpy(mapper.lookup_table.MapScalars(numpy_to_vtk(np.asarray(rgba)), 1, -1))
            rgba = _srgb_to_linear(rgba)

    tex = actor.texture.to_array() if actor.texture is not None else None
    uv = ds.active_texture_coordinates if tex is not None else None
    if uv is not None:
        uv = uv.astype(np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]  # VTK origin bottom-left, glTF top-left
        # per-vertex color from texture, so "points" view of a scan is colored
        h, w = tex.shape[:2]
        px = tex[(np.clip(uv[:, 1], 0, 1) * (h - 1)).astype(int), (np.clip(uv[:, 0], 0, 1) * (w - 1)).astype(int)]
        rgba = _srgb_to_linear(np.hstack([px[:, :3], np.full((len(px), 1), 255, np.uint8)]))
    else:
        tex = None

    tris = ds.triangulate().regular_faces if ds.n_faces else np.empty((0, 3), int)
    lines = ds.lines
    segs = []
    i = 0
    while i < len(lines):  # polylines -> segments
        n = lines[i]
        ids = lines[i + 1 : i + 1 + n]
        segs.append(np.c_[ids[:-1], ids[1:]])
        i += n + 1
    segs = np.vstack(segs) if segs else np.empty((0, 2), int)
    loose = len(tris) == 0 and len(segs) == 0  # point cloud

    color = prop.color
    return dict(
        points=np.asarray(ds.points, np.float32), tris=tris, segs=segs, loose=loose, uv=uv, rgba=rgba, tex=tex,
        lit=bool(prop.lighting) and tex is None, normals=bool(prop.lighting) and tex is None and len(tris) > 0,
        extras=dict(
            label=labels.get(actor.GetAddressAsString(""), ""),
            color=[float(c) for c in color.float_rgb], opacity=float(prop.opacity),
            style=_STYLE.get(actor.GetProperty().GetRepresentation(), "surface"),
            pointSize=float(prop.point_size), lineWidth=float(prop.line_width),
            spheres=bool(prop.render_points_as_spheres),
        ),
    )


def _glb(layers):
    blob, views, accessors, meshes, nodes, materials, images = bytearray(), [], [], [], [], [], []

    def pad():
        while len(blob) % 4:
            blob.append(0)

    def view(data, target=None, stride=None):
        pad()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), **({"target": target} if target else {}),
                      **({"byteStride": stride} if stride else {})})
        blob.extend(data)
        return len(views) - 1

    def add(arr, ctype, typ, target=34962, stride=None, **extra):
        accessors.append({"bufferView": view(arr.tobytes(), target, stride), "componentType": ctype, "count": len(arr), "type": typ, **extra})
        return len(accessors) - 1

    for L in layers:
        pts, uv, rgba = L["points"], L["uv"], L["rgba"]
        normals = pv.PolyData(pts, np.c_[np.full(len(L["tris"]), 3), L["tris"]].ravel()).point_normals.astype(np.float32) if L["normals"] else None
        if len(L["tris"]):  # weld duplicates (OBJ reader splits per face corner)
            keep, remap = _weld(pts, *[a for a in (uv, normals, rgba) if a is not None])
            tris, segs = remap[L["tris"]], remap[L["segs"]]
            pts, uv, normals, rgba = [a[keep] if a is not None else None for a in (pts, uv, normals, rgba)]
        else:
            tris, segs = L["tris"].astype(np.uint32), L["segs"].astype(np.uint32)

        # KHR_mesh_quantization: uint16 positions, uniform scale in node keeps normals valid
        lo = pts.min(0)
        ext = float((pts.max(0) - lo).max()) or 1.0
        qp = np.zeros((len(pts), 4), np.uint16)  # padded to 8 B stride (glTF needs 4 B alignment)
        qp[:, :3] = np.round((pts - lo) / ext * 65535)
        attrs = {"POSITION": add(qp, 5123, "VEC3", stride=8, normalized=True,
                                 min=qp[:, :3].min(0).tolist(), max=qp[:, :3].max(0).tolist())}
        if normals is not None:
            attrs["NORMAL"] = add(normals, 5126, "VEC3")
        if uv is not None:
            if uv.min() >= 0 and uv.max() <= 1:
                attrs["TEXCOORD_0"] = add(np.round(uv * 65535).astype(np.uint16), 5123, "VEC2", normalized=True)
            else:  # repeating texture
                attrs["TEXCOORD_0"] = add(uv, 5126, "VEC2")
        if rgba is not None:
            attrs["COLOR_0"] = add(rgba, 5121, "VEC4", normalized=True)

        mat = {"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1] if rgba is not None else [*np.power(L["extras"]["color"], 2.2).tolist(), 1],
                                        "metallicFactor": 0, "roughnessFactor": 1}, "doubleSided": True}
        if not L["lit"]:
            mat["extensions"] = {"KHR_materials_unlit": {}}
        if L["tex"] is not None:
            from PIL import Image

            b = io.BytesIO()
            Image.fromarray(L["tex"][..., :3]).save(b, "JPEG", quality=90)
            images.append({"bufferView": view(b.getvalue()), "mimeType": "image/jpeg"})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(images) - 1}
        materials.append(mat)

        prims = []
        if len(tris):
            prims.append({"attributes": attrs, "indices": add(tris.ravel(), 5125, "SCALAR", 34963), "mode": 4})
        if len(segs):
            prims.append({"attributes": attrs, "indices": add(segs.ravel(), 5125, "SCALAR", 34963), "mode": 1})
        if L["loose"]:
            prims.append({"attributes": attrs, "mode": 0})
        for p in prims:
            p["material"] = len(materials) - 1
        meshes.append({"primitives": prims})
        nodes.append({"mesh": len(meshes) - 1, "translation": lo.tolist(), "scale": [ext] * 3, "extras": L["extras"]})

    pad()
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
            "nodes": nodes, "meshes": meshes, "materials": materials,
            "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
            "extensionsUsed": ["KHR_materials_unlit", "KHR_mesh_quantization"],
            "extensionsRequired": ["KHR_mesh_quantization"]}
    if images:
        gltf |= {"images": images, "samplers": [{"magFilter": 9729, "minFilter": 9987}],
                 "textures": [{"source": i, "sampler": 0} for i in range(len(images))]}
        k = 0
        for m in materials:
            if "baseColorTexture" in m["pbrMetallicRoughness"]:
                m["pbrMetallicRoughness"]["baseColorTexture"]["index"] = k
                k += 1
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(blob)) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(blob), b"BIN\x00") + bytes(blob))


def _actors(plotter):
    return [a for a in plotter.renderer.actors.values()
            if isinstance(a, pv.Actor) and a.visibility and a.mapper is not None
            and isinstance(getattr(a.mapper, "dataset", None), pv.DataSet)]


def _key(actors):
    """Cheap content hash of everything that ends up in the GLB."""
    h = hashlib.blake2b(_VERSION, digest_size=12)
    for a in actors:
        ds, m, p = a.mapper.dataset, a.mapper, a.prop
        h.update(np.ascontiguousarray(ds.points).data)
        for cells in (getattr(ds, "faces", None), getattr(ds, "lines", None), getattr(ds, "cells", None)):
            if cells is not None:
                h.update(np.ascontiguousarray(cells).data)
        if m.scalar_visibility and m.GetArrayName():
            arr = ds.point_data.get(m.GetArrayName(), ds.cell_data.get(m.GetArrayName()))
            if arr is not None:
                h.update(np.ascontiguousarray(arr).data)
            h.update(repr((m.scalar_range, [m.lookup_table.GetTableValue(i) for i in range(0, 256, 15)])).encode())
        if a.texture is not None:
            t = a.texture.to_image().active_scalars
            h.update(repr(t.shape).encode())
            h.update(np.ascontiguousarray(t[:: max(1, len(t) // 100_000)]).data)  # ponytail: sampled texture hash, full hash costs ~0.3 s on 8k
            if ds.active_texture_coordinates is not None:
                h.update(np.ascontiguousarray(ds.active_texture_coordinates).data)
        h.update(repr((p.color.hex_rgba, p.opacity, p.GetRepresentation(), p.point_size, p.line_width,
                       p.render_points_as_spheres, p.lighting)).encode())
    return h.hexdigest()


def _store(name, build):
    """Write cache file once (atomic), touch on reuse, evict least recently used above CACHE_LIMIT_MB."""
    path = _CACHE / name
    if path.exists():
        os.utime(path)  # mtime = last use
        return
    tmp = path.with_suffix(f".{os.getpid()}.{id(build)}.tmp")
    tmp.write_bytes(build())
    tmp.replace(path)  # atomic, safe with concurrent sessions
    files = sorted(_CACHE.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    total = 0
    for f in files:
        st_ = f.stat()
        total += st_.st_size
        stale_tmp = f.suffix == ".tmp" and time.time() - st_.st_mtime > 3600
        if f != path and (total > CACHE_LIMIT_MB * 2**20 or stale_tmp):
            f.unlink(missing_ok=True)


def clear_cache():
    """Delete all cached scene files."""
    for f in _CACHE.glob("*"):
        f.unlink(missing_ok=True)


def pv_view(obj, height=600, background="#ffffff", brightness=1.0, key=None, **add_mesh_kwargs):
    """Render a pv.Plotter (or a single dataset) in Streamlit.

    obj: pv.Plotter (actors, labels, camera, add_axes, show_grid are used)
         or pv.DataSet (shown via ``Plotter.add_mesh(obj, **add_mesh_kwargs)``)
    brightness: texture multiplier (0.2-3), applied on the GPU; the viewer also has a ☀ slider
    Returns the list of points picked by the user in "add point" mode ([[x, y, z], ...]).
    """
    if isinstance(obj, pv.DataSet):
        p = pv.Plotter(off_screen=True)
        p.add_mesh(obj, **add_mesh_kwargs)
        obj = p
    actors = _actors(obj)
    # .glbz = zlib-compressed GLB, inflated in the browser by DecompressionStream; level 1: ~4x faster than 6, ~same size
    name = _key(actors) + ".glbz"
    labels = {k: v[1] for k, v in obj.renderer._labels.items()}
    _store(name, lambda: zlib.compress(_glb([L for a in actors if (L := _layer(a, labels))]), 1))
    if not obj.camera_set:
        obj.view_isometric()  # what plotter.show() would do
    pos, focal, up = obj.camera_position
    value = _component(
        url=f"../{_cache_server.name}/{name}", background=background, height=height, brightness=brightness,
        camera=dict(position=list(pos), focal=list(focal), up=list(up), viewAngle=obj.camera.view_angle),
        axes=obj.renderer.axes_widget is not None,
        grid=obj.renderer.cube_axes_actor is not None,
        key=key, default=None,
    )
    return (value or {}).get("points", [])
