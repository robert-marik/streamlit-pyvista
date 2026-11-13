"""Self-check: GLB built from a Plotter decodes back to the same triangles and points."""

import json
import struct
import zlib

import numpy as np
import pyvista as pv

import st_pv


def _decode(glb):
    n = struct.unpack_from("<I", glb, 12)[0]
    g, bin_ = json.loads(glb[20 : 20 + n]), glb[28 + n :]

    def acc(i):
        a = g["accessors"][i]
        v = g["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a["componentType"]]
        k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        stride = v.get("byteStride", k * np.dtype(dt).itemsize) // np.dtype(dt).itemsize
        x = np.frombuffer(bin_, dt, a["count"] * stride, v["byteOffset"]).reshape(-1, stride)[:, :k]
        return x / np.iinfo(dt).max if a.get("normalized") else x

    out = []
    for node in g["nodes"]:
        for p in g["meshes"][node["mesh"]]["primitives"]:
            pts = acc(p["attributes"]["POSITION"]) * node["scale"] + node["translation"]
            idx = acc(p["indices"]).reshape(-1, {4: 3, 1: 2}[p["mode"]]) if "indices" in p else None
            out.append((p["mode"], pts, idx))
    return out


def test_roundtrip():
    sphere = pv.Sphere()
    sphere["z"] = sphere.points[:, 2]
    cloud = np.random.default_rng(0).random((50, 3))
    p = pv.Plotter(off_screen=True)
    p.add_mesh(sphere, scalars="z")
    p.add_mesh(pv.PolyData(cloud), color="red", render_points_as_spheres=True, label="cloud")
    glb = zlib.decompress(zlib.compress(st_pv._glb([st_pv._layer(a, {}) for a in st_pv._actors(p)])))
    (m1, pts, idx), (m2, cpts, _) = _decode(glb)
    assert (m1, m2) == (4, 0)
    tri = lambda x: np.sort(np.round(x.reshape(-1, 9), 3), axis=0)
    assert np.allclose(tri(sphere.points[sphere.regular_faces]), tri(pts[idx]), atol=1e-3)
    assert np.allclose(np.sort(cloud, 0), np.sort(cpts, 0), atol=1e-4)


if __name__ == "__main__":
    test_roundtrip()
    print("OK")
