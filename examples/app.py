import time

import numpy as np
import pyvista as pv
import streamlit as st

from st_pv import pv_view

st.set_page_config(layout="wide")
st.title("PyVista ve Streamlitu")


@st.cache_resource
def load():
    return pv.read("examples/scans/Be_len_ska_li_pa.obj"), pv.read_texture("examples/scans/Be_len_ska_li_pa.jpg")


mesh, texture = load()
opacity = st.slider("Průhlednost skenu", 0.0, 1.0, 1.0, 0.05)
brightness = st.slider("Jas textury", 0.2, 3.0, 1.0, 0.05)

# same scene as tomotree render_pyvista: scan + slice points + pivot + polygon
y = 1.0
band = mesh.points[np.abs(mesh.points[:, 1] - y) < 0.01]
pivot = band.mean(0)
ang = np.linspace(0, 2 * np.pi, 60)
polygon = pivot + 0.6 * np.c_[np.cos(ang), np.zeros_like(ang), np.sin(ang)]

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, texture=texture, style="surface", opacity=opacity, point_size=1.0, label="Point Cloud")
plotter.add_mesh(pv.PolyData(band), color="red", point_size=4, render_points_as_spheres=True, label="Slice Points")
plotter.add_mesh(pv.Sphere(radius=0.02, center=pivot), color="black", label="Pivot Point")
plotter.add_mesh(pv.PolyData(polygon), color="blue", opacity=0.5, label="Fitted Polygon 3D")
plotter.add_axes()
plotter.show_grid()
plotter.view_vector((0, 0, -1), viewup=(0, 1, 0))

t = time.perf_counter()
picked = pv_view(plotter, height=700, brightness=brightness, key="scan")
st.caption(f"Python strana: {(time.perf_counter() - t) * 1000:.0f} ms")
if picked:
    st.write("Přidané body:", picked)

st.subheader("Skaláry")
sphere = pv.Sphere(theta_resolution=120, phi_resolution=120)
sphere["z"] = sphere.points[:, 2]
pv_view(sphere, scalars="z", cmap="coolwarm", height=400)
