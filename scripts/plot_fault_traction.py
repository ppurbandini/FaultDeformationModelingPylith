import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
from pathlib import Path

filename = "sources/output/Nepal_kinematic_model-fault.h5"

with h5py.File(filename, "r") as f:
    vertices = f["geometry/vertices"][:]
    cells = f["topology/cells"][:]
    traction = f["vertex_fields/traction_change"][0, :, :]

cells = np.rint(cells).astype(int)

#tranction components and coordinates
x = vertices[:, 0] / 1000.0
y = vertices[:, 1] / 1000.0

left_lateral = traction[:, 0] / 1e6
updip = traction[:, 1] / 1e6

tri = mtri.Triangulation(x, y, triangles=cells)

#plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plots = [(left_lateral, "Left-lateral traction change"), (updip, "Updip traction change")]
max_abs = max(np.max(np.abs(v)) for v, _ in plots)

if max_abs == 0:
    max_abs = 1.0

for ax, (values, title) in zip(axes, plots):
    image = ax.tripcolor(tri, values, shading="gouraud", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)

    ax.set_title(title)
    ax.set_xlabel("Along-dip coordinate (km)")
    ax.set_ylabel("Along-strike coordinate (km)")
    ax.set_aspect("equal")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Traction change (MPa)")
    
ax.set_title("Spatial Distribution of Fault Traction Changes")

plt.tight_layout()
output_directory = Path(r"results")
os.makedirs(output_directory, exist_ok=True)
output_file = output_directory / "fault_traction_change.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
