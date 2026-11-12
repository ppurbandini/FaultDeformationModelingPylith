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
    slip = f["vertex_fields/slip"][0, :, :]

cells = np.rint(cells).astype(int)

#slip components and coordinates
x = vertices[:, 0] / 1000.0 #coordinate in km
y = vertices[:, 1] / 1000.0 #coordinate in km

left_lateral = slip[:, 0]
updip = slip[:, 1]

tri = mtri.Triangulation(x, y, triangles=cells)

#plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plots = [(left_lateral, "Left-lateral slip", "Slip (m)"), (updip, "Updip slip", "Slip (m)")]

for ax, (values, title, colorbar_label) in zip(axes, plots):
    image = ax.tripcolor(tri, values, shading="gouraud", cmap="RdBu_r")

    ax.set_title(title)
    ax.set_xlabel("Along-dip coordinate (km)")
    ax.set_ylabel("Along-strike coordinate (km)")
    ax.set_aspect("equal")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    
ax.set_title("Spatial Distribution of Fault Slip Components")

plt.tight_layout()
output_directory = Path(r"results")
os.makedirs(output_directory, exist_ok=True)
output_file = output_directory / "fault_slip_components.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
