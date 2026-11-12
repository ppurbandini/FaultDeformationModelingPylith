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

#coordinates
x = vertices[:, 0] / 1000.0
y = vertices[:, 1] / 1000.0

slip_magnitude = np.linalg.norm(slip, axis=1)
tri = mtri.Triangulation(x, y, triangles=cells)

#plotting
fig, ax = plt.subplots(figsize=(10, 7))

image = ax.tripcolor(tri, slip_magnitude, shading="gouraud", cmap="viridis")

ax.set_title("Total Fault Slip Magnitude")
ax.set_xlabel("Along-dip coordinate (km)")
ax.set_ylabel("Along-strike coordinate (km)")
ax.set_aspect("equal")

cbar = fig.colorbar(image, ax=ax)
cbar.set_label("Slip magnitude (m)")

plt.tight_layout()
output_directory = Path(r"results")
os.makedirs(output_directory, exist_ok=True)
output_file = output_directory / "total_fault_slip_magnitude.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

