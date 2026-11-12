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

x = vertices[:, 0] / 1000
y = vertices[:, 1] / 1000


slip_magnitude = np.linalg.norm(slip, axis=1)

u = slip[:, 1]  # updip component (0)
v = slip[:, 0]  # left-lateral component (1)

n_bins = 25
ix = np.clip(np.digitize(x, np.linspace(x.min(), x.max(), n_bins + 1)) - 1, 0, n_bins - 1)
iy = np.clip(np.digitize(y, np.linspace(y.min(), y.max(), n_bins + 1)) - 1, 0, n_bins - 1)

labels = ix * n_bins + iy
selected = []

for label in np.unique(labels):
    indices = np.where(labels == label)[0]

    xc = np.mean(x[indices])
    yc = np.mean(y[indices])
    distance = (x[indices] - xc)**2 + (y[indices] - yc)**2

    selected.append(indices[np.argmin(distance)])

selected = np.array(selected)

#plotting
tri = mtri.Triangulation(x, y, triangles=cells)

fig, ax = plt.subplots(figsize=(11, 8))
mesh = ax.tripcolor(tri, slip_magnitude, shading="gouraud", cmap="RdBu_r")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Total slip magnitude (m)")

arrow_exaggeration = 1000

ax.quiver(x[selected], y[selected], u[selected] / 1000 * arrow_exaggeration, v[selected] / 1000 * arrow_exaggeration,
          angles="xy", scale_units="xy", scale=1, color="black", width=0.0025, headwidth=4)

ax.set_xlabel("Along-dip coordinate (km)")
ax.set_ylabel("Along-strike coordinate (km)")
ax.set_title("Total Fault Slip Magnitude and Direction")
ax.set_aspect("equal")

plt.tight_layout()
output_directory = Path(r"results")
os.makedirs(output_directory, exist_ok=True)
output_file = output_directory / "total_slip_vectors.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()