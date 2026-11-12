import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors
import os
from pathlib import Path

filename = "sources/output/Nepal_kinematic_model.h5"

with h5py.File(filename, "r") as f:
    vertices = f["geometry/vertices"][:]
    cells = f["topology/cells"][:]
    displacement = f["vertex_fields/displacement"][0, :, :]

cells = np.rint(cells).astype(int)

vertices_km = vertices / 1000.0
u_magnitude = np.linalg.norm(displacement, axis=1) #total displacement magnitude (m)

all_faces = np.concatenate([cells[:, [0, 1, 2]],
                            cells[:, [0, 1, 3]],
                            cells[:, [0, 2, 3]],
                            cells[:, [1, 2, 3]]])

all_faces = np.sort(all_faces, axis=1)

unique_faces, counts = np.unique(all_faces, axis=0, return_counts=True)

boundary_faces = unique_faces[counts == 1]
# print("Number of exterior faces:", len(boundary_faces))

#calculate average displacement/faces
face_values = np.mean(u_magnitude[boundary_faces], axis=1)

vmin = 0
vmax = np.percentile(u_magnitude, 99) #99 percentile
norm = colors.Normalize(vmin=vmin, vmax=vmax)

#visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

triangles_3d = vertices_km[boundary_faces]

surface = Poly3DCollection(triangles_3d, cmap="viridis", norm=norm, edgecolor="none")
surface.set_array(face_values)
ax.add_collection3d(surface)

#plot limit
ax.set_xlim(vertices_km[:, 0].min(), vertices_km[:, 0].max())
ax.set_ylim(vertices_km[:, 1].min(), vertices_km[:, 1].max())
ax.set_zlim(vertices_km[:, 2].min(), vertices_km[:, 2].max())

#axis labels
ax.set_xlabel("Along-dip coordinate (km)")
ax.set_ylabel("Along-strike coordinate (km)")
ax.set_zlabel("Depth (km)")

ax.set_title("3D Domain Displacement")

#color bar
cbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1, extend="max")
cbar.set_label("Displacement magnitude (m)")

ax.set_box_aspect((np.ptp(vertices_km[:, 0]),
                   np.ptp(vertices_km[:, 1]),
                   np.ptp(vertices_km[:, 2])))

# Viewing angle
ax.view_init(elev=25, azim=-55) #changeable

plt.tight_layout()
output_directory = Path(r"results")
os.makedirs(output_directory, exist_ok=True)
output_file = output_directory / "domain_displacement.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
