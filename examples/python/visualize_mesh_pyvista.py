import numpy as np
import pyvista as pv
import meshio
from stl import mesh

# Structured grid: 2x2 quads → 4 quads → 8 triangles
# Define points (3x3 grid → 9 points)
points = np.array([
    [0, 0, 0],
    [0.5, 0, 0],
    [1, 0, 0],
    [0, 0.5, 0],
    [0.5, 0.5, 0],
    [1, 0.5, 0],
    [0, 1, 0],
    [0.5, 1, 0],
    [1, 1, 0]
])

# Define 8 triangles from the 4 quads
triangles = [
    [0, 1, 4], [0, 4, 3],  # lower-left
    [1, 2, 5], [1, 5, 4],  # lower-right
    [3, 4, 7], [3, 7, 6],  # upper-left
    [4, 5, 8], [4, 8, 7]   # upper-right
]

# Save as .vtk or .msh etc.
cells = [("triangle", np.array(triangles))]
meshio_mesh = meshio.Mesh(points=np.array(points), cells=cells)
# Export to VTK for use in Paraview or similar
meshio.write("eight_triangles.vtk", meshio_mesh)


# Convert to PyVista format: prepend 3 to each triangle
faces = np.hstack([[3] + tri for tri in triangles])


# Create a sphere with a given radius and position
radius = 0.1  # Change to desired radius
sphere = pv.Sphere(radius=radius, center=(0.5, 0.5, 0.5))  # Position the sphere at (0.5, 0.5, 0.5)
mesh = pv.read("eight_triangles.vtk")

# Plot the mesh and the sphere together
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.add_mesh(sphere, color="red", opacity=0.5)  # Semi-transparent sphere
plotter.show()


# # Create and plot mesh
# mesh = pv.PolyData(points, faces)
# mesh.plot(show_edges=True, color="lightblue", line_width=2)


mesh = meshio.read("eight_triangles.vtk")
# meshio.write("eight_triangles.stl", mesh)

# Extract points and triangles
points = mesh.points  # shape (N, 3)
triangles = mesh.cells_dict["triangle"]  # shape (M, 3)
print(points)
print(triangles)

# Compute centroids (average of triangle vertices)
centroids = np.mean(points[triangles], axis=1)

# Compute triangle normals (for 3D surfaces)
v1 = points[triangles[:, 1]] - points[triangles[:, 0]]
v2 = points[triangles[:, 2]] - points[triangles[:, 0]]
normals = np.cross(v1, v2)

# Normalize normals
normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

# Print sample
# print("First centroid:", centroids[0])
# print("First normal:", normals[0])
print(centroids)
print(normals)
