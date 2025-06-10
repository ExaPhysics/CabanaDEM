import meshio
import numpy as np

# Load the mesh file
mesh = meshio.read("../inputs/square_tri.msh")  # Gmsh output
meshio.write("square_tri.vtk", mesh)

# Extract triangle elements
triangles = mesh.cells_dict.get("triangle")
if triangles is None:
    raise ValueError("No triangle elements found in the mesh.")

# Get node coordinates
points = mesh.points  # shape: (n_points, 3)

# Compute centroids of each triangle
centroids = np.mean(points[triangles], axis=1)

# Print centroids
print("Triangle Centroids:")
for i, c in enumerate(centroids):
    print(f"Triangle {i}: (x={c[0]:.3f}, y={c[1]:.3f}, z={c[2]:.3f})")


points = mesh.points
# print(points)
print(triangles)

def compute_inradius(p1, p2, p3):
    # Side lengths
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    s = 0.5 * (a + b + c)  # semi-perimeter
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula
    r = area / s if s > 0 else 0
    return r

# Compute inradii for each triangle
inradii = []
for tri in triangles:
    p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
    r = compute_inradius(p1, p2, p3)
    inradii.append(r)

inradii = np.array(inradii)
max_r = np.max(inradii)

print(f"Maximum inradius among all triangles: {max_r:.6f}")
