import numpy as np

# points = [[1., 0., 0.],
#           [0., 0., -1.],
#           [0., 0., 0.]]
points = np.asarray([[2., 0., 0.],
                     [0., 2., 0.],
                     [0., 0., 0.]])
tri = [[0, 1, 2]]
edges = [[[0, 1], [1, 2], [2, 0]]]

# get the normal of the facet
edge_0_vec = np.asarray([points[edges[0][0][1]][0] - points[edges[0][0][0]][0],
                         points[edges[0][0][1]][1] - points[edges[0][0][0]][1],
                         points[edges[0][0][1]][2] - points[edges[0][0][0]][2]])
edge_1_vec = np.asarray([points[edges[0][1][1]][0] - points[edges[0][1][0]][0],
                         points[edges[0][1][1]][1] - points[edges[0][1][0]][1],
                         points[edges[0][1][1]][2] - points[edges[0][1][0]][2]])
edge_2_vec = np.asarray([points[edges[0][2][1]][0] - points[edges[0][2][0]][0],
                         points[edges[0][2][1]][1] - points[edges[0][2][0]][1],
                         points[edges[0][2][1]][2] - points[edges[0][2][0]][2]])
edges_vec = np.asarray([edge_0_vec, edge_1_vec, edge_2_vec])
# print(edge_0_vec)
# print(edge_1_vec)
# print(edge_2_vec)
T = np.cross(edge_0_vec, edge_1_vec)
n = T / np.linalg.norm(T)
# print(n)

radius = 0.2
C = np.asarray([3.3, 0.3, radius - radius / 10])

# ===================================================
# 4.1 check if the particle is intersecting the plane
# ===================================================
d = np.dot(n, C - points[0])

# ===================================================
# 4.2 Inside-Outside test
# ===================================================
if d <= radius:
    # the projected center onto the plane
    C_projected = C - d * n
    for i, edge in enumerate(edges[0]):
        # print(edge)
        # print(C_projected - points[edges[0][i][0]])
        edge_cross_c_proj = np.cross(edges_vec[i], C_projected - points[edges[0][i][0]])
        print(edge_cross_c_proj )
        s_a = np.dot(edge_cross_c_proj, n)
        if s_a < 0.:
            break
    print(edge)
