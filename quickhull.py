import os
import glob 
import shutil
import copy

import tqdm
import numpy as np
import cv2
import open3d as o3d

from linemesh import LineMesh
from utils import * 


__author__ = "__Girish_Hegde__"
__ref__ = """
    http://algolist.ru/maths/geom/convhull/qhull3d.php
"""


def maximal_simplex(points):
    """ Function to find 4 points which define maximal tetrahedron.

    Args:
        points (np.ndarray): [N, 3] - xyz coordinates.
    
    Returns:
        (np.ndarray, bool): [4, 3] - tetrahadron, nondegenerate
    """
    for dim in range(3):
        imin, imax = np.argmin(points[:, dim]), np.argmax(points[:, dim])
        if points[imin, dim] != points[imax, dim]:
            break
    else:
        return None, False
    v0, v1 = points[[imin, imax]]

    p_p0 = (points - v0)
    cp = np.cross((v1 - v0), p_p0)
    area = np.linalg.norm(cp, axis=1)
    farthest = np.argmax(area)
    if area[farthest] == 0:
        return None, False
    v2 = points[farthest]
    n = cp[farthest]/area[farthest]

    dp = np.dot(p_p0, n)
    dist = np.abs(dp)
    farthest = np.argmax(dist)
    if dist[farthest] == 0:
        return None, False
    v3 = points[farthest]

    if dp[farthest] < 0:
        tetrahedron = np.array([v2, v1, v0, v3])
    else:
        tetrahedron = np.array([v0, v1, v2, v3])

    return tetrahedron, True


def quickhull(points):
    """ Function to find 3D convex hull.

    Args:
        points (np.ndarray): [N, 3] - xyz coordinates.
    
    Returns:
        (np.ndarray, np.ndarray): [M, 3] - polyhedron vertices, [L, 3] - faces
    """
    simplex, nondegenerate = maximal_simplex(points)


def main():
    points = sample_sphere(100, minr=0.6, maxr=1.0)
    simplex, nondegenerate = maximal_simplex(points)

    if nondegenerate:
        pcd = to_pcd(points, [0, 0, 1], viz=True, )
        tetra = LineMesh(simplex, [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], colors=[1, 0, 0], radius=0.01).cylinder_set
        o3d.visualization.draw_geometries([pcd, tetra])


if __name__ == '__main__':
    main()

