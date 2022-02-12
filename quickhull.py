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
from shape import *


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

    # Farthest point to line(v0, v1)
    p_p0 = (points - v0)
    cp = np.cross((v1 - v0), p_p0)
    area = np.linalg.norm(cp, axis=1)
    farthest = np.argmax(area)
    if area[farthest] == 0:
        return None, False
    v2 = points[farthest]
    n = cp[farthest]/area[farthest]

    # Farthest point to plane(v0, v1, v2)
    dp = np.dot(p_p0, n)
    dist = np.abs(dp)
    farthest = np.argmax(dist)
    if dist[farthest] == 0:
        return None, False
    v3 = points[farthest]

    if dp[farthest] < 0:
        print('Front')
        tetrahedron = np.array([v2, v1, v0, v3])
    else:
        print('Back')
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
    v0, v1, v2, v3 = simplex
    faces = [Polygon(v2, v1, v0, ), Polygon(v0, v1, v3), Polygon(v1, v2, v3), Polygon(v2, v0, v3), ]
    
    vertices = np.array([face.vertices[0] for face in faces])
    normals = np.array([face.normal for face in faces])

    below, inside = points_above_planes(points, vertices, normals)
    above, outside = np.logical_not(below), np.logical_not(inside)

    tetra = LineMesh(simplex, [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], colors=[1, 0, 0], radius=0.01).cylinder_set
    indices = above&outside[:, None]
    mask = np.logical_not(indices[:, 0])
    nfaces = len(faces)
    print('total points, inside point, outside points:', points.shape[0], inside.sum(), outside.sum())
    for i in range(nfaces):
        faces[i].points = points[indices[:, i]]
        if i < (nfaces - 1):
            indices[:, i + 1] = indices[:, i + 1]&mask
            mask = mask&np.logical_not(indices[:, i])

        pcd = to_pcd(faces[i].points, [1, 0.5, 0], viz=False, )
        o3d.visualization.draw_geometries([pcd, tetra])
    exit()

    if nondegenerate:
        # Points visualization
        inside_pcd = to_pcd(points[inside], [1, 0.5, 0], viz=False, )
        outside_pcd = to_pcd(points[np.logical_not(inside)], [0, 0, 1], viz=False, )
        pcd = inside_pcd + outside_pcd
        # pcd = inside_pcd
        # Faces visualization
        tetra = LineMesh(simplex, [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], colors=[1, 0, 0], radius=0.01).cylinder_set
        # Normals visualization
        npoints = [] 
        nedges = []
        for i, face in enumerate(faces):
            normal_start = face.vertices[0]
            normal_end = normal_start + face.normal
            npoints.append(normal_start)
            npoints.append(normal_end)
            nedges.append([2*i, 2*i + 1])
        nmesh = LineMesh(npoints, nedges, colors=[0, 1, 0], radius=0.01).cylinder_set
        
        o3d.visualization.draw_geometries([pcd, tetra, nmesh])


if __name__ == '__main__':
    points = sample_sphere(1000, minr=0.0, maxr=1.0)
    quickhull(points)

