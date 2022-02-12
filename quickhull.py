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
    if nondegenerate:
        tetra = LineMesh(simplex, [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], colors=[1, 0, 0], radius=0.01).cylinder_set

        # Assign outside points to faces
        v0, v1, v2, v3 = simplex
        faces = [Polygon(v2, v1, v0, ), Polygon(v0, v1, v3), Polygon(v1, v2, v3), Polygon(v2, v0, v3), ]
        
        vertices = np.array([face.vertices[0] for face in faces])
        normals = np.array([face.normal for face in faces])

        below, inside = points_above_planes(points, vertices, normals)
        above, outside = np.logical_not(below), np.logical_not(inside)

        indices = above&outside[:, None]
        mask = np.logical_not(indices[:, 0])
        nfaces = len(faces)
        temp = []
        print('total points, inside point, outside points:', points.shape[0], inside.sum(), outside.sum())
        for i in range(nfaces):
            faces[i].neighbours = [faces[nbr] for nbr in range(nfaces) if nbr != i]
            above_pts = points[indices[:, i]]
            faces[i].points = above_pts
            if i < (nfaces - 1):
                indices[:, i + 1] = indices[:, i + 1]&mask
                mask = mask&np.logical_not(indices[:, i])
            if len(above_pts):
                temp.append(faces[i])
            # # Visualization
            # pcd = to_pcd(faces[i].points, [1, 0.5, 0], viz=False, )
            # o3d.visualization.draw_geometries([pcd, tetra])
        faces = temp

        while faces:
            # Find most distant point to face
            face = faces.pop()
            if not len(face.points): continue
            dp, distances = face.distance(face.points)
            farthest_idx = np.argmax(distances)
            farthest = face.points[farthest_idx]
            # # Visualization
            # fpt = to_pcd([farthest], [1, 0.5, 0], viz=False, )
            # tri = LineMesh(face.vertices, face.edges, colors=[1, 0, 0], radius=0.01).cylinder_set
            # o3d.visualization.draw_geometries([fpt, tetra])

            # Find all faces that can be seen from that point
            light_faces = []
            for nbr in face.neighbours:
                edge, uc_vtx = face.common_edge(nbr)
                direction = uc_vtx - farthest
                direction = direction / np.linalg.norm(direction)
                x = face.intersect(farthest, direction)
                if x is None:  # if visible
                    light_faces.append(nbr)
                    xpt = fpt
                else:
                    xpt = to_pcd([x], [0, 0, 0], viz=False, )
                # Visualization
                # upt = to_pcd([uc_vtx], [0, 0, 1], viz=False, )
                # o3d.visualization.draw_geometries([fpt, upt, xpt, tri])
            print('light faces:', light_faces)





        # # Points visualization
        # inside_pcd = to_pcd(points[inside], [1, 0.5, 0], viz=False, )
        # outside_pcd = to_pcd(points[np.logical_not(inside)], [0, 0, 1], viz=False, )
        # pcd = inside_pcd + outside_pcd
        # # pcd = inside_pcd
        # # Normals visualization
        # npoints = [] 
        # nedges = []
        # for i, face in enumerate(faces):
        #     normal_start = face.vertices[0]
        #     normal_end = normal_start + face.normal
        #     npoints.append(normal_start)
        #     npoints.append(normal_end)
        #     nedges.append([2*i, 2*i + 1])
        # nmesh = LineMesh(npoints, nedges, colors=[0, 1, 0], radius=0.01).cylinder_set
        
        # o3d.visualization.draw_geometries([pcd, tetra, nmesh])


if __name__ == '__main__':
    points = sample_sphere(1000, minr=0.0, maxr=1.0)
    quickhull(points)

