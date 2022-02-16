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


def classify(points, faces):
    """ Function to assign points
        face.points.append(pt) if pt above(pt, face).

    Args:
        points (np.ndarray): [N, 3] - points.
        faces (list[Face]): list of faces.
    """
    vertices = np.array([face.vertex for face in faces])
    normals = np.array([face.normal for face in faces])

    below, inside = points_above_planes(points, vertices, normals)
    above, outside = np.logical_not(below), np.logical_not(inside)

    indices = above&outside[:, None]
    mask = np.logical_not(indices[:, 0])
    nfaces = len(faces)
    for i in range(nfaces):
        above_pts = points[indices[:, i]]
        faces[i].points = above_pts
        if i < (nfaces - 1):
            indices[:, i + 1] = indices[:, i + 1]&mask
            mask = mask&np.logical_not(indices[:, i])


def triangle2edges(v0, v1, v2):
    """ Function to convert triangle vertices to half-edges

    Args:
        v0 (np.ndarray): [x, y, z] - vertex.
        v1 (np.ndarray): [x, y, z] - vertex.
        v2 (np.ndarray): [x, y, z] - vertex.
    """
    e0 = HalfEdge(v0, face=None, prev=None, nxt=None, twin=None)
    e1 = HalfEdge(v1, face=None, prev=e0, nxt=None, twin=None)
    e2 = HalfEdge(v2, face=None, prev=e1, nxt=e0, twin=None)
    e0.prev, e0.next = e2, e1
    e1.next = e2
    return [e0, e1, e2]


def insert_edge(edge,  edges=None):
    """ Function update edge list.

    Args:
        edge (HalfEdge/list[HalfEdge]): edge/edges to be inserted.
        edges (list[HalfEdge], optional): edge list. Defaults to None.

    Returns:
        list[HalfEdges]: updated list of edges. 
    """
    edges = [] if edges is None else edges 
    edge_list = [edge] if isinstance(edge, (HalfEdge)) else edge
    
    for edge in edge_list:
        tail = edge.tail
        tip = edge.next.tail
        # Find twin if exists
        for other in edges:
            if (other.tail == tip).all():
                if (other.next.tail == tail).all():
                    other.twin = edge
                    edge.twin =  other
                    break
        edges.append(edge)
    return edges


def init_edges_faces(simplex):
    """ Initialization function

    Args:
        simplex (list): init. tetrahedron vertices list.
    Returns:
        list[HalfEdges]: list of edges.
        list[Face]: list of faces.
    """
    v0, v1, v2, v3 = simplex
    edges = None
    faces = []
    e012 = triangle2edges(v0, v1, v3)
    faces.append(Face(e012[0]))
    edges = insert_edge(e012, edges)
    
    e012 = triangle2edges(v1, v2, v3)
    faces.append(Face(e012[0]))
    edges = insert_edge(e012, edges)

    e012 = triangle2edges(v2, v0, v3)
    faces.append(Face(e012[0]))
    edges = insert_edge(e012, edges)

    e012 = triangle2edges(v2, v1, v0)
    faces.append(Face(e012[0]))
    edges = insert_edge(e012, edges)

    return edges, faces


def faces2mesh(faces, clr=(1, 0.3, 0), radius=0.01, viz=False):
    """ faces list to mesh

    Args:
        faces (list[Face]): list of Faces.
        clr (tuple/list): color of mesh. Defaults to (1, 0.3, 0).
        radius (float): radius of edges). Defaults to 0.01.
        viz (bool): do visualize. Defaults to False.
    Returns:
        o3d.geometry.TriangleMesh: mesh
    """
    mesh = None
    for face in faces:
        vertices, edges, _ = face.to_mesh(normal=False)
        if mesh is None:
            mesh = LineMesh(vertices, edges, colors=clr, radius=radius).cylinder_set
        else:
            mesh += LineMesh(vertices, edges, colors=clr, radius=radius).cylinder_set
    if viz:
        o3d.visualization.draw_geometries([mesh, ])
    return mesh


def quickhull(points):
    """ Function to find 3D convex hull.

    Args:
        points (np.ndarray): [N, 3] - xyz coordinates.
    
    Returns:
        (np.ndarray, np.ndarray): [M, 3] - polyhedron vertices, [L, 3] - faces
    """
    simplex, nondegenerate = maximal_simplex(points)
    if not nondegenerate:
        print('Error: points form Degenerate simplex')
        return None

    edges, faces = init_edges_faces(simplex)
    tetra = faces2mesh(faces, clr=(1, 0.3, 0), radius=0.01, viz=True)
    
    classify(points, faces)
    # # Visualization
    # for face in faces:
    #     pcd = to_pcd(face.points, [0, 0, 1], viz=False, )
    #     o3d.visualization.draw_geometries([pcd, tetra])

    exit()


    while faces:
        # Find most distant point to face
        face = faces.pop()
        if (not len(face.points)) or (not face.on_hull): continue
        dp, distances = face.distance(face.points)
        farthest_idx = np.argmax(distances)
        farthest = face.points[farthest_idx]

        # Find all faces that can be seen from that point
        light_faces = [face]
        face.on_hull = False
        for nbr in face.neighbours:
            # edge, uc_vtx = face.common_edge(nbr)
            dp, dist = nbr.distance(farthest)
            if dp > 0:
                light_faces.append(nbr)
                nbr.on_hull = False






    # direction = uc_vtx - farthest
    # direction = direction / np.linalg.norm(direction)
    # x = face.intersect(farthest, direction)
    # if x is None:  # if visible
    #     light_faces.append(nbr)
    # #     xpt = fpt
    # else:
    #     xpt = to_pcd([x], [0, 0, 0], viz=False, )
    # Visualization
    # upt = to_pcd([uc_vtx], [0, 0, 1], viz=False, )
    # o3d.visualization.draw_geometries([fpt, upt, xpt, tri])
# print('light faces:', light_faces)
# if len(light_faces) > 1:
#     print("Sorry this case not handled")
#     fpt = to_pcd([farthest], [1, 0.5, 0], viz=False, )
#     tetra = LineMesh(simplex, [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], colors=[1, 0, 0], radius=0.01).cylinder_set
#     o3d.visualization.draw_geometries([fpt, tetra])



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

