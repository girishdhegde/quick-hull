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


def horizon(eye, face, edges=None):
    """ Function to get horizon edges as seen from eye

    Note:
        This fn. runs DFS and collects horizon edges in CCW order.
        Horizon edge: edge between visible and non-visible faces.

    Args:
        eye (np.ndarray): [x, y, z] - point.
        face (Face): current face.
        edges (list[HalfEdge]): horizon edge collector list.

    Returns:
        list[HalfEdge]: horizon edges.
    """
    if face.visited: return edges
    face.visited = True
    edges = [] if edges is None else edges

    # Visibility test
    pp = eye - face.vertex
    dp = np.dot(pp, face.normal)
    if dp <= 0:
        # Collect edge in CCW order
        edges.append(face.edge.twin)
        return edges

    face.visible = True
    e1 = face.edge
    e0, e2 = e1.prev, e1.next
    face_edges = [e0, e1, e2]

    # DFS
    for edge in face_edges:
        edges = horizon(eye, edge.twin.face, edges)

    return edges


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


def delete_face(faces, edges, index=None, face=None):
    """ Fuction to delete face.

    Args:
        faces (list[Face]): list of faces.
        edges (list[HalfEdge]): list of edges.
        index (int, optional): delete faces[index]. Defaults to None.
        face (Face, optional): delete face. Defaults to None.

    Returns:
        list[Face]: list of faces.
        list[HalfEdge]: list of edges.
    """
    index = -1 if (face is None) and (index is None) else index
    if index is None:
        for i, other in enumerate(faces):
            if other == face:
                index = i
                break
    face = faces.pop(index)
    e1 = face.edge
    e0, e2 = e1.prev, e1.next
    delids = []
    for edge in [e0, e1, e2]:
        edge.twin.twin = None
        for i, other in enumerate(edges):
            if edge == other:
                delids.apppend(i)
    delids = [idx - i for i, idx in enumerate(delids.sort())]
    for idx in delids:
        edges.pop(idx)
    return faces, edges


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

    hull = []
    while faces:
        # Initial ops.
        face = faces.pop()
        if (not len(face.points)):
            hull.append(face)
            continue 
        if face.visited: continue

        # Find most distant point to face
        dp, distances = face.distance(face.points)
        idx = np.argmax(distances)
        eye = face.points[idx]

        horizons = horizon(eye, face)
        print(horizons)

        exit()


if __name__ == '__main__':
    points = sample_sphere(1000, minr=0.0, maxr=1.0)
    quickhull(points)

