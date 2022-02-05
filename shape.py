import numpy as np


__author__ = "__Girish_Hegde__"


class Polygon:
    def __init__(self, *vertices, normal=None, edges=None):
        """ Polygon

        Args:
            vertices (list0, list1, list2, ...): vertices - v0[x0, y0], v1, v2, ...
            normal (np.ndarray, optional): [3, ] unit normal to the plane. Defaults to None.
            edges ([list[list]], optional): [len(vertices), 2] - edges list. Defaults to None.
        """
        self.vertices = np.array(vertices)
        if normal is None:
            v0v1 = vertices[1] - vertices[0]
            v1v2 = vertices[2] - vertices[1]
            normal = np.cross(v0v1, v1v2)
            normal = normal/np.linalg.norm(normal)
        self.normal = normal
        self.nvertices = len(self.vertices)
        if edges is None:
            edges = [[i, i + 1] for i in range(self.nvertices - 1)]
            edges.append([self.nvertices - 1, 0])
        self.edges = edges


def ray_x_plane(origin, direction, vertex, normal):
    """ Function to check if ray(origin, direction) hits plane(vertex, normal)

    Args:
        origin (list/np.ndarray): [x, y, z] - ray origin.
        direction (list/np.ndarray): [x_, y_, z_] - direction vector of ray.
        vertex (list/np.ndarray): [x, y, z] - point on plane.
        normal (list/np.ndarray): [x_, y_, z_] - unit normal to the plane.

    Returns:
        [np.ndarray/None]: [x, y, z] - intersection point or None.
    """
    denom = normal.dot(direction)
    # normals are opposite in direction
    if denom < -1e-6:
        t = ((vertex - origin).dot(normal))/denom
        return origin + t*direction
    if denom > 1e-6:
        t = ((vertex - origin).dot(normal))/denom
        return origin + t*direction
    return None


# def inside(vertex, normal, points):
#     """ Function to check if points inside polygon

#     Args:
#         vertex ([type]): [description]
#         normal ([type]): [description]
#         points ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     for i in range(len(v)-1):
#         edge = v[i+1] - v[i]
#         c = p - v[i]
#         if N.dot(edge.cross(c)) < 0:
#             return None
#     edge = v[0] - v[-1]
#     c = p - v[-1]
#     if N.dot(edge.cross(c)) < 0:
#         return None
#     return p


# def _rayTriX(self, obj):
#     p, _ = self._rayPlaneX(obj.plane)
#     if p is None:
#         return None, None
#     # inside - outside
#     p = insideOutside(p, obj.norm, obj.v0, obj.v1, obj.v2)

#     if p is None:
#         return None, None
    
#     # check if triangle is behind the ray
#     toP = p - self.source
#     if toP.dot(self.direction) < 0:
#         return None, None

#     return p, obj.norm     
