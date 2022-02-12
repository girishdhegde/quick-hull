import numpy as np


__author__ = "__Girish_Hegde__"


def points_above_planes(points, vertices, normals, ):
    """ Function to check if points are above/below given planes.
        It can be used for checking points inside/outside polyhedron.
        author = "Girish D Hegde'
        contact = "girish.dhc@gmail.com"

    Args:
        points (np.ndarray): [N, 3] - matrix of points.
        vertices (np.ndarray): [M, 3] - matrix of point on faces.
        normals (np.ndarray): [M, 3] matrix of normal of faces.
    Returns:
        (np.ndarray, np.ndarray) - [N, M] - above planes bools, [N, ] - inside points bools.
    """
    
    point_vectors = vertices[None, :, :] - points[:, None, :]
    distances = np.einsum('ijk, jk -> ij', point_vectors, normals)
    sign = distances > 0 
    signsum = np.sum(sign, axis=1)
    inside = (signsum == 0)|(signsum == len(normals))
    return sign, inside


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
        self.edge_vectors = []
        for (start, end) in self.edges:
            self.edge_vectors.append(self.vertices[end] - self.vertices[start])
    
    def distance(self, points):
        """ Function to calculate distance of points from plane

        Args:
            points (np.ndarray): [N, 3] - array of xyz points.
        Returns:
            (np.ndarray) - [N, ] dot products(signed distances).
            (np.ndarray) - [N, ] absolute disances.
        """
        p_p0 = points - self.vertices[0]
        dp = np.dot(p_p0, self.normal)
        dist = np.abs(dp)
        return dp, dist
    
    def common_edge(self, other):
        """ Function to find common edge and uncommon vertex

        Args:
            other (Polygon): adjacent polygon face.
        
        Returns:
            [np.ndarray, np.ndarray]: [(xn, yn, zn), (xm, ym, zm)]  - common edge
            (np.ndarray): (x, y, z) - uncommon vertex
        """
        for i, pt in enumerate(other.vertices):
            for v in self.vertices:
                if (pt == v).all():
                    break
            else:
                break
        return [v for iv, v in enumerate(other.vertices) if i != iv], pt

    def inside(self, point):
        """ Function to check if point inside polygon

        Args:
            point (np.ndarray): [x, y, z] - point
        Returns:
            if inside:
                (np.ndarray): [x, y, z] - point
            else:
                None
        """
        for (start, end), edge in zip(self.edges, self.edge_vectors):
            c = point - self.vertices[start]
            if np.dot(self.normal, np.cross(edge, c)) < 0:
                return None
        return point

    def intersect(self, origin, direction):
        """ Function to check ray(origin, direction) intersection

        Args:
            origin (np.ndarray): [x, y, z] - ray origin.
            direction (np.ndarray): [x_, y_, z_] - ray unit direction.
        Returns:
            if intersects:
                (np.ndarray): [x, y, z] - point of intersection.
            else:
                None
        """
        denom = self.normal.dot(direction)
        # normals are opposite in direction
        x = None
        if denom < -1e-6:
            t = ((self.vertices[0] - origin).dot(self.normal))/denom
            x = origin + t*direction
        if denom > 1e-6:
            t = ((self.vertices[0] - origin).dot(self.normal))/denom
            x =  origin + t*direction
        if x is not None:
            return self.inside(x)
        return None


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
