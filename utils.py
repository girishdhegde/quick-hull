import os
import glob 
import shutil
import copy

import tqdm
import numpy as np
import cv2
import open3d as o3d


__author__ = "__Girish_Hegde__"


def to_pcd(points, colors=None, viz=False, filepath=None):
    """ Function to convert points array into o3d.PointCloud
        author: Girish D. Hegde - girish.dhc@gmail.com

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        colors (np.ndarray/List, optional): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.

    Returns:
        (o3d.PointCloud): point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            pcd.colors =  o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color(colors)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    if filepath is not None:
        o3d.io.write_point_cloud(filepath, pcd)
    return pcd


def to_mesh(points, faces, colors=None, viz=False, filepath=None):
    """ Function to convert points array into o3d.geometry.TriangleMesh
        author: Girish D. Hegde - girish.dhc@gmail.com

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        faces (np.ndarray): [M, 3] - list of triangle faces of points.
        colors (np.ndarray/List, optional): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.

    Returns:
        (o3d.geometry.TriangleMesh): mesh
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            mesh.vertex_colors =  o3d.utility.Vector3dVector(colors)
        else:
            mesh.paint_uniform_color(colors)
    if viz:
        o3d.visualization.draw_geometries([mesh])
    if filepath is not None:
        o3d.io.write_triangle_mesh(filepath, mesh)
    return mesh


def to_image(
        img, 
        norm=False, 
        save=None, 
        show=True, 
        delay=0, 
        rgb=True, 
        bg=0,
    ):
    """ Function to show/save image 
        author: Girish D. Hegde - girish.dhc@gmail.com

    Args:
        img (np.ndarray): [h, w, ch] image(grayscale/rgb)
        norm (bool, optional): min-max normalize image. Defaults to False.
        save (str, optional): path to save image. Defaults to None.
        show (bool, optional): show image. Defaults to True.
        delay (int, optional): cv2 window delay. Defaults to 0.
    
    Returns:,
        (np.ndarray): [h, w, ch] - image.
    """
    if rgb:
        img = img[..., ::-1]
    if norm:
        img = (img - img.min())/(img.max() - img.min())
    if save is not None:
        if img.max() <= 1:
            img *=255
        cv2.imwrite(save, img.astype(np.uint8))
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
    return img


def sample_sphere(n=100, center=(0, 0, 0), minr=0, maxr=1, mintheta=0, maxtheta=2*np.pi, minphi=0, maxphi=1):
    """Function to Sample points Between Two concentric Spheres using Inverse Transform Samples
       Reference: https://github.com/girishdhegde/random.fun/tree/master/sampling
                  http://corysimon.github.io/articles/uniformdistn-on-sphere/

    polar coords -> (r, theta, phi)
    theta = uniform[0, 2.pi)
    phi = arccos(1 - 2.uniform[0, 1))
    r = cbrt(uniform(0, 1)) <- inverse transform
    
    cartesian
    x = r.cos(theta).sin(phi)
    y = r.sin(theta).sin(phi)
    z = r.cos(phi)

    Args:
        n (int, optional): number of points. Defaults to 100.
        minr (float, optional): radius of inner sphere
        maxr (float, optional): radius of outer sphere
        mintheta (float, optional): minimum angle wrt x. Defaults to 0
        maxtheta (float, optional): maximum angle wrt x. Defaults to 2pi
        minphi (float, optional): minimum angle wrt z. Defaults to 0
        maxphi (float, optional): maximum angle wrt z. Defaults to 1
    """

    ranger = (maxr - minr)
    ranget = (maxtheta - mintheta)
    rangep = (maxphi - minphi)

    theta = mintheta + np.random.random(n)*ranget
    phi = np.arccos(1 - 2*(minphi + np.random.random(n)*rangep))
    radius = np.random.random(n)**(1/3)
    
    radius = minr + radius*ranger
    # theta = mintheta + radius*ranget
    # phi = minphi + radius*rangep
    
    x = radius*np.cos(theta)*np.sin(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = radius*np.cos(phi)

    cx, cy, cz = center
    x, y, z = x + cx, y + cy, z + cz

    return np.vstack([x, y, z]).T


if __name__ == "__main__":
    points = sample_sphere(100, minr=0.6, maxr=1.0)
    to_pcd(points, [0, 0, 1], viz=True, )