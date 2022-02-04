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
