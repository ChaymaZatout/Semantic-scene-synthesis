"""
Name : point_cloud.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 31/01/21 08:47 م
Desc:
"""
import numpy as np


def compute_occupied_space(depth, ground_img, fx_d=5.8262448167737955e+02, fy_d=5.8269103270988637e+02,
                           cx_d=3.1304475870804731e+02, cy_d=2.3844389626620386e+02):
    indices = np.where(np.all(ground_img != [0, 255, 0], axis=-1))
    pcd = np.array([(indices[1] - cx_d) * depth[indices] / fx_d,
                    -(indices[0] - cy_d) * depth[indices] / fy_d,
                    depth[indices]])
    return np.dstack(pcd)[0]


if __name__ == '__main__':
    import cv2
    import open3d as o3d
    import time

    # compute occupied space:
    depth = cv2.imread("../_in/0_d.png", cv2.CV_16U)
    ground = cv2.imread('../_in/ground.png')
    start = time.time()
    pcd = compute_occupied_space(depth, ground)
    print(f"Computed pcd in {time.time() - start} s.")

    # dispaly:
    pcl = o3d.PointCloud()
    pcl.points = o3d.Vector3dVector(pcd)
    o3d.draw_geometries([pcl])
