"""
Name : point_cloud.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 31/01/21 08:47 م
Desc:
"""
import numpy as np
import open3d as o3d


def compute_occupied_space(depth, ground_img, fx_d=5.8262448167737955e+02, fy_d=5.8269103270988637e+02,
                           cx_d=3.1304475870804731e+02, cy_d=2.3844389626620386e+02):
    indices = np.where(np.all(ground_img != [0, 255, 0], axis=-1))
    pcd = np.array([(indices[1] - cx_d) * depth[indices] / fx_d,
                    -(indices[0] - cy_d) * depth[indices] / fy_d,
                    depth[indices]])
    return np.dstack(pcd)[0]


def downsampling(pcd, voxel_size_mm=5):
    pcl = o3d.PointCloud()
    pcl.points = o3d.Vector3dVector(pcd)
    pcd = o3d.geometry.voxel_down_sample(pcl, voxel_size=voxel_size_mm)
    return pcd.points

def normalization(data):
    # unit sphere:
    centroid = np.mean(data, axis=0)
    data -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(data) ** 2, axis=-1)))
    data /= furthest_distance
    return data


if __name__ == '__main__':
    import cv2
    import time
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt

    # compute occupied space:
    depth = cv2.imread("../_in/0_d.png", cv2.CV_16U)
    ground = cv2.imread('../_in/ground.png')
    start = time.time()
    pcd = compute_occupied_space(depth, ground)
    print(f"Computed pcd in {time.time() - start} s.")

    # down sampling:
    start = time.time()
    pcd = downsampling(pcd, voxel_size_mm=10)
    print(f"Down-sampled pcd in {time.time() - start} s.")

    # segmentation:
    start = time.time()
    pcd = normalization(pcd)
    print(f"Normalized pcd in {time.time() - start} s.")
    model = DBSCAN(eps=0.05, min_samples=60)
    start = time.time()
    model.fit(pcd)
    print(f"Segmentation in {time.time() - start} s.")
    labels = model.labels_

    # clustering visualization:
    n_clusters = len(set(labels))
    colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    colors[labels < 0] = 0

    # dispaly:
    pcl = o3d.PointCloud()
    pcl.points = o3d.Vector3dVector(pcd)
    pcl.colors = o3d.Vector3dVector(colors[:, :3])
    o3d.draw_geometries([pcl])
