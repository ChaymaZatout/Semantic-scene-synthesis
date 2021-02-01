"""
Name : main.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 30/01/21 02:33 م
Desc:
"""
from DCGD.Components.DCGD import *
from DCGD.Components.Visualizer import *
from Scene.point_cloud import compute_occupied_space, downsampling, normalization
from sklearn.cluster import DBSCAN

# import keras

if __name__ == '__main__':
    import open3d as o3d
    # Read Data:
    results_dir = "_out/"
    depth = cv2.imread('_in/0_d.png', cv2.CV_16U)
    pretty = cv2.imread('_in/0_p.png')

    # Camera parameters:
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02
    interval = (800, 4000)

    # ####################################################################################
    # initialization:
    # ####################################################################################
    # GROUND DETECTION
    h_err = 20
    size_err = 3
    step = 15
    dcgd = DCGD(cy_d, fy_d, interval, h_err, size_err, step)
    # SEGMENTATION
    seg_model = DBSCAN(eps=0.05, min_samples=60)
    # CLASSIFICATION
    n_samples = 2048
    # cls_model = keras.models.load_model("pointnet")

    # ####################################################################################
    # PROCESSING:
    # ####################################################################################
    # Ground Detection:
    _, _, _, minimalFloorPoints = dcgd.cgd_process_downsampling(depth)
    Visualizer.viz_on_depth_downsampling(pretty, depth, minimalFloorPoints, interval,
                                         h_err, step, cy_d, fy_d)

    # Point cloud:
    occupied_space_pcd = compute_occupied_space(depth, pretty, fx_d, fy_d, cx_d, cy_d)
    occupied_space_pcd = downsampling(occupied_space_pcd, voxel_size_mm=10)
    occupied_space_pcd = normalization(occupied_space_pcd)

    # Segmentation:
    seg_model.fit(occupied_space_pcd)

    # clustering visualization:
    labels = seg_model.labels_
    n_clusters = len(set(labels))
    colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    colors[labels < 0] = 0

    # dispaly:
    pcl = o3d.PointCloud()
    pcl.points = o3d.Vector3dVector(occupied_space_pcd)
    pcl.colors = o3d.Vector3dVector(colors[:, :3])
    o3d.draw_geometries([pcl])

    # construct Segments:
    clusters = []
    labels = seg_model.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    core_samples_mask = np.zeros_like(seg_model.labels_, dtype=bool)
    core_samples_mask[seg_model.core_sample_indices_] = True
    unique_labels = set(labels)
    for cls in unique_labels:
        if cls != -1:  # not noise
            class_member_mask = (labels == cls)
            pcd = occupied_space_pcd[class_member_mask & core_samples_mask]
            clusters += [pcd[np.random.choice(len(pcd), n_samples, replace=True)]]

    # Classification:

    # Mapping:
