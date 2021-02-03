"""
Name : main.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 30/01/21 02:33 م
Desc:
"""
from DCGD.Components.DCGD import *
from DCGD.Components.Visualizer import *
from BASISR.objects import Segment
from Scene.point_cloud import compute_occupied_space, downsampling, normalization, compute_ground_height
from sklearn.cluster import DBSCAN
from BASISR.simulator import BASISR
import keras


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
    min_prob = 0.96
    CLASSES_6_MAP = {0: 'bathtub5', 1: 'chair', 2: 'dresser', 3: 'downstair', 4: 'table', 5: 'bathtub5'}
    cls_model = keras.models.load_model("pointnet")
    cls_model.compile()
    # MAPPING
    # init visualizer:
    vis = o3d.visualization.Visualizer()
    vis.create_window("BASSAR")
    vis.get_render_option().background_color = [0.75, 0.75, 0.75]
    vis.get_render_option().mesh_show_back_face = True
    # create BASISR:
    small_base = 50
    base = 250
    height = 183.52
    basisr = BASISR(small_base, base, height)
    vis.add_geometry(basisr.create_base([255, 255, 255]))
    # ####################################################################################
    # PROCESSING:
    # ####################################################################################
    # Ground Detection:
    _, _, _, minimalFloorPoints = dcgd.cgd_process_downsampling(depth)
    Visualizer.viz_on_depth_downsampling(pretty, depth, minimalFloorPoints, interval,
                                         h_err, step, cy_d, fy_d)
    # Point cloud:
    ground_y = compute_ground_height(depth, pretty, fy_d, cy_d)
    occupied_space_pcd = compute_occupied_space(depth, pretty, fx_d, fy_d, cx_d, cy_d)
    occupied_space_pcd = downsampling(occupied_space_pcd, voxel_size_mm=10)
    occupied_space_pcd = occupied_space_pcd - [0, ground_y, 0]

    # Segmentation:
    normlized_pcd = normalization(occupied_space_pcd)
    seg_model.fit(normlized_pcd)

    # construct Segments:
    clusters_pointnet = []
    clusters = []
    labels = seg_model.labels_
    core_samples_mask = np.zeros_like(seg_model.labels_, dtype=bool)
    core_samples_mask[seg_model.core_sample_indices_] = True
    unique_labels = set(labels)
    for cls in unique_labels:
        if cls != -1:  # not noise
            class_member_mask = (labels == cls)
            clusters += [occupied_space_pcd[class_member_mask & core_samples_mask]]
            pcd = normlized_pcd[class_member_mask & core_samples_mask]
            clusters_pointnet += [pcd[np.random.choice(len(pcd), n_samples, replace=True)]]

    # Classification:
    predicted = cls_model.predict(np.array(clusters_pointnet))
    cls_labels = predicted.argmax(axis=-1)
    cls_probs = predicted.max(axis=-1)

    # compute segments:
    segments = []
    for i in range(len(clusters)):
        if cls_probs[i] > min_prob:
            cls = CLASSES_6_MAP[cls_labels[i]]
        else:
            cls = None
        # create segments to map
        seg = Segment(clusters[i], ground_y, cls)
        segments.append(seg)

    # Mapping:
    basisr.map_segments(segments)
    basisr.update_pinsClolor()
    pins = basisr.create_pins()
    print(f'Pins: {len(pins)}')
    for p in pins:
        vis.add_geometry(p)
    # visualize:
    vis.run()
    vis.destroy_window()
