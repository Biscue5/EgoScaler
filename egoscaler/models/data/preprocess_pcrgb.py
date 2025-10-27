import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import os
import multiprocessing as mp
from egoscaler.configs.camera import CameraConfig as camera_cfg

# Global variables to be shared among worker processes
global_id2data = {}
global_root_dir = ""
global_split = ""

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def get_points_colors(rgb, depth, width, height,
                      principal_point,  # [cx, cy]
                      focal_lengths,    # [fx, fy]
                      d_thres=None):
    
    cx, cy = principal_point
    fx, fy = focal_lengths
    
    image = np.array(rgb)
    z = np.array(depth, dtype=np.float32)  # Convert to float32 just in case
    
    # Create image coordinates (u, v)
    #  x-axis direction: [0, 1, 2, ..., width-1]
    #  y-axis direction: [0, 1, 2, ..., height-1]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Pinhole model: (u - cx) / fx, (v - cy) / fy
    x = (x - cx) / fx
    y = (y - cy) / fy
    
    # Calculate 3D points as [X, Y, Z]
    # X = x * Z, Y = y * Z, Z = Z
    #  shape is (height, width, 3) → reshape to (height*width, 3)
    points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)
    
    # Normalize RGB values to [0.0, 1.0] and correspond them
    colors = image.reshape(-1, 3) / 255.0

    # Create mask to determine valid pixels
    #  e.g.: Complete black (0,0,0) is invalid or depth=0 is also invalid
    valid_color_indices = np.all(image != 0, axis=2)  # (H, W) with True/False
    if d_thres is not None:
        valid_depth_indices = (z < d_thres) & (z > 0)  # OK if greater than 0 and less than or equal to d_thres
        valid_indices = valid_color_indices & valid_depth_indices
    else:
        valid_indices = valid_color_indices

    valid_indices = valid_indices.reshape(-1)  # Flatten to (H*W,)
    points = points[valid_indices, :]
    colors = colors[valid_indices, :]

    return points, colors

def farthest_point_sample(point, npoint):
    """
    Input:
        point: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        sampled_pointcloud: sampled pointcloud data, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    sampled_point = point[centroids]
    return sampled_point

def initializer(id2data, root_dir, split):
    global global_id2data
    global global_root_dir
    global global_split
    global_id2data = id2data
    global_root_dir = root_dir
    global_split = split

def process_annotation(annot):
    try:
        image_id = annot['image_id']
        data = global_id2data.get(image_id, None)
        if data is None:
            return  # Skip if no data found for image_id

        dataset_name = data['dataset_name']
        video_uid = data['video_uid']
        file_name = data['file_name']

        output_path = os.path.join(global_root_dir, 'pcrgbs', dataset_name, video_uid, f"{file_name}.npy")
        if os.path.exists(output_path):
            return  # Skip if already processed

        # Load image and depth
        image_path = os.path.join(global_root_dir, 'obs_images', dataset_name, video_uid, f"{file_name}.jpg")
        depth_path = os.path.join(global_root_dir, 'depths', dataset_name, video_uid, f"{file_name}.npy")
        
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        width, height = pil_image.size
        depth = np.load(depth_path)

        if dataset_name in ["egoexo4d", "hot3d"]:
            principal_point = (camera_cfg.devices.aria.principal_point, camera_cfg.devices.aria.principal_point)
            focal_lengths = (camera_cfg.devices.aria.focal_len, camera_cfg.devices.aria.focal_len)
        elif dataset_name == "h2o":
            principal_point = (camera_cfg.devices.h2o.principal_point_x, camera_cfg.devices.h2o.principal_point_y)
            focal_lengths = (camera_cfg.devices.h2o.focal_len_x, camera_cfg.devices.h2o.focal_len_y)
        else:
            camera_intrs_path = os.path.join(global_root_dir, 'camera_intrs', dataset_name, video_uid, f"{file_name}.npy")
            camera_intrs = np.load(camera_intrs_path) 
            principal_point = (camera_intrs[0], camera_intrs[0])
            focal_lengths = (camera_intrs[1], camera_intrs[2])
            # NOTE: we do not consider distortion
            
        # Generate point cloud
        points, colors = get_points_colors(
            image, depth, width, height, 
            principal_point=principal_point, focal_lengths=focal_lengths,
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Voxel downsampling
        for voxel_size in [0.05, 0.03, 0.01]:
            down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            if np.asarray(down_pcd.points).shape[0] >= 8192:
                break
            
        if np.asarray(down_pcd.points).shape[0] < 8192:
            print(np.asarray(down_pcd.points).shape[0])
            return

        # Farthest point sampling
        num_samples = 8192  
        points_down = np.asarray(down_pcd.points)
        if points_down.shape[0] >= num_samples:
            fps_points = farthest_point_sample(points_down, num_samples)
        else:
            fps_points = points_down  # Use all points if fewer than num_samples

        fps_pcd = o3d.geometry.PointCloud()
        fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        # Assign colors using nearest neighbor
        tree = o3d.geometry.KDTreeFlann(down_pcd)
        colors_assigned = []
        down_colors = np.asarray(down_pcd.colors)
        for point in fps_points:
            [k, idx, _] = tree.search_knn_vector_3d(point, 1)
            colors_assigned.append(down_colors[idx[0]])
        fps_pcd.colors = o3d.utility.Vector3dVector(colors_assigned)

        points_array = np.asarray(fps_pcd.points)
        colors_array = np.asarray(fps_pcd.colors)
        pc_rgb = np.concatenate((points_array, colors_array), axis=1)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Save the processed point cloud
        np.save(output_path, pc_rgb)
    except Exception as e:
        print(f"Error processing annotation {annot.get('id', 'unknown')}: {e}")

def main(args):
    # Load dataset
    dataset_path = os.path.join(f"{args.split}.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    id2data = {entry['id']: entry for entry in dataset['images']}
    annotations = dataset['annotations']
    
    if args.end_index == -1:
        annotations = annotations[args.start_index:]
    else:
        annotations = annotations[args.start_index:args.end_index]

    # Set up multiprocessing pool
    num_workers = mp.cpu_count()  # You can set this to a specific number if desired
    pool = mp.Pool(processes=num_workers, initializer=initializer, initargs=(id2data, args.root_dir, args.split))

    # Use tqdm to display progress
    for _ in tqdm(pool.imap_unordered(process_annotation, annotations), total=len(annotations)):
        pass

    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset to generate point clouds with colors.")
    parser.add_argument('--split', type=str, required=True, help="Dataset split (e.g., train, val, test)")
    parser.add_argument('--root_dir', type=str, default="your/dataset/path", help="Root directory of the dataset")
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)
    args = parser.parse_args()
    main(args)
