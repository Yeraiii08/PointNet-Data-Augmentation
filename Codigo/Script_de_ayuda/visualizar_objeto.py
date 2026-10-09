# Open a .laz file to visualize the point cloud

import laspy
import open3d as o3d
import numpy as np

def visualize_laz(file_path):
    print(f"Loading: {file_path}...")
    
    las = laspy.read(file_path)
    
    # Extract (X, Y, Z) coordinates from the LAZ points
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create Open3D PointCloud object
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(points)

    # Map colors correctly (otherwise it displays weird rainbow colors)
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        o3d_cloud.colors = o3d.utility.Vector3dVector(colors / 65535.0)

    # Visualize
    print(f"Visualizing {len(points)} points.")
    o3d.visualization.draw_geometries([o3d_cloud], window_name=file_path)


file_path = "../../DataSet/raw_data/Bench/9_10_2025_BenchB823d_segmented.laz"

visualize_laz(file_path)