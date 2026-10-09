import numpy as np
import open3d as o3d
import os

test_file = "../../DataSet/processed_data/Bench/1_11_2025_BenchB001c_segmented.npy"

if os.path.exists(test_file):
    data = np.load(test_file)

    # Visualize
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)
    
    # Set a specific color for better visibility
    cloud.paint_uniform_color([0, 0.65, 0.92]) 

    # Create reference sphere
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    wire_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_mesh)
    wire_sphere.paint_uniform_color([1, 0, 0])
    
    o3d.visualization.draw_geometries([cloud, wire_sphere], window_name="Normalization Verification")
else:
    print("File does not exist. Run preprocessing first.")