import numpy as np
import open3d as o3d
import sys
import os

# Import the augmentations.py file to use its class
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..', 'Preparar_Datos') 
sys.path.append(parent_dir)
from augmentations import Augmentor

# Point cloud to apply transformations to
file_path = "../../DataSet/processed_data/Bench/7_12_2025_BenchB036a_Segmented.npy"
# Techniques we want to apply
techniques = ['jitter']

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    sys.exit()

original_data = np.load(file_path)
augmentor = Augmentor()

# Visualization. Original in gray, transformed alongside in red.
def visualize_comparison(original, transformed, techniques_list):
    title = f"Techniques: {techniques_list}"
    
    # 1. Original Point Cloud (Gray)
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original)
    pcd_original.paint_uniform_color([0.5, 0.5, 0.5]) # Gray
    
    # 2. Transformed Point Cloud (Red)
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(transformed)
    pcd_trans.paint_uniform_color([1, 0, 0]) # Red
    
    # 3. Move the transformed cloud to the right to see them side by side
    pcd_trans.translate((1.5, 0, 0)) 
    
    # Draw coordinate frame to visualize axes better
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd_original, pcd_trans, axes], window_name=title)


mod_data = augmentor.apply_all(original_data, techniques)
visualize_comparison(original_data, mod_data, techniques)