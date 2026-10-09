import os
import laspy
import numpy as np
from tqdm import tqdm # Progress bar for better visualization


NUM_POINTS = 2048  # Number of points each point cloud will contain
DATASET_RAW = "../../DataSet/raw_data"
DATASET_PROCESSED = "../../DataSet/processed_data"
CLASSES = ["Bathe", "Bench", "Bin", "Block", "ElectricityBox", "Fountain", "Garbage", "MailBox", "Pivot", "Plant", "Sewer", "Stairs", "Stone", "Structure", "WoodBox"]

def normalize_pc(points):
    # 1. Center the point cloud at the origin (0,0,0).
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # 2. Scale the point cloud so it fits inside a unit sphere (radius 1).
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / furthest_distance

    return points

def sample_pc(points, target_num_points):
    """
    If there are more points: Random Sampling without replacement.
    If there are fewer points: Random Padding with replacement.
    """
    current_num_points = points.shape[0]
    
    if current_num_points >= target_num_points:
        # Choose random indices without replacement (do not repeat if we have enough points)
        indices = np.random.choice(current_num_points, target_num_points, replace=False)
    else:
        # Choose random indices with replacement (repeat points to fill the gap)
        indices = np.random.choice(current_num_points, target_num_points, replace=True)
    
    return points[indices]

def process_dataset():
    print(f"Starting preprocessing... Target: {NUM_POINTS} points per object.")
    
    for cls in CLASSES:
        source_path = os.path.join(DATASET_RAW, cls)
        dest_path = os.path.join(DATASET_PROCESSED, cls)
        
        # Create destination folder if it doesn't exist
        os.makedirs(dest_path, exist_ok=True)
        
        files = [f for f in os.listdir(source_path) if f.endswith('.laz')]
        print(f"Processing class '{cls}': {len(files)} files.")
        
        # tqdm displays a progress bar for better UX
        for filename in tqdm(files):
            laz_path = os.path.join(source_path, filename)
            
            try:
                # 1. Read LAZ file
                las = laspy.read(laz_path)
                points = np.vstack((las.x, las.y, las.z)).transpose()
                
                # 2. Normalize: center at (0, 0, 0) and scale
                points = normalize_pc(points)
                
                # 3. Sample: ensure exact number of points
                points = sample_pc(points, NUM_POINTS)
                
                # 4. Save as .npy (Much faster for training later)
                name_without_ext = filename.split('.')[0]
                np.save(os.path.join(dest_path, f"{name_without_ext}.npy"), points)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_dataset()