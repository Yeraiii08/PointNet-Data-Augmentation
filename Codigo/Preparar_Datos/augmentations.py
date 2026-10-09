import numpy as np

class Augmentor:

    def __init__(self):
        # Name mapping to specify which technique to apply.
        # Key is the technique name, value is the corresponding function.
        self.function_map = {
            'rotate_z': self.rotate_point_cloud_z,
            'jitter': self.jitter_point_cloud,
            'scale': self.random_scale_point_cloud,
            'dropout': self.random_dropout
        }

    @staticmethod
    def rotate_point_cloud_z(batch_data):
        """
        Random rotation around the Z axis (vertical).
        
        Source: https://github.com/charlesq34/pointnet/blob/master/provider.py - def rotate_point_cloud
        The original implementation rotates on the Y axis, we rotate on the Z axis.
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        
        # Z-axis rotation matrix (different from the original GitHub code)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        
        rotated_data = np.dot(batch_data.reshape((-1, 3)), rotation_matrix)
        return rotated_data
    
    @staticmethod
    def jitter_point_cloud(batch_data, sigma=0.01, clip=0.04):
        """
        Adds Gaussian noise (Jittering) to each point individually. Simulates real LiDAR sensor noise.
        
        Source: https://github.com/charlesq34/pointnet/blob/master/provider.py - def jitter_point_cloud
        """
        N, C = batch_data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        jittered_data += batch_data
        return jittered_data

    @staticmethod
    def random_scale_point_cloud(batch_data, scale_low=0.5, scale_high=1.5):
        """
        Scales the object, making it slightly larger or smaller.
        """
        scale = np.random.uniform(scale_low, scale_high)
        scaled_data = batch_data * scale
        return scaled_data

    @staticmethod
    def random_dropout(batch_data, dropout_ratio=0.7): # 1/8 of the points by default
        """
        Randomly removes points from the point cloud.
        To achieve this without changing tensor shapes, we assign dropped points to the same coordinates.
        """
        N, C = batch_data.shape
        
        # Choose random indices based on dropout_ratio
        dropout_indices = np.random.choice(N, int(N * dropout_ratio), replace=False)
        
        # Make a copy to avoid modifying the original batch_data
        dropped_data = batch_data.copy()
        
        # Assign all dropped points the exact same coordinates as the first point.
        # This way, PointNet will detect it as a single point, effectively ignoring the rest.
        # We cannot simply delete them because PointNet requires a fixed number of points (e.g., 2048).
        dropped_data[dropout_indices] = dropped_data[0] 
        
        return dropped_data

    def apply_all(self, point_cloud, techniques_list):
        """
        Applies all requested techniques from the list.
        
        Args:
            point_cloud: Original point cloud
            techniques_list: List of techniques to apply (matching keys in function_map)
        
        Returns:
            np.array: Transformed point cloud.
        """
        # Work on a copy to avoid altering the original array outside the class
        pc = point_cloud.copy()
        
        if not techniques_list:
            return pc

        for tech_name in techniques_list:
            if tech_name in self.function_map:
                # Call the corresponding function from the dictionary
                pc = self.function_map[tech_name](pc)
            else:
                print(f"The technique '{tech_name}' does not exist or is misspelled.")
        
        return pc