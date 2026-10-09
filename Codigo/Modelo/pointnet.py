import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes=4):
        super(PointNet, self).__init__()
        
        # --- PART 1: Feature Extraction ---
        # The goal is to map 3 coordinates (x, y, z) into 1024 complex features.
        # We use Conv1d because we treat the points as a sequence/set.
        
        # Layer 1: From 3 channels (x, y, z) to 64 features
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64) # Normaliza para aprender más rápido

        # Layer 2: From 64 to 128
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        # Layer 3: From 128 to 1024
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)


        # --- PART 2: Classifier ---
        # Once we have the global feature vector of the object (1024 numbers), we classify it.
        # This maps the 1024 features to num_classes (the number of object categories).
        
        # Layer 4: From 1024 to 512
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        # Layer 5: From 512 to 256
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Layer 6: Dropout layer (randomly drops neurons to prevent memorization/overfitting)
        self.dropout = nn.Dropout(p=0.3)

        # Final Layer: From 256 to "num_classes"
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Input x shape: (Batch_Size, 3, Num_Points) -> E.g., (32, 3, 2048)
        """
        
        # 1. Point-wise feature extraction (Enrichment)
        x = F.relu(self.bn1(self.conv1(x))) # Output: (Batch, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # Output: (Batch, 128, N)
        x = F.relu(self.bn3(self.conv3(x))) # Output: (Batch, 1024, N)

        # 2. MAX POOLING
        # We compute the max value across all points.
        # This creates a global signature and ensures permutation invariance (point order doesn't matter).
        x = torch.max(x, 2, keepdim=True)[0]
        
        # Flatten the result to feed it into the classifier
        x = x.view(-1, 1024) # Output: (Batch, 1024)

        # 3. Classification
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    # SMALL TEST TO VERIFY FUNCTIONALITY
    simulated_data = torch.rand(2, 3, 2048) 
    print("Input data shape:", simulated_data.shape)
    
    # Initialize the model
    model = PointNet(num_classes=4)
    print("Model initialized successfully.")
    
    # Forward pass
    output = model(simulated_data)
    print("Output shape (Probabilities for 4 classes):", output.shape)
    print(output)