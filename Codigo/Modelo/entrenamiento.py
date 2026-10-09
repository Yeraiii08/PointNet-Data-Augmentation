import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter # Imported to count classes

# --- 1. PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "DataSet", "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "ResultadosEntrenamiento", "modelo_entrenado")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.append(os.path.join(BASE_DIR, "..", "Modelo"))
sys.path.append(os.path.join(BASE_DIR, "..", "Preparar_Datos"))

from pointnet import PointNet
from augmentations import Augmentor

# --- 2. TRAINING CONFIGURATION ---
EXPERIMENT_NAME = "baseline_model"
AUGMENTATION_TECHNIQUES = [] 
EPOCHS = 20       
BATCH_SIZE = 32       
LEARNING_RATE = 0.001 

# --- 3. DATA LOADER CLASS ---
class LidarDataset(Dataset):
    def __init__(self, root_dir, valid_files, class_map, augment=False, augmentation_list=[]):
        self.files = valid_files
        self.class_map = class_map
        self.augment = augment
        self.augmentor = Augmentor()
        self.augmentation_list = augmentation_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Read file
        path, label_str = self.files[idx]
        point_cloud = np.load(path) 

        # 2. Apply Augmentation 
        if self.augment and self.augmentation_list:
            point_cloud = self.augmentor.apply_all(point_cloud, self.augmentation_list)

        # 3. Prepare for PyTorch
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32).transpose(0, 1)
        label = torch.tensor(self.class_map[label_str], dtype=torch.long)
        
        return point_cloud, label, path

# --- 4. DATA PREPARATION ---
def prepare_data(augmentation_list):
    classes = sorted(os.listdir(DATASET_DIR)) 
    class_map = {name: i for i, name in enumerate(classes)} 
    print(f"Detected classes ({len(classes)}): {class_map}")

    files_per_class = {c: [] for c in classes}

    # 1. Load all files organized by class
    for cls in classes:
        cls_path = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            for f in files:
                files_per_class[cls].append((os.path.join(cls_path, f), cls))
    
    train_files = []
    test_files = []

    print("\n--- Original Distribution and Split ---")
    
    # 2. Train / Test Split
    for cls, files in files_per_class.items():
        np.random.shuffle(files)
        split_idx = int(len(files) * 0.8)
        
        if split_idx == len(files) and len(files) > 1:
            split_idx = len(files) - 1
            
        class_train = files[:split_idx]
        class_test = files[split_idx:]
        
        train_files.extend(class_train)
        test_files.extend(class_test)
        
        print(f"Class '{cls}': Total {len(files)} -> Train: {len(class_train)}, Test: {len(class_test)}")

    # 3. Balance the Dataset (Oversampling for minority classes)
    final_train_files = train_files

    if len(augmentation_list) > 0:
        
        train_counts = Counter([label for path, label in train_files])
        max_count = max(train_counts.values())

        train_files_balanced = []
        
        for cls in classes:
            current_class_files = [x for x in train_files if x[1] == cls]
            
            if len(current_class_files) == 0:
                continue

            current_count = len(current_class_files)
            chosen_indices = np.random.choice(current_count, max_count, replace=True)
            
            for idx in chosen_indices:
                train_files_balanced.append(current_class_files[idx])
        
        # Update the final list with the balanced one
        final_train_files = train_files_balanced
        np.random.shuffle(final_train_files)

    
    return final_train_files, test_files, class_map, len(classes)

# --- 5. TRAINING LOOP ---
def main():
    # 1. Prepare data (Now returns balanced data if augmentation is enabled)
    train_files, test_files, class_map, num_classes = prepare_data(AUGMENTATION_TECHNIQUES)
    
    idx_to_class = {v: k for k, v in class_map.items()}

    # By oversampling minority classes, __getitem__ will be called multiple times for the same file.
    # However, the Augmentor will generate a different rotation/noise each time (due to randomness).
    train_ds = LidarDataset(DATASET_DIR, train_files, class_map, augment=True, augmentation_list=AUGMENTATION_TECHNIQUES)
    test_ds = LidarDataset(DATASET_DIR, test_files, class_map, augment=False) 

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load the Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PointNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss() 

    save_path = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}.pth")
    best_accuracy = 0.0

    # 3. Train
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train() 
        running_loss = 0.0
        
        for points, labels, _ in train_loader:
            points, labels = points.to(device), labels.to(device)
            
            optimizer.zero_grad()       
            outputs = model(points)     
            loss = criterion(outputs, labels) 
            loss.backward()             
            optimizer.step()            
            
            running_loss += loss.item()

        # 4. Validate per epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(): 
            for points, labels, _ in test_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} -> Loss: {running_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            
    print("\n--- Training Finished ---")

    # --- 6. FAILURE ANALYSIS ---
    print(f"\nBest model accuracy: {best_accuracy:.2f}%\n")
    
    best_model = PointNet(num_classes=num_classes).to(device)
    best_model.load_state_dict(torch.load(save_path))
    best_model.eval()

    detected_errors = []

    with torch.no_grad():
        for points, labels, paths in test_loader:
            points, labels = points.to(device), labels.to(device)
            outputs = best_model(points)
            _, predicted = torch.max(outputs.data, 1)
            
            pred_list = predicted.cpu().numpy()
            label_list = labels.cpu().numpy()
            
            for i in range(len(points)):
                if pred_list[i] != label_list[i]:
                    file_name = os.path.basename(paths[i])
                    real_class = idx_to_class[label_list[i]]
                    confused_class = idx_to_class[pred_list[i]]
                    
                    detected_errors.append({
                        "file": file_name,
                        "real": real_class,
                        "predicted": confused_class
                    })
    
    if len(detected_errors) == 0:
        print("No errors found on the test set.")
    else:
        print(f"{'FILE':<40} | {'GROUND TRUTH':<15} | {'PREDICTED':<15}")
        print("-" * 76)
        for error in detected_errors:
            print(f"{error['file']:<40} | {error['real']:<15} | {error['predicted']:<15}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()