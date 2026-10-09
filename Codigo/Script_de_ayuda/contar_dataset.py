# Counts the number of images/samples for each class. 
# i.e., how many bins, how many benches, how many garbages...

import os
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "../../DataSet/raw_data"
object_types = ["Bathe", "Bench", "Bin", "Block", "ElectricityBox", "Fountain", "Garbage", "MailBox", "Pivot", "Plant", "Sewer", "Stairs", "Stone", "Structure", "WoodBox"]

object_counts = {}

for obj_type in object_types:
    obj_path = dataset_path + "/" + obj_type
    if os.path.exists(obj_path):
        # Count files ending in .laz
        files = []
        for file in os.listdir(obj_path):
            if file.endswith(".laz"):
                files.append(file)
        object_counts[obj_type] = len(files)
    else:
        print(f"Directory does not exist: {obj_type}")
        object_counts[obj_type] = 0

print("Number of objects in the Dataset:")
print(object_counts)

# Generate chart
plt.bar(object_counts.keys(), object_counts.values(), color=['blue', 'green', 'red', 'orange'])
plt.title("Class Distribution in the Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.savefig("distribucion_clases.png") # Save it as .png
plt.show()