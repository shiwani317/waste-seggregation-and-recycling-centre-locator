import os
import shutil
from sklearn.model_selection import train_test_split

# Paths

DATASET_DIR = "../dataset/garbage_classification/Garbage classification/Garbage classification"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Create train and val folders if not exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Get all class folders except train/val
classes = [d for d in os.listdir(DATASET_DIR)
           if os.path.isdir(os.path.join(DATASET_DIR, d)) and d not in ['train', 'val']]

for cls in classes:
    class_dir = os.path.join(DATASET_DIR, cls)
    images = os.listdir(class_dir)
    
    if len(images) < 2:
        print(f"⚠️ Skipping '{cls}' because it has only {len(images)} image(s).")
        continue
    
    # Split 80% train / 20% validation
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Create class subfolders
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    
    # Copy images
    for img in train_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(TRAIN_DIR, cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(VAL_DIR, cls, img))
    
    print(f"✅ {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

print("\n🎉 Dataset split completed successfully!")
