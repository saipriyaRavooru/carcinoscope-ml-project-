import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === PATH CONFIGURATION ===
base_dir = r"C:\Users\MBU\Desktop\ml project\carcinoscope\dataset"
img_dir_1 = os.path.join(base_dir, "HAM10000_images_part_1")
img_dir_2 = os.path.join(base_dir, "HAM10000_images_part_2")
metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")

# === LOAD METADATA ===
metadata = pd.read_csv(metadata_path)
print("✅ Metadata loaded:", metadata.shape)
print(metadata.head())

# === MAP IMAGE FILE PATHS ===
def get_image_path(image_id):
    path1 = os.path.join(img_dir_1, f"{image_id}.jpg")
    path2 = os.path.join(img_dir_2, f"{image_id}.jpg")
    return path1 if os.path.exists(path1) else path2

metadata["image_path"] = metadata["image_id"].apply(get_image_path)

# === CREATE TRAIN / TEST / VALIDATION SPLITS ===
train_df, temp_df = train_test_split(metadata, test_size=0.3, stratify=metadata["dx"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=42)

print(f"✅ Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

# === CREATE OUTPUT FOLDERS ===
output_dir = os.path.join(base_dir, "split_data")
for folder in ["train", "val", "test"]:
    for label in metadata["dx"].unique():
        os.makedirs(os.path.join(output_dir, folder, label), exist_ok=True)

# === COPY IMAGES INTO THEIR SPLIT FOLDERS ===
def copy_images(df, folder_name):
    for _, row in df.iterrows():
        label_dir = os.path.join(output_dir, folder_name, row.dx)
        dest_path = os.path.join(label_dir, os.path.basename(row.image_path))
        if not os.path.exists(dest_path):
            shutil.copy(row.image_path, dest_path)

print("📦 Copying train images...")
copy_images(train_df, "train")
print("📦 Copying validation images...")
copy_images(val_df, "val")
print("📦 Copying test images...")
copy_images(test_df, "test")

print("\n✅ Dataset split and organized successfully!")
print("➡️ Output folders created at:", output_dir)
