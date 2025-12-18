import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to your dataset
data_path = r"C:\Users\MBU\Desktop\ml project\carcinoscope\dataset"

# Load metadata CSV
metadata = pd.read_csv(os.path.join(data_path, "HAM10000_metadata.csv"))
print("✅ Metadata loaded successfully!")
print(metadata.head(), "\n")

# Check image folders
img_dir_1 = os.path.join(data_path, "HAM10000_images_part_1")
img_dir_2 = os.path.join(data_path, "HAM10000_images_part_2")

# Combine image paths
metadata["image_path"] = metadata["image_id"].apply(
    lambda x: os.path.join(img_dir_1, f"{x}.jpg")
    if os.path.exists(os.path.join(img_dir_1, f"{x}.jpg"))
    else os.path.join(img_dir_2, f"{x}.jpg")
)

# Verify how many images exist
existing = metadata["image_path"].apply(os.path.exists).sum()
print(f"✅ Found {existing} images out of {len(metadata)} total records.")

# Display a few random samples
sample = metadata.sample(9, random_state=42)

plt.figure(figsize=(10, 10))
for i, row in enumerate(sample.itertuples(), 1):
    img = mpimg.imread(row.image_path)
    plt.subplot(3, 3, i)
    plt.imshow(img)
    plt.title(row.dx)
    plt.axis("off")

plt.suptitle("Sample HAM10000 Images by Diagnosis", fontsize=16)
plt.show()
