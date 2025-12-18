import pandas as pd
import os

# Path to dataset
data_path = r"C:\Users\MBU\Desktop\ml project\carcinoscope\dataset"

# Load metadata
metadata = pd.read_csv(os.path.join(data_path, "HAM10000_metadata.csv"))
print("✅ Metadata loaded successfully!")
print(metadata.head())

# Count total rows
print(f"\nTotal records: {len(metadata)}")
