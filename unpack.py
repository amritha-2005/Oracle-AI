import zipfile
import os

# Paths to your raw folders
raw_dirs = ['ml_pipeline/data/raw/era5', 'ml_pipeline/data/raw/imd']

for directory in raw_dirs:
    if os.path.exists(directory):
        print(f"Scanning {directory}...")
        for file in os.listdir(directory):
            if file.endswith('.zip'):
                zip_path = os.path.join(directory, file)
                print(f"Extracting {file}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(directory)
                print(f"Successfully extracted {file}")
                # Optional: remove zip file after extraction to save space
                # os.remove(zip_path)

print("🎉 All zip files have been extracted successfully!")