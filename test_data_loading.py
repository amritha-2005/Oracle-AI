import os
import xarray as xr
from pathlib import Path

# Define root data directory relative to backend
DATA_DIR = Path("ml_pipeline/data/raw")

def test_load_datasets():
    print("🔍 Starting dataset loading test...")
    
    # Check ERA5
    era5_dir = DATA_DIR / "era5"
    if era5_dir.exists():
        era5_files = list(era5_dir.glob("*.nc"))
        print(f"✅ Found {len(era5_files)} NetCDF (.nc) files in ERA5 directory.")
        if len(era5_files) > 0:
            # Try opening the first ERA5 file to verify readability
            try:
                sample_ds = xr.open_dataset(era5_files[0])
                print(f"   Sample ERA5 file loaded successfully: {era5_files[0].name}")
                print(f"   Dimensions: {list(sample_ds.dims.keys())}")
                sample_ds.close()
            except Exception as e:
                print(f"   ⚠️ Could not open ERA5 file: {e}")
    else:
        print("❌ ERA5 directory not found at ml_pipeline/data/raw/era5")

    # Check IMD
    imd_dir = DATA_DIR / "imd"
    if imd_dir.exists():
        imd_files = [f for f in imd_dir.iterdir() if f.suffix in ['.nc', '.grd', '.bin']]
        print(f"✅ Found {len(imd_files)} files in IMD directory.")
        if len(imd_files) > 0:
            for imd_file in imd_files:
                if imd_file.suffix == '.nc':
                    try:
                        sample_imd = xr.open_dataset(imd_file)
                        print(f"   Sample IMD NetCDF loaded successfully: {imd_file.name}")
                        sample_imd.close()
                    except Exception as e:
                        print(f"   ⚠️ Could not open IMD file {imd_file.name}: {e}")
    else:
        print("❌ IMD directory not found at ml_pipeline/data/raw/imd")

    print("✅ Data loading test script finished execution!")

if __name__ == "__main__":
    test_load_datasets()