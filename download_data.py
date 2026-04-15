import os
from huggingface_hub import snapshot_download
import argparse
import os


# ============================================================
# WHAT TO EDIT FIRST
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, required=True, help='Output directory for download')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Hugging Face dataset repo
REPO_ID = "jacoblin/cloud-stereo"

# Only download lidar dataset subfolder
SUBFOLDER = "synthetic_dataset"
# ============================================================
# DOWNLOAD
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting download from Hugging Face...")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=OUTPUT_DIR,
    token=HF_TOKEN,
)

print("✅ Download complete.")
print(f"Saved to: {OUTPUT_DIR}")