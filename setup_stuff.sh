set -e

echo "Setting up virtual environment and installing dependencies..."
cd "blacksky-code/cloud_4d"
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

echo "downloading datasets" 
python download_datasets.py --output
echo "Step 1: Pretrain on Terragen Data"
accelerate launch --config_file deepspeed_config.yaml src/train.py \
    --pretrain \
    --stage 1 \
    --lr 1e-4 \
    --data_dir /path/to/terragen/data \
    --volume_dir /path/to/terragen/volumes \
    --steps 20000 \
    --checkpoint_path ./checkpoints/stage1_pretrain


echo "Step 2: Fine-tune on LES Data"
accelerate launch --config_file deepspeed_config.yaml src/train.py \
    --stage 1 \
    --lr 1e-5 \
    --data_dir /path/to/les/data \
    --volume_dir /path/to/les/volumes \
    --stage1_checkpoint ./checkpoints/stage1_pretrain/stage1_model_20000.pth \
    --steps 40000 \
    --checkpoint_path ./checkpoints/stage1_les

echo "Step 3: Train Stage 2 on LES Data"
accelerate launch --config_file deepspeed_config.yaml src/train.py \
    --stage 2 \
    --lr 1e-5 \
    --data_dir /path/to/les/data \
    --volume_dir /path/to/les/volumes \
    --stage1_checkpoint ./checkpoints/stage1_les/stage1_model_40000.pth \
    --steps 30000 \
    --checkpoint_path ./checkpoints/stage2_les
