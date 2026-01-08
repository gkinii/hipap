import torch
from torch.utils.data import DataLoader
import os
import time
from .model.model_params import ModelParams
from .model.cvae import TransformerCVAE
from .dataset.synthetic.torch_dataset_task_synth import TrajectoryPredictionDataset
from .dataset.torch_dataset_station import StationTrajectoryPredictionDataset
from .utils.cvae_training import validate

# === PATHS ===
DATA_ROOT = "scripts/data/synthetic"

# Select which test dataset to use
TEST_PATH = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/all_stations"
# TEST_PATH = f"{DATA_ROOT}/400_test"
# TEST_PATH = f"{DATA_ROOT}/256_cell_new"

CKPT_DIR = "/home/gkini/Human-Traj-Prediction/scripts/checkpoints"
CKPT_PATH = f"{CKPT_DIR}/model_epoch_005.pth"  # Update this to your checkpoint
OUTPUT_BASE_DIR = "trajectory_plots"

# === MODEL PARAMS ===
params = ModelParams()
seq_len = params.seq_len
pred_len = params.pred_len

# === DATASET ===
# Use StationTrajectoryPredictionDataset for Oxford station data
test_dataset = StationTrajectoryPredictionDataset(
    data_path=TEST_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
    stride=1
)

# Or use TrajectoryPredictionDataset for synthetic data
# test_dataset = TrajectoryPredictionDataset(
#     data_path=TEST_PATH,
#     seq_len=seq_len,
#     pred_len=pred_len,
#     stride=1
# )

# === DATALOADER ===
batch_size = 32
num_workers = 4
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerCVAE(params=params).to(device)

# === LOAD CHECKPOINT ===
if os.path.exists(CKPT_PATH):
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from {CKPT_PATH}")
            if 'epoch' in checkpoint:
                print(f"   Checkpoint epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✅ Loaded checkpoint from {CKPT_PATH}")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print(f"✅ Loaded checkpoint from {CKPT_PATH}")
        else:
            # Assume the dict IS the state dict
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint from {CKPT_PATH}")
    else:
        # Checkpoint is the state dict directly
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint from {CKPT_PATH}")
else:
    print(f"⚠️ No checkpoint found at {CKPT_PATH}, using random weights")

# === INFERENCE WITH VISUALIZATION ===
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"Starting inference on {len(test_dataset)} samples...")
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"{'='*60}\n")

# Start timing
start_time = time.time()

# Run validation (which performs inference)
v_min, v_mean, total_hits = validate(
    model=model,
    dataloader=test_loader,
    device=device,
    description="Inference",
    metric='mse',
    best_weight=1.0,
    worst_weight=0.0,
    visualize_traj=True,
)

# End timing
end_time = time.time()
total_time = end_time - start_time
total_samples = len(test_dataset)
time_per_sample = total_time / total_samples

# === RESULTS ===
print(f"\n{'='*60}")
print(f"Inference complete!")
print(f"Best-of-N Loss (v_min): {v_min:.4f}")
print(f"Mean Loss (v_mean): {v_mean:.4f}")
print(f"Total Obstacle Hits: {total_hits}")
print(f"Total Samples: {total_samples}")
print(f"Total Time: {total_time:.2f}s")
print(f"Time per Sample: {time_per_sample*1000:.2f}ms ({1/time_per_sample:.1f} samples/sec)")
print(f"Visualizations saved in: '{OUTPUT_BASE_DIR}'")
print(f"{'='*60}")