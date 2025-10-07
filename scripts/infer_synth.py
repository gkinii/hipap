from .model.preprocess import GridPreprocessLayer
from .utils.visualize import visualize
from .utils.training import test
from .dataset.synthetic.torch_dataset import TrajectoryPredictionDataset
from torch.utils.data import DataLoader
import torch
from .model.model_params import ModelParams
from .model.model import DummyModel  # or HumanRobotInteractionTransformer
from .model.vae import VAE
import os

# === CONFIG ===
DATA_PATH = "scripts/data/synthetic/demo"  
CKPT_DIR = "/home/gkini/Human-Traj-Prediction/scripts/checkpoints"
CKPT_PATH = f"{CKPT_DIR}/checkpoint.pth" 
OUTPUT_BASE_DIR = "trajectory_plots"

# === PARAMS ===
params = ModelParams()
seq_len = params.pred_len
pred_len = params.pred_len

# === DATASET (single folder) ===
dataset = TrajectoryPredictionDataset(
    data_path=DATA_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)
assert len(dataset) > 0, f"No samples found in {DATA_PATH}"

# === DATALOADER ===
batch_size = 8
num_workers = 4
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,  # keep the last partial batch for visualization
)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DummyModel(params=params).to(device)  # swap to your actual model if needed

# === EVAL ===
all_past_seq, all_target_seq, all_preds = test(
    model=model,
    dataloader=test_loader,
    device=device,
    description="Test",
    ckpt_path=CKPT_PATH,
)

# === GRID & VISUALIZATION ===
grid_layer = GridPreprocessLayer(params)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

for batch_idx, (past_seq, target_seq, preds) in enumerate(zip(all_past_seq, all_target_seq, all_preds)):
    print(f"Processing batch {batch_idx + 1}/{len(all_past_seq)}")

    # Build map/coord grid for this batch
    map_image, coord_grid = grid_layer(past_seq)

    # Per-batch output dir
    batch_output_dir = os.path.join(OUTPUT_BASE_DIR, f"batch_{batch_idx:04d}")

    visualize(
        coord_grid=coord_grid,
        target_seq=target_seq['prediction_pos'],  # [B, 1, pred_len, 2]
        predicted_seq=preds,                      # [B, 1, pred_len, 2]
        past_seq=past_seq['human_pos'],           # [B, 1, seq_len, 2]
        output_dir=batch_output_dir,
        map_image=map_image
    )

print(f"Visualization complete! All batches saved in '{OUTPUT_BASE_DIR}'")
