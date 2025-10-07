from .model.preprocess import GridPreprocessLayer
from .utils.visualize import visualize
from .utils.training import test
from .dataset.torch_dataset import TrajectoryPredictionDataset
from torch.utils.data import DataLoader
import torch
from .model.model_params import ModelParams
from .model.model import HumanRobotInteractionTransformer
from .model.vae import VAE
import os
VAL_PATH = "/home/gkini/Human-Traj-Prediction/HumanTrajectoryPredictionDataset/data/test"
CKPT_DIR = "/home/gkini/Human-Traj-Prediction/scripts/checkpoints"

params = ModelParams()

seq_len = params.pred_len
pred_len = params.pred_len

test_dataset = TrajectoryPredictionDataset(
    data_path=VAL_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)

batch_size = 32
num_workers = 4
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers, 
    drop_last=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HumanRobotInteractionTransformer(params=params).to(device)

# Get all batches
all_past_seq, all_target_seq, all_preds = test(
    model, 
    test_loader, 
    device, 
    description="Test", 
    ckpt_path=f"{CKPT_DIR}/checkpoint.pth"
)

# Create grid preprocessing layer
grid_layer = GridPreprocessLayer(params)

# Process and visualize each batch
output_base_dir = "trajectory_plots"
os.makedirs(output_base_dir, exist_ok=True)

for batch_idx, (past_seq, target_seq, preds) in enumerate(zip(all_past_seq, all_target_seq, all_preds)):
    print(f"Processing batch {batch_idx + 1}/{len(all_past_seq)}")
    
    # Process the grid for this batch
    processed_grid = grid_layer(past_seq)
    coord_grid = processed_grid['scene/coord'] * 6.425  # [batch_size, 2, 256, 256]
    map_image = processed_grid['scene/grid']  # [batch_size, 1, 256, 256]
    
    # Create a subdirectory for this batch
    batch_output_dir = os.path.join(output_base_dir, f"batch_{batch_idx:04d}")
    
    # Visualize this batch
    visualize(
        coord_grid=coord_grid,
        target_seq=target_seq['prediction_pos'],  # Tensor [batch_size, 1, 10, 2]
        predicted_seq=preds,  # Tensor [batch_size, 1, 10, 2]
        past_seq=past_seq['human_pos'],  # Tensor [batch_size, 1, 10, 2]
        output_dir=batch_output_dir,
        map_image=map_image  # use map image as background
    )

print(f"Visualization complete! All batches saved in '{output_base_dir}'")
