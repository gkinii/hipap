import torch
from torch.utils.data import DataLoader, random_split

from .model.model import HumanRobotInteractionTransformer
from .model.model_params import ModelParams
from .dataset.torch_dataset import TrajectoryPredictionDataset
from .model.vae import VAE
from .utils.training import train

TRAIN_PATH = "/home/gkini/Human-Traj-Prediction/HumanTrajectoryPredictionDataset/data/train_val"
VAL_PATH = "/home/gkini/Human-Traj-Prediction/HumanTrajectoryPredictionDataset/data/test"
CKPT_DIR = "/home/gkini/Human-Traj-Prediction/scripts/checkpoints"

params = ModelParams()

seq_len  = params.pred_len
pred_len = params.pred_len

train_dataset = TrajectoryPredictionDataset(
    data_path=TRAIN_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)

test_dataset = TrajectoryPredictionDataset(
    data_path=VAL_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)

test_ratio = 0.4
test_size  = max(1, int(len(test_dataset) * test_ratio))
val_size = max(1, len(test_dataset) - test_size)
val_dataset, test_dataset = random_split(
    test_dataset, [val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 32
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = HumanRobotInteractionTransformer(params=params).to(device)

history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    num_epochs=8,          
    patience=3,
    learning_rate=2e-4,
    weight_decay=2e-5,
    ckpt_dir=CKPT_DIR,
    metric='mse',
    beta_end=0.01
)

print("Done. Last epoch losses:", {k: v[-1] for k, v in history.items() if len(v) > 0})

