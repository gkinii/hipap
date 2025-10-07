import torch
from torch.utils.data import DataLoader

from .model.model import HumanRobotInteractionTransformer, DummyModel
from .model.model_params import ModelParams
from .dataset.synthetic.torch_dataset import TrajectoryPredictionDataset
from .model.vae import VAE
from .model.encoder import HumanRobotInteractionEncoder
from .utils.training import train

# === PATHS ===
DATA_ROOT = "scripts/data/synthetic"
TRAIN_PATH = f"{DATA_ROOT}/train2"
VAL_PATH = f"{DATA_ROOT}/val"
TEST_PATH = f"{DATA_ROOT}/test"
CKPT_DIR = "/home/gkini/Human-Traj-Prediction/scripts/checkpoints"

# === MODEL PARAMS ===
params = ModelParams()
seq_len = params.pred_len
pred_len = params.pred_len

# === DATASETS ===
train_dataset = TrajectoryPredictionDataset(
    data_path=TRAIN_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)
val_dataset = TrajectoryPredictionDataset(
    data_path=VAL_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)
test_dataset = TrajectoryPredictionDataset(
    data_path=TEST_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)

# === DATALOADERS ===
batch_size = 32
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DummyModel(params=params).to(device)

# === TRAIN ===
history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    num_epochs=15,
    patience=5,
    learning_rate=1e-4,
    weight_decay=1e-5,
    ckpt_dir=CKPT_DIR,
    metric='ade',
)

print("Done. Last epoch losses:", {k: v[-1] for k, v in history.items() if len(v) > 0})
