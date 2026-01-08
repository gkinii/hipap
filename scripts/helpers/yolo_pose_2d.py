from ultralytics import YOLO

model = YOLO('yolo11l-pose.pt')
results = model.predict('/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/cell_real/images/infra', save=True, conf=0.5)