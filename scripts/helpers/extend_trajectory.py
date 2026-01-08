# extend_trajectory_realistic.py
import pandas as pd
import numpy as np

# ========================= CONFIG =========================
input_csv  = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/256_cell_new/paths_1.csv"
output_csv = "/home/gkini/Human-Traj-Prediction/scripts/data/synthetic/256_cell_new/paths_1_extended_realistic.csv"

goal_x = -1
goal_y = 5.2

n_intermediate = 5                  # ← exactly 3 new points
noise_std = 0.1                    # ← natural human-like deviation (in meters)
np.random.seed(42)                  # ← reproducible realism
# =========================================================

# Load original trajectory
df = pd.read_csv(input_csv)
print(f"Original trajectory: {len(df)} points")
last_x = df['x'].iloc[-1]
last_y = df['y'].iloc[-1]

print(f"Last recorded position: ({last_x:.3f}, {last_y:.3f})")
print(f"Goal (Tools station):   ({goal_x:.3f}, {goal_y:.3f})\n")

# Generate 3 intermediate points along the straight line
t = np.linspace(0, 1, n_intermediate + 2)[1:-1]   # 3 values between 0 and 1 (excluding start/end)
interp_x = (1 - t) * last_x + t * goal_x
interp_y = (1 - t) * last_y + t * goal_y

# Add realistic human-like noise (Gaussian, zero-mean)
noise_x = np.random.normal(0, noise_std, n_intermediate)
noise_y = np.random.normal(0, noise_std, n_intermediate)

final_x = np.append(interp_x + noise_x, goal_x)   # last point = exact goal
final_y = np.append(interp_y + noise_y, goal_y)

# Create new rows
new_rows = pd.DataFrame({
    'x': final_x,
    'y': final_y
})

# Append to original and save
df_extended = pd.concat([df, new_rows], ignore_index=True)
df_extended.to_csv(output_csv, index=False)

print(f"Added {n_intermediate + 1} new points (3 noisy + 1 exact goal)")
print(f"Final trajectory length: {len(df_extended)} points")
print(f"Final position: ({df_extended['x'].iloc[-1]:.3f}, {df_extended['y'].iloc[-1]:.3f}) → Goal reached exactly!")
print(f"Saved to:\n    {output_csv}")