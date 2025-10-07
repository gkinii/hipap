import matplotlib.pyplot as plt
import torch
import os

def visualize(coord_grid, target_seq, predicted_seq, past_seq, output_dir="trajectory_plots", map_image=None):

    # Squeeze agent dimension if present and convert to numpy
    if len(target_seq.shape) == 4 and target_seq.shape[1] == 1:
        target_seq = target_seq.squeeze(1)
    if len(predicted_seq.shape) == 4 and predicted_seq.shape[1] == 1:
        predicted_seq = predicted_seq.squeeze(1)
    if len(past_seq.shape) == 4 and past_seq.shape[1] == 1:
        past_seq = past_seq.squeeze(1)
    
    target_traj = target_seq.cpu().numpy()
    pred_traj = predicted_seq.cpu().numpy()
    past_traj = past_seq.cpu().numpy()
    coord_grid = coord_grid.cpu().numpy()
    
    batch_size, channels, height, width = coord_grid.shape
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for batch_idx in range(batch_size):
        # Extract x, y coordinates for the current batch
        target_x, target_y = target_traj[batch_idx, :, 0], target_traj[batch_idx, :, 1]
        pred_x, pred_y = pred_traj[batch_idx, :, 0], pred_traj[batch_idx, :, 1]
        past_x, past_y = past_traj[batch_idx, :, 0], past_traj[batch_idx, :, 1]
        
        # Create figure
        plt.figure(figsize=(8, 8))
        
        # Use map_image if provided, otherwise use coord_grid
        if map_image is not None:
            map_img = map_image.cpu().numpy()[batch_idx, 0]
            grid_x = coord_grid[batch_idx, 0]
            grid_y = coord_grid[batch_idx, 1] if channels > 1 else grid_x
            extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            plt.imshow(map_img, cmap='gray', alpha=0.5, extent=extent)
        else:
            grid_x = coord_grid[batch_idx, 0]
            grid_y = coord_grid[batch_idx, 1] if channels > 1 else grid_x
            plt.imshow(grid_x, cmap='viridis', alpha=0.3, 
                      extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()))
        
        # Plot trajectories
        plt.plot(past_x, past_y, 'b-', label='Past Trajectory', linewidth=2, marker='o')
        plt.plot(target_x, target_y, 'g-', label='Actual Trajectory', linewidth=2, marker='s')
        plt.plot(pred_x, pred_y, 'r--', label='Predicted Trajectory', linewidth=2, marker='^')
        
        # Add labels and legend
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Trajectory Visualization (Batch {batch_idx})')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'trajectory_batch_{batch_idx}.png')
        plt.savefig(plot_path)
        plt.close()
    
    print(f"Saved {batch_size} plots to {output_dir}")