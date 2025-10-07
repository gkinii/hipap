import mujoco, mujoco.viewer
import numpy as np, time
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import heapq
from collections import defaultdict

XML_PATH = "scripts/mujoco/models/cell/cell.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

BATTERY_SITES = ["battery_c0", "battery_c1", "battery_c2", "battery_c3"]
TOOL_SITES = ["tool_c0", "tool_c1", "tool_c2", "tool_c3"]
OUTLINE_SITES = ["outline_c0", "outline_c1", "outline_c2",
                "outline_c4", "outline_c5", "outline_c6"]
AGV_SITES = ["agv_c0", "agv_c1", "agv_c2", "agv_c3"]
ROBOT_SITES = ["robot_c0", "robot_c1", "robot_c2", "robot_c3"]

# Path planning parameters
START_GOAL = np.array([0.0, 0.0])
END_GOAL = np.array([-3.0, 0.0])

# Separate fillable and outline sites
STATIC_SITE_GROUPS = [BATTERY_SITES, TOOL_SITES]
OUTLINE_SITE_GROUPS = [OUTLINE_SITES]
MOVING_SITE_GROUPS = [AGV_SITES, ROBOT_SITES]

dx_dy = 0.1
FILL_POLYS = True  # Now only affects fillable polygons, not outline

def world_xy(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    x, y, *_ = data.site_xpos[sid]
    return np.array([x, y])

def fetch_polygons(site_groups):
    """Return [np.ndarray(N_i,2), …] for each group in *current* data frame."""
    return [np.stack([world_xy(model, data, n) for n in names])
            for names in site_groups]

def world_to_idx(xy, origin):
    rc = ((xy - origin) / dx_dy).astype(int)
    return rc[:,1], rc[:,0] # row (y), col (x)

def idx_to_world(r, c, origin):
    """Convert grid indices back to world coordinates."""
    y = r * dx_dy + origin[1]
    x = c * dx_dy + origin[0]
    return np.array([x, y])

def rasterise_line(p, q):
    length = np.linalg.norm(q - p)
    n = max(int(np.ceil(length / (dx_dy * 0.5))), 1)
    return p + (q - p) * np.linspace(0, 1, n, endpoint=True)[:, None]

def draw_polygon(poly, grid, origin):
    """
    Draw polygon outline on a 2D grid.
    poly: (N, 2) array-like of vertex coords (x, y) in world units
    grid: 2D numpy array (ny, nx) modified in place
    origin: whatever your world_to_idx expects as origin/reference
    """
    poly = np.asarray(poly, dtype=float)
    ny, nx = grid.shape

    # Iterate edges as (v_i -> v_{i+1}), wrapping last to first, no pairwise needed
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        pts = list(rasterise_line(a, b))              # iterable of points along the edge
        if not pts:
            continue
        pts = np.asarray(pts, dtype=float)            # (M, 2)

        # Convert all points at once to grid indices
        r, c = world_to_idx(pts, origin)              # expect arrays shaped (M,) each
        r = np.asarray(r).astype(int, copy=False)
        c = np.asarray(c).astype(int, copy=False)

        # Keep only in-bounds indices
        m = (0 <= r) & (r < ny) & (0 <= c) & (c < nx)
        if np.any(m):
            grid[r[m], c[m]] = 1

def fill_polygons(grid, polygons, origin):
    """Fill the interior of specified polygons."""
    if not FILL_POLYS:
        return
    
    # Create a mask of just the polygons we want to fill
    fill_mask = np.zeros_like(grid)
    for poly in polygons:
        draw_polygon(poly, fill_mask, origin)
    
    filled_mask = binary_fill_holes(fill_mask)
    grid[:,:] = np.logical_or(grid, filled_mask).astype(np.uint8)

def print_object_bounds(site_groups, group_names, origin):
    """Print bounding positions for each object group"""
    print("\n" + "="*80)
    print("OBJECT BOUNDING POSITIONS (World Coordinates)")
    print("="*80)
    
    for group_sites, group_name in zip(site_groups, group_names):
        polygons = fetch_polygons([group_sites])
        
        for i, poly in enumerate(polygons):
            print(f"\n{group_name.upper()} #{i}")
            print("-" * 40)
            
            # Get all corner positions
            print("Corner positions:")
            for j, corner in enumerate(poly):
                print(f"  Corner {j}: ({corner[0]:6.3f}, {corner[1]:6.3f})")
            
            # Calculate bounding box
            min_x, min_y = poly.min(axis=0)
            max_x, max_y = poly.max(axis=0)
            print(f"Bounding box:")
            print(f"  Min: ({min_x:6.3f}, {min_y:6.3f})")
            print(f"  Max: ({max_x:6.3f}, {max_y:6.3f})")
            print(f"  Size: {max_x-min_x:6.3f} x {max_y-min_y:6.3f}")
            
            # Calculate center
            center_x, center_y = poly.mean(axis=0)
            print(f"  Center: ({center_x:6.3f}, {center_y:6.3f})")
            
            # Grid coordinates
            r_coords, c_coords = world_to_idx(poly, origin)
            print(f"Grid indices (row, col):")
            for j, (r, c) in enumerate(zip(r_coords, c_coords)):
                print(f"  Corner {j}: ({r:3d}, {c:3d})")

def heuristic(a, b):
    """Euclidean distance heuristic for A*"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(node, grid_shape):
    """Get valid 8-connected neighbors"""
    r, c = node
    ny, nx = grid_shape
    neighbors = []
    
    # 8-connected neighbors
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < ny and 0 <= nc < nx:
                neighbors.append((nr, nc))
    
    return neighbors

def astar_path(grid, start, goal):
    """A* pathfinding algorithm"""
    ny, nx = grid.shape
    
    # Priority queue: (f_score, node)
    open_set = [(0, start)]
    came_from = {}
    
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    
    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = heuristic(start, goal)
    
    open_set_hash = {start}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        open_set_hash.remove(current)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in get_neighbors(current, grid.shape):
            # Skip if neighbor is an obstacle
            if grid[neighbor] == 1:
                continue
            
            # Distance is sqrt(2) for diagonal, 1 for orthogonal
            dr, dc = abs(neighbor[0] - current[0]), abs(neighbor[1] - current[1])
            if dr + dc == 2:  # diagonal
                dist = np.sqrt(2)
            else:  # orthogonal
                dist = 1.0
            
            tentative_g_score = g_score[current] + dist
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
    
    return None  # No path found

def generate_multiple_paths(grid, start, goal, num_paths=5):
    """Generate multiple paths by adding slight randomness to the heuristic"""
    paths = []
    original_grid = grid.copy()
    
    for i in range(num_paths):
        # Create a slightly modified grid by adding small random costs
        modified_grid = original_grid.copy().astype(float)
        
        # Add small random noise to free spaces (not obstacles)
        noise = np.random.normal(0, 0.1, modified_grid.shape)
        noise[original_grid == 1] = 0  # Don't modify obstacles
        modified_grid += noise
        
        # Convert back to binary for pathfinding
        binary_grid = (modified_grid > 0.5).astype(int)
        
        path = astar_path(binary_grid, start, goal)
        if path and path not in paths:
            paths.append(path)
        
        # If we have enough diverse paths, break
        if len(paths) >= num_paths:
            break
    
    return paths

# Get all polygons to determine grid bounds
all_initial_polys = fetch_polygons(STATIC_SITE_GROUPS + OUTLINE_SITE_GROUPS + MOVING_SITE_GROUPS)
xy_all = np.concatenate(all_initial_polys)

# Include start and end goals in bounds calculation
goals = np.vstack([START_GOAL, END_GOAL])
xy_all = np.vstack([xy_all, goals])

xmin, ymin = xy_all.min(axis=0) - dx_dy
xmax, ymax = xy_all.max(axis=0) + dx_dy

nx = int(np.ceil((xmax - xmin) / dx_dy))
ny = int(np.ceil((ymax - ymin) / dx_dy))
origin = np.array([xmin, ymin])

# Print object bounds information
all_site_groups = STATIC_SITE_GROUPS + OUTLINE_SITE_GROUPS + MOVING_SITE_GROUPS
group_names = ["Battery", "Tool", "Outline", "AGV", "Robot"]
print_object_bounds(all_site_groups, group_names, origin)

# Convert start and end goals to grid coordinates
start_r, start_c = world_to_idx(START_GOAL[None, :], origin)
end_r, end_c = world_to_idx(END_GOAL[None, :], origin)
start_grid = (start_r[0], start_c[0])
end_grid = (end_r[0], end_c[0])

print(f"\n" + "="*80)
print("PATH PLANNING INFORMATION")
print("="*80)
print(f"Start Goal: {START_GOAL} -> Grid: {start_grid}")
print(f"End Goal: {END_GOAL} -> Grid: {end_grid}")
print(f"Grid bounds: X=[{xmin:.2f}, {xmax:.2f}], Y=[{ymin:.2f}, {ymax:.2f}] meters")
print(f"Grid resolution: {dx_dy} meters per cell")
print(f"Grid size: {nx} x {ny} cells")

# Create static grid
static_grid = np.zeros((ny, nx), dtype=np.uint8)

# Draw all static polygons (outlines only)
for poly in fetch_polygons(STATIC_SITE_GROUPS + OUTLINE_SITE_GROUPS):
    draw_polygon(poly, static_grid, origin)

# Fill only the non-outline polygons
fill_polygons(static_grid, fetch_polygons(STATIC_SITE_GROUPS), origin)

# Generate initial paths on static grid
initial_paths = generate_multiple_paths(static_grid, start_grid, end_grid, num_paths=3)

# Set up visualization with world coordinates
fig, ax = plt.subplots(figsize=(12, 8))

# Set extent to map grid to world coordinates
# extent = [left, right, bottom, top] in world coordinates
extent = [xmin, xmax, ymin, ymax]
img = ax.imshow(static_grid, origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.7, extent=extent)

# Plot start and end goals in world coordinates
ax.plot(START_GOAL[0], START_GOAL[1], 'go', markersize=10, label='Start (0,0)')
ax.plot(END_GOAL[0], END_GOAL[1], 'ro', markersize=10, label='End (-3,0)')

# Plot initial paths in world coordinates
colors = ['blue', 'cyan', 'magenta', 'yellow', 'orange']
path_lines = []
for i, path in enumerate(initial_paths):
    if path:
        # Convert path from grid indices to world coordinates
        path_world = np.array([idx_to_world(r, c, origin) for r, c in path])
        line, = ax.plot(path_world[:, 0], path_world[:, 1], 
                       color=colors[i % len(colors)], linewidth=2, 
                       alpha=0.8, label=f'Path {i+1}')
        path_lines.append(line)

ax.set_title("Occupancy Grid with Path Planning (World Coordinates)")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.pause(0.01)

print(f"\nFound {len(initial_paths)} initial paths from {START_GOAL} to {END_GOAL}")

# Main simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 20.0
    viewer.cam.azimuth = -90
    viewer.cam.elevation = -80
    viewer.fullscreen = True
    
    frame_count = 0
    
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        
        # Start with static grid
        current_grid = static_grid.copy()
        
        # Add moving objects
        moving_polys = fetch_polygons(MOVING_SITE_GROUPS)
        for poly in moving_polys:
            draw_polygon(poly, current_grid, origin)
        
        # Recompute paths every 30 frames (for performance)
        if frame_count % 30 == 0:
            current_paths = generate_multiple_paths(current_grid, start_grid, end_grid, num_paths=3)
            
            # Clear previous path lines
            for line in path_lines:
                line.remove()
            path_lines.clear()
            
            # Plot new paths in world coordinates
            for i, path in enumerate(current_paths):
                if path:
                    # Convert path from grid indices to world coordinates
                    path_world = np.array([idx_to_world(r, c, origin) for r, c in path])
                    line, = ax.plot(path_world[:, 0], path_world[:, 1], 
                                   color=colors[i % len(colors)], linewidth=2, 
                                   alpha=0.8, label=f'Path {i+1}')
                    path_lines.append(line)
        
        # Update visualization
        img.set_data(current_grid)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
        viewer.sync()
        frame_count += 1
        
        # Maintain real-time simulation
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)