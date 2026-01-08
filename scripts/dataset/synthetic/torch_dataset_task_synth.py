# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# import os
# from tqdm import tqdm
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import yaml
# import glob
# import re
# from collections import OrderedDict

# # Added 'stations_pos/mask' and 'robot_pos/mask' to FEATURE_COLS
# FEATURE_COLS = ['x', 'y', 'z', 'agent_type', 'frame_id', 'agent_id', 'task_id', 
#                 'station_1_x', 'station_1_y', 'station_2_x', 'station_2_y', 'pos_mask',
#                 'stations_pos/mask', 'robot_pos/mask']

# class TrajectoryPredictionDataset(Dataset):
#     def __init__(self, data_path: str, seq_len: int, pred_len: int, stride: int):
#         self.data_path = data_path
#         self.seq_len: int = seq_len
#         self.pred_len: int = pred_len
#         self.max_human_agents: int = 1
#         self.max_robot_agents: int = 2
#         self.max_stations: int = 2
#         self.stride = stride
#         self.max_cache_size = 80
        
#         self.map_cache: OrderedDict = OrderedDict()
#         # Cache for map metadata
#         self.map_metadata_cache: dict = {}
        
#         self.input_windows: list[dict] = []
#         self.prediction_windows: list[dict] = []
#         self.map_ids: list[str] = []
#         # Track which dataset each sample belongs to
#         self.dataset_indices: list[int] = []
        
#         self._read_data()
    
#     def _find_paths_and_maps(self):
#         """Find all paths_*.csv files and their corresponding maps_* folders."""
#         paths_files = glob.glob(os.path.join(self.data_path, "paths_*.csv"))
        
#         if not paths_files:
#             print(f"No paths_*.csv files found in {self.data_path}")
#             return []
        
#         # Extract dataset indices and sort
#         dataset_pairs = []
#         for paths_file in paths_files:
#             # Extract number from paths_X.csv
#             match = re.search(r'paths_(\d+)\.csv', os.path.basename(paths_file))
#             if match:
#                 idx = int(match.group(1))
#                 maps_folder = os.path.join(self.data_path, f"maps_{idx}")
                
#                 # Verify maps folder exists
#                 if os.path.exists(maps_folder):
#                     dataset_pairs.append((idx, paths_file, maps_folder))
#                 else:
#                     print(f"Warning: maps_{idx} folder not found for {paths_file}")
        
#         # Sort by index
#         dataset_pairs.sort(key=lambda x: x[0])
#         return dataset_pairs
    
#     def _load_map_metadata(self, map_id: str, dataset_idx: int):
#         """Load map metadata from YAML file."""
#         cache_key = (dataset_idx, map_id)
#         if cache_key in self.map_metadata_cache:
#             return self.map_metadata_cache[cache_key]
        
#         maps_folder = os.path.join(self.data_path, f"maps_{dataset_idx}")
#         yaml_path = os.path.join(maps_folder, f"map_{map_id}.yaml")
        
#         if not os.path.exists(yaml_path):
#             metadata = {'origin': [-6.4, -6.4], 'resolution': 0.05}
#             self.map_metadata_cache[cache_key] = metadata
#             return metadata
        
#         try:
#             with open(yaml_path, 'r') as f:
#                 yaml_data = yaml.safe_load(f)
                
#             metadata = {
#                 'origin': yaml_data.get('origin', [-6.4, -6.4]),
#                 'resolution': yaml_data.get('resolution', 0.05)
#             }
#             self.map_metadata_cache[cache_key] = metadata
#             return metadata
#         except Exception as e:
#             metadata = {'origin': [-6.4, -6.4], 'resolution': 0.05}
#             self.map_metadata_cache[cache_key] = metadata
#             return metadata
    
#     def _load_map_image(self, map_id: str, dataset_idx: int):
#         """Load raw map image with LRU caching."""
#         cache_key = (dataset_idx, map_id)
        
#         # Check cache - if found, move to end (most recently used)
#         if cache_key in self.map_cache:
#             self.map_cache.move_to_end(cache_key)
#             return self.map_cache[cache_key]
        
#         # Load image from disk
#         maps_folder = os.path.join(self.data_path, f"maps_{dataset_idx}")
#         image_path = os.path.join(maps_folder, f"grid_{map_id}.pgm")
        
#         if not os.path.exists(image_path):
#             # Return empty tensor if map not found
#             empty_image = torch.zeros(1, 256, 256)
#             self._add_to_cache(cache_key, empty_image)
#             return empty_image
        
#         try:
#             # Load image and convert to tensor (no preprocessing)
#             image = Image.open(image_path)
#             if image.mode != 'L':
#                 image = image.convert('L')
#             image_tensor = transforms.ToTensor()(image)
#             self._add_to_cache(cache_key, image_tensor)
#             return image_tensor
#         except Exception as e:
#             print(f"Error loading map image {image_path}: {e}")
#             empty_image = torch.zeros(1, 256, 256)
#             self._add_to_cache(cache_key, empty_image)
#             return empty_image
    
#     def _add_to_cache(self, cache_key, image_tensor):
#         """Add image to cache and evict oldest if necessary."""
#         self.map_cache[cache_key] = image_tensor
        
#         # Evict oldest entry if cache exceeds max size
#         if len(self.map_cache) > self.max_cache_size:
#             self.map_cache.popitem(last=False)
    
#     def _process_csv_file(self, csv_file: str, dataset_idx: int):
#         """Process a single CSV file."""
#         if not os.path.exists(csv_file):
#             print(f"CSV file {csv_file} not found")
#             return
        
#         df = pd.read_csv(csv_file)
#         if 'grid_id' not in df.columns:
#             print(f"No grid_id in {csv_file}, skipping")
#             return
        
#         # Get unique traj_ids for progress bar
#         traj_ids = df['traj_id'].unique()
        
#         # Initialize tqdm for trajectory processing
#         for traj_id in tqdm(traj_ids, desc=f'Processing paths_{dataset_idx}', unit='trajectory'):
#             group = df[df['traj_id'] == traj_id]
#             map_id = str(group['grid_id'].iloc[0])
#             sensor_data = group[FEATURE_COLS].values
#             total_frames = int(sensor_data[:, 4].max()) + 1  # max frame_id + 1
            
#             # Pre-organize data by frame for O(1) access
#             frame_dict = {}
#             for row in sensor_data:
#                 fid = int(row[4])
#                 if fid not in frame_dict:
#                     frame_dict[fid] = []
#                 frame_dict[fid].append(row)
            
#             # Process windows
#             window_len = self.seq_len + self.pred_len
#             for start_frame in range(0, total_frames - window_len + 1, self.stride):
#                 # Initialize arrays
#                 human_window = np.full((self.max_human_agents, self.seq_len, 3), np.nan, dtype=np.float32)
#                 prediction_window = np.full((self.max_human_agents, self.pred_len, 3), np.nan, dtype=np.float32)
#                 robot_window = np.full((self.max_robot_agents, self.seq_len, 3), np.nan, dtype=np.float32)
                
#                 # Initialize mask arrays (1 = visible, 0 = masked/missing)
#                 human_pos_mask = np.zeros((self.max_human_agents, self.seq_len), dtype=np.float32)
#                 prediction_pos_mask = np.zeros((self.max_human_agents, self.pred_len), dtype=np.float32)
#                 robot_pos_mask = np.zeros((self.max_robot_agents, self.seq_len), dtype=np.float32)
#                 stations_pos_mask = np.zeros((self.max_stations, self.seq_len), dtype=np.float32)
                
#                 # Initialize velocity arrays (scalar magnitude)
#                 human_velocity = np.zeros((self.max_human_agents, self.seq_len), dtype=np.float32)
#                 robot_velocity = np.zeros((self.max_robot_agents, self.seq_len), dtype=np.float32)
                
#                 # Initialize task_id array with shape [max_human_agents, 1, 1]
#                 task_id_array = np.zeros((self.max_human_agents, 1, 1), dtype=np.float32)
                
#                 # Initialize station array with shape [max_stations, seq_len, 2]
#                 station_window = np.zeros((self.max_stations, self.seq_len, 2), dtype=np.float32)
                
#                 # Collect all agents in window and create ID mapping
#                 human_ids = set()
#                 robot_ids = set()
#                 for fid in range(start_frame, start_frame + window_len):
#                     if fid in frame_dict:
#                         for row in frame_dict[fid]:
#                             if row[3] == 1:  # human
#                                 human_ids.add(int(row[5]))
#                             else:  # robot
#                                 robot_ids.add(int(row[5]))
                
#                 # Create consistent mappings
#                 human_id_to_slot = {aid: i for i, aid in enumerate(sorted(human_ids)[:self.max_human_agents])}
#                 robot_id_to_slot = {aid: i for i, aid in enumerate(sorted(robot_ids)[:self.max_robot_agents])}
                
#                 # Fill prediction window with masking
#                 for j in range(self.pred_len):
#                     fid = start_frame + self.seq_len + j
#                     if fid in frame_dict:
#                         for row in frame_dict[fid]:
#                             if row[3] == 1 and int(row[5]) in human_id_to_slot:
#                                 slot = human_id_to_slot[int(row[5])]
#                                 pos_mask = row[11]  # Extract pos_mask (12th column in FEATURE_COLS)
                                
#                                 if pos_mask == 1:  # Position is visible
#                                     prediction_window[slot, j, :] = row[0:3]
#                                     prediction_pos_mask[slot, j] = 1.0
#                                 else:  # Position is masked - set to zero
#                                     prediction_window[slot, j, :] = [0.0, 0.0, 0.0]
#                                     prediction_pos_mask[slot, j] = 0.0
                
#                 if np.isnan(prediction_window).all():
#                     continue
                
#                 # Fill input window with masking
#                 for i in range(self.seq_len):
#                     fid = start_frame + i
                    
#                     if fid in frame_dict:
#                         for row in frame_dict[fid]:
#                             agent_id = int(row[5])
#                             pos_mask = row[11]  # Extract pos_mask (12th column in FEATURE_COLS)
                            
#                             if row[3] == 1 and agent_id in human_id_to_slot:
#                                 slot = human_id_to_slot[agent_id]
#                                 if pos_mask == 1:  # Position is visible
#                                     human_window[slot, i, :] = row[0:3]
#                                     human_pos_mask[slot, i] = 1.0
#                                 else:  # Position is masked - set to zero
#                                     human_window[slot, i, :] = [0.0, 0.0, 0.0]
#                                     human_pos_mask[slot, i] = 0.0
                                    
#                             elif row[3] == 0 and agent_id in robot_id_to_slot:
#                                 slot = robot_id_to_slot[agent_id]
#                                 robot_window[slot, i, :] = row[0:3]
#                                 # Extract robot_pos/mask (14th column in FEATURE_COLS, index 13)
#                                 robot_mask = row[13]
#                                 robot_pos_mask[slot, i] = robot_mask
                        
#                         # Extract station coordinates and mask from the first row of this frame
#                         # (stations should be the same across all agents in a frame)
#                         if len(frame_dict[fid]) > 0:
#                             first_row = frame_dict[fid][0]
#                             # row indices: 7=station_1_x, 8=station_1_y, 9=station_2_x, 10=station_2_y
#                             station_window[0, i, 0] = first_row[7]  # station_1_x
#                             station_window[0, i, 1] = first_row[8]  # station_1_y
#                             station_window[1, i, 0] = first_row[9]  # station_2_x
#                             station_window[1, i, 1] = first_row[10] # station_2_y
                            
#                             # Extract stations_pos/mask (13th column in FEATURE_COLS, index 12)
#                             # This mask applies to all stations at this timestep
#                             stations_mask = first_row[12]
#                             stations_pos_mask[:, i] = stations_mask  # Broadcast to all stations
                
#                 # Extract task_id from the last timestep of input window
#                 last_frame_id = start_frame + self.seq_len - 1
#                 if last_frame_id in frame_dict:
#                     for row in frame_dict[last_frame_id]:
#                         agent_id = int(row[5])
#                         if row[3] == 1 and agent_id in human_id_to_slot:
#                             slot = human_id_to_slot[agent_id]
#                             # row[6] is task_id (7th column in FEATURE_COLS)
#                             task_id_array[slot, 0, 0] = row[6]
                
#                 # Skip window only if all human positions are NaN (no data at all)
#                 if np.isnan(human_window).all():
#                     continue
                
#                 # Calculate velocities for humans (magnitude only)
#                 # Only calculate velocity for visible (non-masked) positions
#                 for agent_idx in range(self.max_human_agents):
#                     for t in range(1, self.seq_len):
#                         if (human_pos_mask[agent_idx, t] == 1 and 
#                             human_pos_mask[agent_idx, t-1] == 1 and
#                             not np.isnan(human_window[agent_idx, t, 0]) and 
#                             not np.isnan(human_window[agent_idx, t-1, 0])):
#                             vel_vector = human_window[agent_idx, t, 0:2] - human_window[agent_idx, t-1, 0:2]
#                             human_velocity[agent_idx, t] = np.linalg.norm(vel_vector)
#                     # Set first timestep velocity to second timestep velocity
#                     human_velocity[agent_idx, 0] = human_velocity[agent_idx, 1]
                
#                 # Calculate velocities for robots (magnitude only)
#                 # Only calculate velocity for visible (non-masked) positions
#                 for agent_idx in range(self.max_robot_agents):
#                     for t in range(1, self.seq_len):
#                         if (robot_pos_mask[agent_idx, t] == 1 and 
#                             robot_pos_mask[agent_idx, t-1] == 1 and
#                             not np.isnan(robot_window[agent_idx, t, 0]) and 
#                             not np.isnan(robot_window[agent_idx, t-1, 0])):
#                             vel_vector = robot_window[agent_idx, t, 0:2] - robot_window[agent_idx, t-1, 0:2]
#                             robot_velocity[agent_idx, t] = np.linalg.norm(vel_vector)
#                     # Set first timestep velocity to second timestep velocity
#                     robot_velocity[agent_idx, 0] = robot_velocity[agent_idx, 1]
                
#                 # Replace NaN with 0
#                 human_window = np.nan_to_num(human_window, 0.0).astype(np.float32)
#                 robot_window = np.nan_to_num(robot_window, 0.0).astype(np.float32)
#                 prediction_window = np.nan_to_num(prediction_window, 0.0).astype(np.float32)
                
#                 # Concatenate all positions: humans + robots + stations
#                 # Shape: [max_human_agents + max_robot_agents + max_stations, seq_len, 2]
#                 all_agents_pos = np.concatenate([
#                     human_window[:,:,0:2],  # [max_human_agents, seq_len, 2]
#                     robot_window[:,:,0:2],  # [max_robot_agents, seq_len, 2]
#                     station_window          # [max_stations, seq_len, 2]
#                 ], axis=0)
                
#                 # Store windows
#                 self.input_windows.append({
#                     'human_pos': human_window[:,:,0:2],
#                     'human_vel': human_velocity,
#                     'human_pos/mask': np.expand_dims(human_pos_mask, axis=-1),  # Shape: [max_human_agents, seq_len, 1]
#                     'robot_pos': robot_window[:,:,0:2],
#                     'robot_vel': robot_velocity,
#                     'robot_pos/mask': np.expand_dims(robot_pos_mask, axis=-1),  # Shape: [max_robot_agents, seq_len, 1]
#                     'task_id': task_id_array,  # Shape: [max_human_agents, 1, 1]
#                     'stations_pos': station_window,  # Shape: [max_stations, seq_len, 2]
#                     'stations_pos/mask': np.expand_dims(stations_pos_mask, axis=-1),  # Shape: [max_stations, seq_len, 1]
#                     'all_agents_pos': all_agents_pos,  # Shape: [max_human_agents + max_robot_agents + max_stations, seq_len, 2]
#                 })
                
#                 self.prediction_windows.append({
#                     'prediction_pos': prediction_window[:,:,0:2],
#                     'prediction_pos_mask': prediction_pos_mask,  # Shape: [max_human_agents, pred_len]
#                 })
                
#                 self.map_ids.append(map_id)
#                 self.dataset_indices.append(dataset_idx)
    
#     def _read_data(self):
#         """Read all paths and maps datasets."""
#         dataset_pairs = self._find_paths_and_maps()
        
#         if not dataset_pairs:
#             print("No valid dataset pairs found")
#             return
        
#         print(f"Found {len(dataset_pairs)} dataset pairs")
        
#         # Process each dataset
#         for dataset_idx, paths_file, maps_folder in dataset_pairs:
#             print(f"\nProcessing dataset {dataset_idx}: {os.path.basename(paths_file)} with {os.path.basename(maps_folder)}")
#             self._process_csv_file(paths_file, dataset_idx)
        
#         # Print total number of extracted windows
#         print(f"\nTotal number of extracted windows across all datasets: {len(self.input_windows)}")
    
#     def __len__(self):
#         return len(self.input_windows)
    
#     def __getitem__(self, idx: int):
#         input_dict = self.input_windows[idx]
#         pred_dict = self.prediction_windows[idx]
#         map_id = self.map_ids[idx]
#         dataset_idx = self.dataset_indices[idx]
#         # Load raw map image and metadata from the correct maps folder
#         map_image = self._load_map_image(map_id, dataset_idx)
#         map_metadata = self._load_map_metadata(map_id, dataset_idx)
#         x = {
#             key: torch.tensor(value, dtype=torch.float32)
#             for key, value in input_dict.items()
#         }
        
#         # Add raw image and metadata for preprocessing in model
#         x['map_image'] = map_image
#         x['map_origin'] = torch.tensor(map_metadata['origin'], dtype=torch.float32)
#         x['map_resolution'] = torch.tensor(map_metadata['resolution'], dtype=torch.float32)
        
#         y = {
#             key: torch.tensor(value, dtype=torch.float32)
#             for key, value in pred_dict.items()
#         }
        
#         return x, y

import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import yaml
import glob
import re
from collections import OrderedDict

class TrajectoryPredictionDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int, pred_len: int, stride: int):
        self.data_path = data_path
        self.seq_len: int = seq_len
        self.pred_len: int = pred_len
        self.max_human_agents: int = 1
        self.max_robot_agents: int = 2
        
        # This will be updated dynamically based on the CSV content
        self.max_stations: int = 0
        
        self.stride = stride
        self.max_cache_size = 80
        
        self.map_cache: OrderedDict = OrderedDict()
        self.map_metadata_cache: dict = {}
        
        self.input_windows: list[dict] = []
        self.prediction_windows: list[dict] = []
        self.map_ids: list[str] = []
        self.dataset_indices: list[int] = []
        
        self._read_data()
    
    def _find_paths_and_maps(self):
        """Find all paths_*.csv files and their corresponding maps_* folders."""
        paths_files = glob.glob(os.path.join(self.data_path, "paths_*.csv"))
        
        if not paths_files:
            print(f"No paths_*.csv files found in {self.data_path}")
            return []
        
        dataset_pairs = []
        for paths_file in paths_files:
            match = re.search(r'paths_(\d+)\.csv', os.path.basename(paths_file))
            if match:
                idx = int(match.group(1))
                maps_folder = os.path.join(self.data_path, f"maps_{idx}")
                
                if os.path.exists(maps_folder):
                    dataset_pairs.append((idx, paths_file, maps_folder))
                else:
                    print(f"Warning: maps_{idx} folder not found for {paths_file}")
        
        dataset_pairs.sort(key=lambda x: x[0])
        return dataset_pairs
    
    def _load_map_metadata(self, map_id: str, dataset_idx: int):
        """Load map metadata from YAML file."""
        cache_key = (dataset_idx, map_id)
        if cache_key in self.map_metadata_cache:
            return self.map_metadata_cache[cache_key]
        
        maps_folder = os.path.join(self.data_path, f"maps_{dataset_idx}")
        yaml_path = os.path.join(maps_folder, f"map_{map_id}.yaml")
        
        # Default metadata
        default_meta = {'origin': [-6.4, -6.4], 'resolution': 0.05}

        if not os.path.exists(yaml_path):
            print(f"yaml with name {yaml_path} does not exist")
            self.map_metadata_cache[cache_key] = default_meta
            return default_meta
        
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                
            metadata = {
                'origin': yaml_data.get('origin'),
                'resolution': yaml_data.get('resolution', 0.05)
            }
            self.map_metadata_cache[cache_key] = metadata
            return metadata
        except Exception as e:
            print(f"Error occured=======: {e}")
            self.map_metadata_cache[cache_key] = default_meta
            return default_meta
    
    def _load_map_image(self, map_id: str, dataset_idx: int):
        """Load raw map image with LRU caching."""
        cache_key = (dataset_idx, map_id)
        
        if cache_key in self.map_cache:
            self.map_cache.move_to_end(cache_key)
            return self.map_cache[cache_key]
        
        maps_folder = os.path.join(self.data_path, f"maps_{dataset_idx}")
        image_path = os.path.join(maps_folder, f"grid_{map_id}.pgm")
        
        if not os.path.exists(image_path):
            empty_image = torch.zeros(1, 256, 256)
            self._add_to_cache(cache_key, empty_image)
            return empty_image
        
        try:
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            image_tensor = transforms.ToTensor()(image)
            self._add_to_cache(cache_key, image_tensor)
            return image_tensor
        except Exception as e:
            print(f"Error loading map image {image_path}: {e}")
            empty_image = torch.zeros(1, 256, 256)
            self._add_to_cache(cache_key, empty_image)
            return empty_image
    
    def _add_to_cache(self, cache_key, image_tensor):
        """Add image to cache and evict oldest if necessary."""
        self.map_cache[cache_key] = image_tensor
        
        if len(self.map_cache) > self.max_cache_size:
            self.map_cache.popitem(last=False)
    
    def _process_csv_file(self, csv_file: str, dataset_idx: int):
        """Process a single CSV file with dynamic station handling."""
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} not found")
            return
        
        df = pd.read_csv(csv_file)
        if 'grid_id' not in df.columns:
            print(f"No grid_id in {csv_file}, skipping")
            return

        # --- DYNAMIC COLUMN DETECTION START ---
        all_cols = df.columns.tolist()
        
        # Find all station columns using regex
        station_x_cols = [c for c in all_cols if re.match(r'station_\d+_x', c)]
        station_y_cols = [c for c in all_cols if re.match(r'station_\d+_y', c)]
        
        # Ensure they are sorted (station_0, station_1, ...)
        station_x_cols.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))
        station_y_cols.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))
        
        # Update self.max_stations based on the current file
        current_num_stations = len(station_x_cols)
        
        # Update class attribute if this file has more stations or if it's the first file
        # Note: If multiple CSVs have different numbers of stations, this ensures 
        # we know the count for the current processing block. 
        # CAUTION: If CSVs in the same dataset have DIFFERENT station counts, 
        # the resulting tensors will have different shapes, which will crash 
        # a standard DataLoader. We assume consistency within a dataset run 
        # or that the user handles ragged batches.
        self.max_stations = current_num_stations

        # Construct the feature columns list dynamically
        base_cols = ['x', 'y', 'z', 'agent_type', 'frame_id', 'agent_id']
        mask_cols = ['pos_mask', 'stations_pos/mask', 'robot_pos/mask']
        
        # Interleave station x and y columns
        station_cols_ordered = []
        for sx, sy in zip(station_x_cols, station_y_cols):
            station_cols_ordered.extend([sx, sy])
            
        feature_cols = base_cols + station_cols_ordered + mask_cols
        
        # Create a mapping from column name to index in the numpy array
        # This replaces hardcoded indices (e.g., row[4], row[6])
        col_map = {name: i for i, name in enumerate(feature_cols)}
        
        # --- DYNAMIC COLUMN DETECTION END ---

        traj_ids = df['traj_id'].unique()
        
        for traj_id in tqdm(traj_ids, desc=f'Processing paths_{dataset_idx}', unit='trajectory'):
            group = df[df['traj_id'] == traj_id]
            map_id = str(group['grid_id'].iloc[0])
            
            # Select only the columns we identified
            sensor_data = group[feature_cols].values
            
            # Use col_map for frame_id access
            frame_col_idx = col_map['frame_id']
            total_frames = int(sensor_data[:, frame_col_idx].max()) + 1
            
            frame_dict = {}
            for row in sensor_data:
                fid = int(row[frame_col_idx])
                if fid not in frame_dict:
                    frame_dict[fid] = []
                frame_dict[fid].append(row)
            
            window_len = self.seq_len + self.pred_len
            
            # Pre-fetch indices to avoid dictionary lookups inside the inner loop
            idx_agent_type = col_map['agent_type']
            idx_agent_id = col_map['agent_id']
            idx_pos_mask = col_map['pos_mask']
            idx_robot_mask = col_map['robot_pos/mask']
            idx_stations_mask = col_map['stations_pos/mask']
            
            for start_frame in range(0, total_frames - window_len + 1, self.stride):
                human_window = np.full((self.max_human_agents, self.seq_len, 3), np.nan, dtype=np.float32)
                prediction_window = np.full((self.max_human_agents, self.pred_len, 3), np.nan, dtype=np.float32)
                robot_window = np.full((self.max_robot_agents, self.seq_len, 3), np.nan, dtype=np.float32)
                
                human_pos_mask = np.zeros((self.max_human_agents, self.seq_len), dtype=np.float32)
                prediction_pos_mask = np.zeros((self.max_human_agents, self.pred_len), dtype=np.float32)
                robot_pos_mask = np.zeros((self.max_robot_agents, self.seq_len), dtype=np.float32)
                stations_pos_mask = np.zeros((self.max_stations, self.seq_len), dtype=np.float32)
                
                human_velocity = np.zeros((self.max_human_agents, self.seq_len), dtype=np.float32)
                robot_velocity = np.zeros((self.max_robot_agents, self.seq_len), dtype=np.float32)
                
                station_window = np.zeros((self.max_stations, self.seq_len, 2), dtype=np.float32)
                
                human_ids = set()
                robot_ids = set()
                
                # Identify agents in current window
                for fid in range(start_frame, start_frame + window_len):
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            if row[idx_agent_type] == 1:  # human
                                human_ids.add(int(row[idx_agent_id]))
                            else:  # robot
                                robot_ids.add(int(row[idx_agent_id]))
                
                human_id_to_slot = {aid: i for i, aid in enumerate(sorted(human_ids)[:self.max_human_agents])}
                robot_id_to_slot = {aid: i for i, aid in enumerate(sorted(robot_ids)[:self.max_robot_agents])}
                
                # Fill prediction window
                for j in range(self.pred_len):
                    fid = start_frame + self.seq_len + j
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            a_id = int(row[idx_agent_id])
                            if row[idx_agent_type] == 1 and a_id in human_id_to_slot:
                                slot = human_id_to_slot[a_id]
                                pos_mask = row[idx_pos_mask]
                                
                                if pos_mask == 1:
                                    prediction_window[slot, j, :] = row[0:3] # x,y,z are always 0,1,2
                                    prediction_pos_mask[slot, j] = 1.0
                                else:
                                    prediction_window[slot, j, :] = [0.0, 0.0, 0.0]
                                    prediction_pos_mask[slot, j] = 0.0
                
                if np.isnan(prediction_window).all():
                    continue
                
                # Fill input window
                for i in range(self.seq_len):
                    fid = start_frame + i
                    
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            agent_id = int(row[idx_agent_id])
                            pos_mask = row[idx_pos_mask]
                            
                            if row[idx_agent_type] == 1 and agent_id in human_id_to_slot:
                                slot = human_id_to_slot[agent_id]
                                if pos_mask == 1:
                                    human_window[slot, i, :] = row[0:3]
                                    human_pos_mask[slot, i] = 1.0
                                else:
                                    human_window[slot, i, :] = [0.0, 0.0, 0.0]
                                    human_pos_mask[slot, i] = 0.0
                                    
                            elif row[idx_agent_type] == 0 and agent_id in robot_id_to_slot:
                                slot = robot_id_to_slot[agent_id]
                                robot_window[slot, i, :] = row[0:3]
                                robot_mask = row[idx_robot_mask]
                                robot_pos_mask[slot, i] = robot_mask
                        
                        # Extract station coordinates
                        if len(frame_dict[fid]) > 0:
                            first_row = frame_dict[fid][0]
                            
                            # Dynamic station extraction
                            for station_idx in range(self.max_stations):
                                # Construct keys based on standard naming
                                sx_key = f'station_{station_idx}_x'
                                sy_key = f'station_{station_idx}_y'
                                
                                if sx_key in col_map and sy_key in col_map:
                                    station_window[station_idx, i, 0] = first_row[col_map[sx_key]]
                                    station_window[station_idx, i, 1] = first_row[col_map[sy_key]]
                            
                            stations_mask = first_row[idx_stations_mask]
                            stations_pos_mask[:, i] = stations_mask
                
                if np.isnan(human_window).all():
                    continue
                
                # Calculate velocities for humans
                for agent_idx in range(self.max_human_agents):
                    for t in range(1, self.seq_len):
                        if (human_pos_mask[agent_idx, t] == 1 and 
                            human_pos_mask[agent_idx, t-1] == 1 and
                            not np.isnan(human_window[agent_idx, t, 0]) and 
                            not np.isnan(human_window[agent_idx, t-1, 0])):
                            vel_vector = human_window[agent_idx, t, 0:2] - human_window[agent_idx, t-1, 0:2]
                            human_velocity[agent_idx, t] = np.linalg.norm(vel_vector)
                    human_velocity[agent_idx, 0] = human_velocity[agent_idx, 1]
                
                # Calculate velocities for robots
                for agent_idx in range(self.max_robot_agents):
                    for t in range(1, self.seq_len):
                        if (robot_pos_mask[agent_idx, t] == 1 and 
                            robot_pos_mask[agent_idx, t-1] == 1 and
                            not np.isnan(robot_window[agent_idx, t, 0]) and 
                            not np.isnan(robot_window[agent_idx, t-1, 0])):
                            vel_vector = robot_window[agent_idx, t, 0:2] - robot_window[agent_idx, t-1, 0:2]
                            robot_velocity[agent_idx, t] = np.linalg.norm(vel_vector)
                    robot_velocity[agent_idx, 0] = robot_velocity[agent_idx, 1]
                
                # Replace NaN with 0
                human_window = np.nan_to_num(human_window, 0.0).astype(np.float32)
                robot_window = np.nan_to_num(robot_window, 0.0).astype(np.float32)
                prediction_window = np.nan_to_num(prediction_window, 0.0).astype(np.float32)
                
                # Concatenate all positions
                all_agents_pos = np.concatenate([
                    human_window[:,:,0:2],
                    robot_window[:,:,0:2],
                    station_window
                ], axis=0)
                
                self.input_windows.append({
                    'human_pos': human_window[:,:,0:2],
                    'human_vel': human_velocity,
                    'human_pos/mask': np.expand_dims(human_pos_mask, axis=-1),
                    'robot_pos': robot_window[:,:,0:2],
                    'robot_vel': robot_velocity,
                    'robot_pos/mask': np.expand_dims(robot_pos_mask, axis=-1),
                    'stations_pos': station_window,
                    'stations_pos/mask': np.expand_dims(stations_pos_mask, axis=-1),
                    'all_agents_pos': all_agents_pos,
                })
                
                self.prediction_windows.append({
                    'prediction_pos': prediction_window[:,:,0:2],
                    'prediction_pos_mask': prediction_pos_mask,
                    'prediction_pos/mask': np.expand_dims(prediction_pos_mask, axis=-1),  # NumPy equivalent
                })

                self.map_ids.append(map_id)
                self.dataset_indices.append(dataset_idx)
    
    def _read_data(self):
        """Read all paths and maps datasets."""
        dataset_pairs = self._find_paths_and_maps()
        
        if not dataset_pairs:
            print("No valid dataset pairs found")
            return
        
        print(f"Found {len(dataset_pairs)} dataset pairs")
        
        for dataset_idx, paths_file, maps_folder in dataset_pairs:
            print(f"\nProcessing dataset {dataset_idx}: {os.path.basename(paths_file)} with {os.path.basename(maps_folder)}")
            self._process_csv_file(paths_file, dataset_idx)
        
        print(f"\nTotal number of extracted windows across all datasets: {len(self.input_windows)}")
    
    def __len__(self):
        return len(self.input_windows)
    
    def __getitem__(self, idx: int):
        input_dict = self.input_windows[idx]
        pred_dict = self.prediction_windows[idx]
        map_id = self.map_ids[idx]
        dataset_idx = self.dataset_indices[idx]
        
        map_image = self._load_map_image(map_id, dataset_idx)
        map_metadata = self._load_map_metadata(map_id, dataset_idx)
        
        x = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in input_dict.items()
        }
        
        x['map_image'] = map_image
        x['map_origin'] = torch.tensor(map_metadata['origin'], dtype=torch.float32)
        x['map_resolution'] = torch.tensor(map_metadata['resolution'], dtype=torch.float32)

        y = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in pred_dict.items()
        }
        
        return x, y