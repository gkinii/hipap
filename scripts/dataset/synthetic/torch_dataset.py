import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import yaml

FEATURE_COLS = ['x', 'y', 'z', 'agent_type', 'frame_id', 'agent_id']

class TrajectoryPredictionDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int, pred_len: int):
        self.data_path = data_path
        self.csv_file = os.path.join(data_path, "paths.csv")
        self.seq_len: int = seq_len
        self.pred_len: int = pred_len
        self.max_human_agents: int = 1
        self.max_robot_agents: int = 2
        self.stride: int = 1
        
        # Cache for raw map images
        self.map_cache: dict = {}
        # Cache for map metadata
        self.map_metadata_cache: dict = {}
        
        self.input_windows: list[dict] = []
        self.prediction_windows: list[dict] = []
        self.map_ids: list[str] = []
        
        self._read_data()
    
    def _load_map_metadata(self, map_id: str):
        """Load map metadata from YAML file."""
        if map_id in self.map_metadata_cache:
            return self.map_metadata_cache[map_id]
        
        yaml_path = os.path.join(self.data_path, f"maps/map_{map_id}.yaml")
        if not os.path.exists(yaml_path):
            metadata = {'origin': [-6.4, -6.4], 'resolution': 0.05}
            self.map_metadata_cache[map_id] = metadata
            return metadata
        
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                
            metadata = {
                'origin': yaml_data.get('origin', [-6.4, -6.4]),
                'resolution': yaml_data.get('resolution', 0.05)
            }
            self.map_metadata_cache[map_id] = metadata
            return metadata
        except Exception as e:
            metadata = {'origin': [-6.4, -6.4], 'resolution': 0.05}
            self.map_metadata_cache[map_id] = metadata
            return metadata
    
    def _load_map_image(self, map_id: str):
        """Load raw map image."""
        if map_id in self.map_cache:
            return self.map_cache[map_id]
        
        image_path = os.path.join(self.data_path, f"maps/grid_{map_id}.pgm")
        
        if not os.path.exists(image_path):
            # Return empty tensor if map not found
            empty_image = torch.zeros(1, 256, 256)
            self.map_cache[map_id] = empty_image
            return empty_image
        
        try:
            # Load image and convert to tensor (no preprocessing)
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            image_tensor = transforms.ToTensor()(image)
            self.map_cache[map_id] = image_tensor
            return image_tensor
        except Exception as e:
            print(f"Error loading map image {image_path}: {e}")
            empty_image = torch.zeros(1, 256, 256)
            self.map_cache[map_id] = empty_image
            return empty_image
    
    def _read_data(self):
        if not os.path.exists(self.csv_file):
            print(f"CSV file {self.csv_file} not found")
            return
        
        df = pd.read_csv(self.csv_file)
        if 'grid_id' not in df.columns:
            print(f"No grid_id in {self.csv_file}, skipping")
            return
        
        # Get unique traj_ids for progress bar
        traj_ids = df['traj_id'].unique()
        
        # Initialize tqdm for trajectory processing
        for traj_id in tqdm(traj_ids, desc='Processing trajectories', unit='trajectory'):
            group = df[df['traj_id'] == traj_id]
            map_id = str(group['grid_id'].iloc[0])
            sensor_data = group[FEATURE_COLS].values
            total_frames = int(sensor_data[:, 4].max()) + 1  # max frame_id + 1
            
            # Pre-organize data by frame for O(1) access
            frame_dict = {}
            for row in sensor_data:
                fid = int(row[4])
                if fid not in frame_dict:
                    frame_dict[fid] = []
                frame_dict[fid].append(row)
            
            # Process windows
            window_len = self.seq_len + self.pred_len
            for start_frame in range(0, total_frames - window_len + 1, self.stride):
                # Initialize arrays
                human_window = np.full((self.max_human_agents, self.seq_len, 3), np.nan, dtype=np.float32)
                prediction_window = np.full((self.max_human_agents, self.pred_len, 3), np.nan, dtype=np.float32)
                robot_window = np.full((self.max_robot_agents, self.seq_len, 3), np.nan, dtype=np.float32)
                
                # Collect all agents in window and create ID mapping
                human_ids = set()
                robot_ids = set()
                for fid in range(start_frame, start_frame + window_len):
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            if row[3] == 1:  # human
                                human_ids.add(int(row[5]))
                            else:  # robot
                                robot_ids.add(int(row[5]))
                
                # Create consistent mappings
                human_id_to_slot = {aid: i for i, aid in enumerate(sorted(human_ids)[:self.max_human_agents])}
                robot_id_to_slot = {aid: i for i, aid in enumerate(sorted(robot_ids)[:self.max_robot_agents])}
                
                # Fill prediction window
                for j in range(self.pred_len):
                    fid = start_frame + self.seq_len + j
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            if row[3] == 1 and int(row[5]) in human_id_to_slot:
                                slot = human_id_to_slot[int(row[5])]
                                prediction_window[slot, j, :] = row[0:3]
                
                if np.isnan(prediction_window).all():
                    continue
                
                # Fill input window
                for i in range(self.seq_len):
                    fid = start_frame + i
                    
                    if fid in frame_dict:
                        for row in frame_dict[fid]:
                            agent_id = int(row[5])
                            if row[3] == 1 and agent_id in human_id_to_slot:
                                slot = human_id_to_slot[agent_id]
                                human_window[slot, i, :] = row[0:3]
                            elif row[3] == 0 and agent_id in robot_id_to_slot:
                                slot = robot_id_to_slot[agent_id]
                                robot_window[slot, i, :] = row[0:3]
                
                # Skip window only if all human positions are NaN (no data at all)
                if np.isnan(human_window).all():
                    continue
                
                # Replace NaN with 0
                human_window = np.nan_to_num(human_window, 0.0).astype(np.float32)
                robot_window = np.nan_to_num(robot_window, 0.0).astype(np.float32)
                prediction_window = np.nan_to_num(prediction_window, 0.0).astype(np.float32)
                
                # Store windows
                self.input_windows.append({
                    'human_pos': human_window[:,:,0:2],
                    'robot_pos': robot_window[:,:,0:2],
                })

                
                self.prediction_windows.append({
                    'prediction_pos': prediction_window[:,:,0:2],
                })
                
                self.map_ids.append(map_id)
        
        # Print total number of extracted windows
        print(f"Total number of extracted windows: {len(self.input_windows)}")
    
    def __len__(self):
        return len(self.input_windows)
    
    def __getitem__(self, idx: int):
        input_dict = self.input_windows[idx]
        pred_dict = self.prediction_windows[idx]
        map_id = self.map_ids[idx]
        
        # Load raw map image and metadata
        map_image = self._load_map_image(map_id)
        map_metadata = self._load_map_metadata(map_id)
        
        x = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in input_dict.items()
        }
        
        # Add raw image and metadata for preprocessing in model
        x['map_image'] = map_image
        x['map_origin'] = torch.tensor(map_metadata['origin'], dtype=torch.float32)
        x['map_resolution'] = torch.tensor(map_metadata['resolution'], dtype=torch.float32)
        
        y = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in pred_dict.items()
        }
        
        return x, y