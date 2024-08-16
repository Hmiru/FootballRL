import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from network import SoccerMap

# 데이터셋 클래스
class SoccerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        state = np.load(row['state_path'])
        end_x = row['end_x']
        end_y = row['end_y']
        success = 1.0 if row['outcome_name'] == 'success' else 0.0
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1)
        end_x_tensor = torch.tensor(end_x, dtype=torch.float32)
        end_y_tensor = torch.tensor(end_y, dtype=torch.float32)
        success_tensor = torch.tensor(success, dtype=torch.float32)
        return state_tensor, end_x_tensor, end_y_tensor, success_tensor