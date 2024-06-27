

from utils import *

class SoccerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        state = np.load(row['state_path'])
        action = [int(row['end_y']), int(row['end_x'])]
        if action[0] >= 80:
            action[0] = 79
        if action[1] >= 120:
            action[1] = 119
        reward = 1 if row['outcome_name'] == 'success' else -1

        try:
            ravel_action = np.ravel_multi_index(action, (80, 120))
        except ValueError as e:
            print(f"Invalid action coordinates: {action}")
            raise e

        return state, ravel_action, reward
