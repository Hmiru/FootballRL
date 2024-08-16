import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from from_unXpass.dataset.Into_Soccermap_tensor import *


class SoccerDataset(Dataset):
    def __init__(self, samples, transform=None, train=True):
        self.samples = samples
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        original_data = sample.copy()  # Store the original data

        sample = self.transform(sample)
        if self.train:
            return sample  # Always return both

        return sample, original_data  # Always return both
if __name__ == "__main__":
    # Load your data
    data_path = '../dataset/total_data_with_state_label_mask.csv'
    df = pd.read_csv(data_path, index_col=0)

    # Apply the transformation to each row
    samples = df.apply(convert_row_to_sample, axis=1).tolist()
    transform = ToSoccerMapTensor(dim=(68, 104))
    dataset = SoccerDataset(samples, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Test dataloader to visualize data
    test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    batch, original_data_batch = next(iter(test_dataloader))

    # Print the transformed batch and original data
    print(batch)
    print(original_data_batch)
