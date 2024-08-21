import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from from_unXpass.Model.Soccermap import PytorchSoccerMapModel, SoccerMapComponent
from from_unXpass.unxpass.datasets import PassesDataset
from from_unXpass.dataset.Into_Soccermap_tensor import ToSoccerMapTensor

if __name__ == "__main__":
    # Initialize your PyTorch model
    features = {
        "startlocation": ["start_x", "start_y"],
        "endlocation": ["end_x", "end_y"],
        "freeze_frame_360": ["freeze_frame_360"]
    }
    label = ["success"]

    # Prepare the dataset
    dataset = PassesDataset(
        xfns=features,
        yfns=label,
        transform=ToSoccerMapTensor(dim=(68, 104))
    )

    # Create DataLoaders for training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model
    model = PytorchSoccerMapModel(lr=1e-4)
    component = SoccerMapComponent(model=model)

    # Set up PyTorch Lightning callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val/loss', patience=3, mode='min')

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stop_callback],
        gpus=1 if torch.cuda.is_available() else 0,
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(component, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test the model
    test_results = trainer.test(test_dataloaders=val_loader)
    print(f"Test Results: {test_results}")
