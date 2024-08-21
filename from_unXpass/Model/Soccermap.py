
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Any, Dict, List
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from from_unXpass.Model.base_architecture import SoccerMap, pixel
from from_unXpass.dataset.Into_Soccermap_tensor import ToSoccerMapTensor
from from_unXpass.Model.base import UnxPassPytorchComponent, UnxpassComponent




class PassSuccessComponent(UnxpassComponent):
    """The pass success probability component.

    From any given game situation where a player controls the ball, the model
    estimates the success probability of a pass attempted towards a potential
    destination location.
    """

    component_name = "pass_success"

    def _get_metrics(self, y, y_hat):
        y_pred = y_hat > 0.5
        return {
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "log_loss": log_loss(y, y_hat),
            "brier": brier_score_loss(y, y_hat),
            "roc_auc": roc_auc_score(y, y_hat),
        }




class PytorchSoccerMapModel(pl.LightningModule):
    """A pass success probability model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-4,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(in_channels=7)
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

    def step(self, batch: Any):
        x, mask, y = batch
        surface = self.forward(x)
        y_hat = pixel(surface, mask)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)


        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, _, _ = batch
        surface = self(x)
        return surface

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

class SoccerMapComponent(PassSuccessComponent, UnxPassPytorchComponent):
    """A SoccerMap deep-learning model."""

    def __init__(self, model: PytorchSoccerMapModel):
        super().__init__(
            model=model,
            features={
                "startlocation": ["start_x", "start_y"],
                "endlocation": ["end_x", "end_y"],
                "freeze_frame_360": ["freeze_frame_360"],
            },
            label=["success"],
            transform=ToSoccerMapTensor(dim=(68, 104)),
        )