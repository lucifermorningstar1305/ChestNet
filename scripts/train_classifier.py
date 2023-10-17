"""
@author Adityam Ghosh
Date: 10-17-2023

"""
from typing import Any, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl
import torchmetrics

from scripts.losses import LabelSmoothingLoss, FocalLoss

class LitClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: Optional[float]=1e-3, 
                 smoothing_factor: Optional[float]=1e-1, min_lr: Optional[float]=1e-6, as_anomaly: Optional[bool]=False):

        super().__init__()

        self.classifier = model
        self.lr = lr
        n_classes = model.model.fc.out_features
        self.criterion = LabelSmoothingLoss(num_classes=n_classes, smoothing_factor=smoothing_factor)
        # self.criterion = LabelSmoothingLoss(num_classes=n_classes, smoothing_factor=smoothing_factor) if not as_anomaly else FocalLoss(alpha=.25, gamma=2)
        self.min_lr = min_lr

        self.recall = torchmetrics.Recall(task="binary" if as_anomaly else "multiclass", num_classes=n_classes, average="micro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def _compute_loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(yhat, y)
    
    def _common_steps(self, batch: torch.Tensor) -> Dict:

        x, y = batch["img"], batch["label"]
        yhat = self(x)

        loss = self._compute_loss(yhat, y)
        # print(yhat.shape, y.shape)
        recall_score = self.recall(torch.argmax(yhat, dim=1), y)

        return {
            "y": y,
            "yhat": yhat,
            "loss": loss,
            "recall_score": recall_score
        }
    
    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:
        
        res = self._common_steps(batch)
        self.log("train_loss", res["loss"], on_step=True, on_epoch=True, logger=True, prog_bar=True, rank_zero_only=True)

        return {
            "loss": res["loss"]
        }
    
    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):

        res = self._common_steps(batch)
        self.log("val_loss", res["loss"], on_step=False, on_epoch=True, logger=True, prog_bar=True, rank_zero_only=True)
        self.log("val_recall", res["recall_score"], on_step=False, on_epoch=True, logger=True, prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(.9, .999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
