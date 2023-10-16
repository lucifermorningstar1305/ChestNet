"""
@author Adityam Ghosh
Date: 10-17-2023

"""
from typing import Any, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import lightning.pytorch as pl

from scripts.label_smooth_loss import LabelSmoothingLoss

class LitClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, n_classes: int, lr: Optional[float]=1e-3, 
                 smoothing_factor: Optional[float]=1e-1, min_lr: Optional[float]=1e-6):

        super().__init__()

        self.classifier = model
        self.lr = lr
        self.criterion = LabelSmoothingLoss(num_classes=n_classes, smoothing_factor=smoothing_factor)
        self.min_lr = min_lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def _compute_loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(yhat, y)
    
    def _common_steps(self, batch: torch.Tensor) -> Dict:

        x, y = batch["img"], batch["label"]
        yhat = self(x)

        loss = self._compute_loss(yhat, y)

        return {
            "y": y,
            "yhat": yhat,
            "loss": loss
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

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(.9, .999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
