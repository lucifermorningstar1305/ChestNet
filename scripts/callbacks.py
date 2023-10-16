from typing import Optional, Dict, Any, Callable

import numpy as np
import torch
import torch.utils.data as td
import torchvision
import os
import wandb

from glob import glob


class CustomModelCheckpoint(object):
    def __init__(self, monitor: str, direction: Optional[str] = "min", 
                 chkpt_dir:Optional[str]="../checkpoints", chkpt_fn: Optional[str]="checkpoint", verbose: Optional[bool]=False):

        self.monitor = monitor
        self.direction = direction
        self.chkpt_dir = chkpt_dir
        self.chkpt_fn = chkpt_fn
        self.verbose = verbose

        self._loss = np.inf if direction == "min" else 0

        n_files = len(glob(self.chkpt_dir+f"/{self.chkpt_fn}*.ckpt"))
        self.chkpt_path = os.path.join(self.chkpt_dir, self.chkpt_fn+f"_v-{n_files}.ckpt")

        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)

    def save(self, losses: Dict, models: Dict, optimizers: Dict, fabric: Callable):
        
        if self.monitor not in losses:
            raise Exception(f"Could not find {self.monitor} in losses")


        state = {
            "generator": models["generator"],
            "discriminator": models["discriminator"],
            "optimizer_g": optimizers["generator"],
            "optimizer_d": optimizers["discriminator"]
        }

        if (self.direction == "min" and losses[self.monitor] < self._loss) or (self.direction == "max" and losses[self.monitor] > self._loss):
            if self.verbose:
                print(f"Saving model checkpoint to {self.chkpt_path}")

            self._loss = losses[self.monitor]
            fabric.save(self.chkpt_path, state)

        
class CustomEarlyStopping(object):
    def __init__(self, monitor: str, direction: Optional[str]="min", patience: Optional[int]=3, verbose: Optional[bool]=False):

        self.monitor = monitor
        self.direction = direction
        self.patience = patience
        self.verbose = verbose
        self._loss = np.inf if direction == "min" else 0
        self.counter = 0

    def check(self, losses: Dict):

        if self.monitor not in losses:
            raise Exception(f"Could not find {self.monitor} in the losses")
        
        if (self.direction == "min" and losses[self.monitor] < self._loss) or (self.direction == "max" and losses[self.monitor] > self._loss):
            if self.verbose:
                print(f"New {self.monitor} reached {losses[self.monitor]:.5f}")
            
            self._loss = losses[self.monitor]
            self.counter = 0
        
        elif losses[self.monitor] == np.nan:
            return True
        else:
            print(f"{losses[self.monitor]} was not in the top-1")
            self.counter += 1

        return self.counter >= self.patience

            
class GenerateCallbacks(object):
    def __init__(self, samples: torch.utils.data.Dataset):

        self.samples = samples

    def log_image(self, model: torch.nn.Module, logger: Any):

        with torch.no_grad():
            model.eval()
            
            imgs = torch.stack([self.samples[i]["img"] for i in range(8)], dim=0).to(model.device)
            _, gen_imgs, _ = model(imgs)

            final_imgs = torch.stack([imgs.detach(), gen_imgs.detach()], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(final_imgs, nrow=2)
            logger.log({"samples": [wandb.Image(grid, caption="Reconstructed images")]})
