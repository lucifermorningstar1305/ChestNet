from typing import Optional, Dict, Any, Callable

import numpy as np
import os

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

        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)

    def save(self, losses: Dict, models: Dict, optimizers: Dict, fabric: Callable):
        
        if self.monitor not in losses:
            raise Exception(f"Could not find {self.monitor} in losses")

        n_files = len(glob(self.chkpt_dir+f"/{self.chkpt_fn}*.ckpt"))
        chkpt_path = os.path.join(self.chkpt_dir, self.chkpt_fn+f"_v-{n_files}.ckpt")

        state = {
            "generator": models["generator"],
            "discriminator": models["discriminator"],
            "optimizer_g": optimizers["generator"],
            "optimizer_d": optimizers["discriminator"]
        }

        if (self.direction == "min" and losses[self.monitor] < self._loss) or (self.direction == "max" and losses[self.monitor] > self._loss):
            if self.verbose:
                print(f"Saving model checkpoint to {chkpt_path}")

            self._loss = losses[self.monitor]
            fabric.save(chkpt_path, state)

        
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
        
        else:
            print(f"{losses[self.monitor]} was not in the top-1")
            self.counter += 1

        return self.counter >= self.patience

            
