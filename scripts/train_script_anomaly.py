"""
@author: Adityam Ghosh
Date: 10-16-2023

"""
from typing import List, Tuple, Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import os
import sys

from scripts.callbacks import CustomEarlyStopping, CustomModelCheckpoint
from rich.progress import Progress, MofNCompleteColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

class CustomTrainer(object):
    def __init__(self, train_cfg: Optional[Dict]={}, 
                 accelerator: Optional[str]="auto", 
                 strategy="auto", 
                 devices: Optional[int]=1, 
                 max_epochs: Optional[int]=None, 
                 precision: Optional[int|str]="32-true", 
                 logger: Optional[Callable]=None, 
                 grad_accum_steps: Optional[int]=1, 
                 ckpt_interval: Optional[int]=1, 
                 validate_interval: Optional[int]=1, 
                 callbacks: Optional[Dict]=None):
        

        assert isinstance(train_cfg, dict), f"Expected train_cfg to be of type Dictionary. Found {type(train_cfg)}"
        
        self.max_epochs = max_epochs
        self.logger = logger

        self.n_critic_repeats = train_cfg.get("n_critic_repeats", 1)
        self.advesarial_loss_wt = train_cfg.get("adversarial_loss_wt", 1)
        self.contextual_loss_wt = train_cfg.get("contextual_loss_wt", 50)
        self.encoder_loss_wt = train_cfg.get("encoder_loss_wt", 1)
        self.betas_g = train_cfg.get("betas_g", (0.9, 0.999))
        self.betas_d = train_cfg.get("betas_d", (.9, .999))
        self.lr_gen = train_cfg.get("lr_g", 1e-3)
        self.lr_disc = train_cfg.get("lr_d", 1e-3)

        self.global_step = 0
        self.grad_accum_steps = grad_accum_steps

        self.ckpt_interval = ckpt_interval

        self.validate_interval = validate_interval


        self.ckpt_callback = callbacks.get("checkpoint_callback", None) if callbacks is not None else None
        self.early_stop_callback = callbacks.get("early_stop_callback", None) if callbacks is not None else None


        self.fabric = L.Fabric(accelerator=accelerator, strategy=strategy, devices=devices, precision=precision)
        self.fabric.launch()


    def _get_gradient(self, discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        
        mixed_imgs = real * epsilon + fake * (1 - epsilon)
        mixed_scores, _ = discriminator(mixed_imgs)

        gradient = torch.autograd.grad(
            inputs = mixed_imgs,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]

        return gradient
    
    def _gradient_penalty(self, gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)

        return penalty
    
    def _get_l2_loss(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: Optional[str]=None) -> torch.Tensor:

        if reduction == "mean":
            return torch.mean(torch.pow((inputs - targets), 2))
        
        else:
            return torch.pow((inputs - targets), 2)
    
    def _get_l1_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        return torch.mean(torch.abs(inputs - targets))
    
    def _get_disc_loss(self, real: torch.Tensor, fake: torch.Tensor, model: nn.Module) -> torch.Tensor:

        real_labels = torch.ones(real.size(0)).reshape(-1, 1).type_as(real)
        real_pred, _ = model(real)

        real_loss = F.binary_cross_entropy_with_logits(real_pred, real_labels)

        fake_labels = torch.zeros(fake.size(0)).reshape(-1, 1).type_as(fake)
        fake_pred, _ = model(fake.detach())
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)

        eps = torch.rand(real.size(0), 1, 1, 1, device=real.device, requires_grad=True)
        gradient = self._get_gradient(model, real, fake.detach(), eps)
        gp = self._gradient_penalty(gradient)

        dloss = fake_loss - real_loss + 10. * gp

        return dloss


    def _get_gen_loss(self, real: torch.Tensor, gen_model: nn.Module, disc_model: nn.Module) -> torch.Tensor:

        z, gen_img, zhat = gen_model(real)

        _, real_img_rep = disc_model(real)
        _, fake_img_rep = disc_model(gen_img)


        adv_loss = self._get_l2_loss(real_img_rep, fake_img_rep, reduction="mean")
        cont_loss = self._get_l1_loss(real, gen_img)
        enc_loss = self._get_l2_loss(z, zhat, reduction="mean")

        gen_loss = -1 * (self.advesarial_loss_wt*adv_loss + self.contextual_loss_wt * cont_loss + self.encoder_loss_wt * enc_loss)

        return gen_loss



    def fit(self, model: Dict, train_loader: torch.utils.data.DataLoader, optimizer: Dict, val_loader: Optional[torch.utils.data.DataLoader]=None):


        train_loader = self.fabric.setup_dataloaders(train_loader)

        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader)


        generator = model["generator"]
        discriminator = model["discriminator"]

        optimizer_g = optimizer["generator"](generator.parameters(), lr=self.lr_gen, betas=self.betas_g)
        optimizer_d = optimizer["discriminator"](discriminator.parameters(), lr=self.lr_disc, betas=self.betas_d)

        generator, optimizer_g = self.fabric.setup(generator, optimizer_g)
        discriminator, optimizer_d = self.fabric.setup(discriminator, optimizer_d)


        if val_loader is not None:
            self.val_step(val_loader, dict(generator=generator, discriminator=discriminator), do_log=False, sanity_check=True)

        for epoch in range(self.max_epochs):

            train_res = self.train_step(epoch, 
                                        dict(generator=generator, discriminator=discriminator), 
                                        train_loader, 
                                        dict(optimizer_g=optimizer_g, optimizer_d=optimizer_d))

            generator = train_res["generator"]
            discriminator = train_res["discriminator"]
            optimizer_g = train_res["optimizer_g"]
            optimizer_d = train_res["optimizer_d"]
            
            train_loss_gen_per_step = train_res["train_loss_gen_per_step"]
            train_loss_disc_per_step = train_res["train_loss_disc_per_step"]

            self.logger.log({"train_loss_gen_per_epoch": torch.Tensor([np.mean(train_loss_gen_per_step)])})
            self.logger.log({"train_loss_disc_per_epoch": torch.Tensor([np.mean(train_loss_disc_per_step)])})

            val_loss_gen_per_step = None
            val_loss_disc_per_step = None

            if val_loader is not None:
                if (epoch + 1) % self.validate_interval == 0:
                    
                    val_res = self.val_step(val_loader, 
                                            dict(generator=generator, discriminator=discriminator), 
                                            do_log=True, 
                                            sanity_check=False)
                    
                    val_loss_gen_per_step = val_res["val_loss_gen_per_step"]
                    val_loss_disc_per_step = val_res["val_loss_disc_per_step"]

                    self.logger.log({"val_loss_gen_per_epoch": torch.Tensor([np.mean(val_loss_gen_per_step)])})
                    self.logger.log({"val_loss_disc_per_epoch": torch.Tensor([np.mean(val_loss_disc_per_step)])})

            if self.early_stop_callback is not None:
                res = self.early_stop_callback.check(dict(
                    train_loss_gen_per_step = train_loss_gen_per_step,
                    train_loss_disc_per_step = train_loss_disc_per_step,
                    val_loss_gen_per_step = val_loss_gen_per_step,
                    val_loss_disc_per_step = val_loss_disc_per_step,
                    train_loss_gen_per_epoch = np.mean(train_loss_gen_per_step),
                    train_loss_disc_per_epoch = np.mean(train_loss_disc_per_step),
                    val_loss_gen_per_epoch = np.mean(val_loss_gen_per_step) if val_loss_gen_per_step is not None else None,
                    val_loss_disc_per_epoch = np.mean(val_loss_disc_per_step) if val_loss_disc_per_step is not None else None
                ))

                if res:
                    break

            if (epoch + 1) % self.ckpt_interval == 0 and self.ckpt_callback is not None:
                self.ckpt_callback.save(
                        dict(train_loss_gen_per_step = train_loss_gen_per_step, 
                             train_loss_disc_per_step=train_loss_disc_per_step, 
                             val_loss_gen_per_step=val_loss_gen_per_step, 
                             val_loss_disc_per_step=val_loss_disc_per_step, 
                             train_loss_gen_per_epoch = np.mean(train_loss_gen_per_step), 
                             train_loss_disc_per_epoch=np.mean(train_loss_disc_per_step),
                             val_loss_gen_per_epoch = np.mean(val_loss_gen_per_step) if val_loss_gen_per_step is not None else None,
                             val_loss_disc_per_epoch = np.mean(val_loss_disc_per_step) if val_loss_disc_per_step is not None else None),
                        dict(generator=generator, discriminator=discriminator), 
                        dict(generator=optimizer_g, discriminator=optimizer_d), 
                        self.fabric)



    def train_step(self, epoch: int, model: Dict, train_loader: torch.utils.data.DataLoader, optimizer: Dict) -> Dict:

        prog_bar = Progress(
            TextColumn("[progress.percentage] {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=True
        )

        generator = model["generator"]
        discriminator = model["discriminator"]
        optimizer_g = optimizer["optimizer_g"]
        optimizer_d = optimizer["optimizer_d"]

        train_loss_gen_per_step = list()
        train_loss_disc_per_step = list()

        with prog_bar as p:
            
            idx = 0
            for batch in p.track(train_loader, description=f"Epoch: {epoch}/{self.max_epochs - 1}"):
                
                img = batch["img"]
                idx += 1

                ####################################################################################
                ############################# TRAIN DISCRIMINATOR ##################################
                ####################################################################################

                d_losses = list()
                discriminator.train()

                for _ in range(self.n_critic_repeats):
                    
                    optimizer_d.zero_grad()
                    
                    _, fake_img, _ = generator(img)

                    dloss = self._get_disc_loss(img, fake_img, discriminator)

                    d_losses.append(dloss.item())
                    
                    self.fabric.backward(dloss)

                    if idx % self.grad_accum_steps == 0:
                        optimizer_d.step()

                train_loss_disc_per_step.append(np.mean(d_losses))

                ###########################################################################################
                ################################ TRAIN GENERATOR ##########################################
                ###########################################################################################

                generator.train()
                optimizer_g.zero_grad()
                
                gen_loss = self._get_gen_loss(img, generator, discriminator)

                self.fabric.backward(gen_loss)

                if idx % self.grad_accum_steps == 0:
                    optimizer_g.step()

                train_loss_gen_per_step.append(gen_loss.item())

                self.logger.log({"gen_train_loss_step": torch.Tensor([train_loss_gen_per_step[idx-1]])})
                self.logger.log({"disc_train_loss_step": torch.Tensor([train_loss_disc_per_step[idx-1]])})

        
        return {
            "generator" : generator,
            "discriminator": discriminator,
            "optimizer_g": optimizer_g,
            "optimizer_d": optimizer_d,
            "train_loss_gen_per_step": train_loss_gen_per_step,
            "train_loss_disc_per_step": train_loss_disc_per_step
        }

            

    def val_step(self, val_loader: torch.utils.data.DataLoader, model: Dict, do_log: Optional[bool]=True, 
                 sanity_check: Optional[bool]=False) -> Dict:
        
        prog_bar = Progress(
            TextColumn("[progress.percentage] {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=True
        )

        generator = model["generator"]
        discriminator = model["discriminator"]

        val_loss_gen_per_step = list()
        val_loss_disc_per_step = list()

        if sanity_check:
            val_iter = iter(val_loader)
            new_val_loader = [next(val_iter), next(val_iter)]

        else:
            new_val_loader = val_loader

        idx = 0
        with prog_bar as p:

            for batch in p.track(new_val_loader, description="Sanity Checking" if sanity_check else "Validation"):
                
                if idx == 2 and sanity_check:
                    break

                generator.eval()
                discriminator.eval()

                img = batch["img"]
                
                ###########################################################
                ############### DISCRIMINATOR LOSS ########################
                ###########################################################
                
                _, fake_img, _ = generator(img)
                disc_loss = self._get_disc_loss(img, fake_img, discriminator)
                val_loss_disc_per_step.append(disc_loss.item())

                ###############################################################
                ################### GENERATOR LOSS ############################
                ###############################################################
                gen_loss = self._get_gen_loss(img, generator, discriminator)
                val_loss_gen_per_step.append(gen_loss.item())

                if do_log:
                    self.logger.log({"val_loss_gen_per_step": gen_loss.item()})
                    self.logger.log({"val_loss_disc_per_step": disc_loss.item()})

                idx += 1

        return {
            "val_loss_disc_per_step": val_loss_disc_per_step,
            "val_loss_gen_per_step": val_loss_gen_per_step
        }

            






                    



                


        
        
        
        

        
