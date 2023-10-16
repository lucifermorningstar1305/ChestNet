"""
@author: Adityam Ghosh
Date: 10-16-2023

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import wandb
import lightning as L
import argparse

from scripts.callbacks import CustomEarlyStopping, CustomModelCheckpoint
from scripts.make_dataloader import AnomalyDataLoader
from scripts.train_script_anomaly import CustomTrainer

from models.anomaly_models import Generator, Discriminator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--objective", "-o", required=True, type=str, help="the objective anomaly/classification")
    parser.add_argument("--csv_path", "-p", required=True, type=str, help="the csv path for the tabular data")
    parser.add_argument("--latent_dim", "-l", required=False, type=int, default=100, help="the latent dimension")
    parser.add_argument("--train_batch_size", "-B", required=False, default=32, type=int, help="training batch size")
    parser.add_argument("--val_batch_size", "-b", required=False, default=64, type=int, help="validation batch size")
    parser.add_argument("--checkpoint_dir", "-C", required=False, default="./checkpoints", type=str, help="model checkpoint directory")
    parser.add_argument("--checkpoint_file", "-c", required=False, default="checkpoint", type=str, help="model checkpoint filename")
    parser.add_argument("--max_epochs", "-E", required=False, default=100, type=int, help="the max number of epochs")

    args = parser.parse_args()

    objective = args.objective
    csv_path = args.csv_path
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    checkpoint_dir = args.checkpoint_dir
    checkpoint_file = args.checkpoint_file
    latent_dim = args.latent_dim
    max_epochs = args.max_epochs

    if objective == "anomaly":
        wandb.login()
        run = wandb.init(project="GANomaly_ChestNet", name="ede_disc")
        
        df = pd.read_csv(csv_path)
        anom_dataloader_obj = AnomalyDataLoader(normal_label="No Finding", val_size=.2, 
                                                batch_sizes={"train":train_batch_size, "val":val_batch_size})
        
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        anom_dataloader_obj.setup(df, transformations=dict(train=train_transforms, val=val_transforms))

        train_ds, train_dl = anom_dataloader_obj.get_train()
        val_ds, val_dl = anom_dataloader_obj.get_val()

        generator = Generator(latent_dim=latent_dim)
        discriminator = Discriminator()
        optimizer_g = torch.optim.Adam
        optimizer_d = torch.optim.Adam

        early_stopping = CustomEarlyStopping(monitor="val_loss_gen_per_epoch", direction="min", patience=10, verbose=True)
        model_chkpt = CustomModelCheckpoint(monitor="val_loss_gen_per_epoch", direction="min", chkpt_dir=checkpoint_dir, 
                                            chkpt_fn=checkpoint_file, verbose=True)
        
        trainer = CustomTrainer(accelerator="cuda", max_epochs=max_epochs, precision=16, logger=run, 
                                callbacks={"early_stop_callback": early_stopping, "checkpoint_callback": model_chkpt})
        
        trainer.fit(dict(generator=generator, discriminator=discriminator), 
                    train_dl, 
                    dict(generator=optimizer_g, discriminator=optimizer_d), 
                    val_loader=val_dl)
        
    