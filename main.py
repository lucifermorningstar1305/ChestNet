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
import lightning.pytorch as pl
import argparse

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger

from scripts.callbacks import CustomEarlyStopping, CustomModelCheckpoint, GenerateCallbacks
from scripts.make_dataloader import AnomalyDataLoader, NormalDataLoader
from scripts.train_script_anomaly import CustomTrainer
from scripts.train_classifier import LitClassifier

from models.anomaly_models import Generator, Discriminator
from models.classifier_model import Classifier

L.seed_everything(32)
torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--objective", "-o", required=True, type=str, help="the objective anomaly/classification")
    parser.add_argument("--csv_path", "-p", required=True, type=str, help="the csv path for the tabular data")
    parser.add_argument("--latent_dim", "-l", required=False, type=int, default=100, help="the latent dimension")
    parser.add_argument("--lr", "-L", required=False, type=float, default=1e-3, help="the learning rate")
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
    lr = args.lr

    df = pd.read_csv(csv_path)

    if objective == "anomaly":
        wandb.login()
        run = wandb.init(project="GANomaly_ChestNet", name="ede_disc")

        norm_dataloader_obj = NormalDataLoader(normal_label="No Finding", val_size=.2, 
                                                batch_sizes={"train":train_batch_size, "val":val_batch_size})
        
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        norm_dataloader_obj.setup(df, transformations=dict(train=train_transforms, val=val_transforms))

        train_ds, train_dl = norm_dataloader_obj.get_train()
        val_ds, val_dl = norm_dataloader_obj.get_val()

        generator = Generator(latent_dim=latent_dim)
        discriminator = Discriminator()

        optimizer_g = torch.optim.Adam
        optimizer_d = torch.optim.Adam

        early_stopping = CustomEarlyStopping(monitor="val_loss_gen_per_epoch", direction="min", patience=10, verbose=True)
        model_chkpt = CustomModelCheckpoint(monitor="val_loss_gen_per_epoch", direction="min", chkpt_dir=checkpoint_dir, 
                                            chkpt_fn=checkpoint_file, verbose=True)
        
        gen_callback = GenerateCallbacks(val_ds)
        
        train_cfg = {
            "lr_g": lr,
            "lr_d": lr,
            "betas_g": (.5, .999),
            "betas_d": (.5, .999)

        }
        trainer = CustomTrainer(train_cfg=train_cfg, 
                                accelerator="cuda", 
                                max_epochs=max_epochs, 
                                precision=16, 
                                logger=run, 
                                callbacks={"early_stop_callback": early_stopping, "checkpoint_callback": model_chkpt, "gen_callback": gen_callback})
        
        trainer.fit(dict(generator=generator, discriminator=discriminator), 
                    train_dl, 
                    dict(generator=optimizer_g, discriminator=optimizer_d), 
                    val_loader=val_dl)
        
    else:
        anom_data_obj = AnomalyDataLoader("No Finding", val_size=.2, 
                                          batch_sizes=dict(train=train_batch_size, val=val_batch_size))
        

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

        anom_data_obj.setup(df, transformations=dict(train=train_transforms, val=val_transforms))

        train_ds, train_dl = anom_data_obj.get_train()
        val_ds, val_dl = anom_data_obj.get_val()

        n_classes = anom_data_obj.n_classes

        classifier = Classifier(n_classes=n_classes)
        lit_classifier = LitClassifier(classifier, n_classes=n_classes, lr=lr)

        early_stopping = EarlyStopping(monitor="val_loss", 
                                       mode="min", 
                                       patience=10, 
                                       verbose=True)
        
        model_checkpoint = ModelCheckpoint(monitor="val_loss", 
                                           mode="min", 
                                           dirpath=checkpoint_dir, 
                                           filename=checkpoint_file, 
                                           save_on_train_epoch_end=False)
        
        rich_prog_bar = RichProgressBar()
        callbacks =  [early_stopping, model_checkpoint, rich_prog_bar]
        logger = WandbLogger(name="resnet18", project="Anomaly_Classification_ChestNet")

        trainer = pl.Trainer(accelerator="cuda", 
                             devices=1, 
                             precision=16, 
                             callbacks=callbacks, 
                             logger=logger, 
                             max_epochs=max_epochs)
        
        trainer.fit(lit_classifier, train_dataloaders=train_dl, val_dataloaders=val_dl)
        




        
    