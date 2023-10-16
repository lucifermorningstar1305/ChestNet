"""
@author: Adityam Ghosh
Date: 10-16-2023

"""


from typing import List, Tuple, Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        def build_conv_layers(in_channels: int, out_channels: int, kernel_size: int|Tuple, stride: int|Tuple, padding: int|Tuple) -> Callable:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=.2, inplace=True)
            )


        self.latent_dim = latent_dim

        self.model = nn.Sequential()

        self.model.add_module("layer_1", build_conv_layers(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)) # (112 x 112 x 32)
        self.model.add_module("layer_2", build_conv_layers(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)) # (56 x 56 x 64)
        self.model.add_module("layer_3", build_conv_layers(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)) # (28 x 28 x 128)
        self.model.add_module("layer_4", build_conv_layers(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)) # (14 x 14 x 256)
        self.model.add_module("layer_5", build_conv_layers(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)) # (7 x 7 x 512)

        self.model.add_module("latent_rep", build_conv_layers(in_channels=512, out_channels=latent_dim, kernel_size=3, stride=1, padding=1)) # (7 x 7 x latent_dim)
        self.model.add_module("avg_pool", nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
    

class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        def build_upscale_layers(in_channels: int, out_channels: int, kernel_size: int|Tuple, stride: int|Tuple, 
                                 padding: int|Tuple, is_last: Optional[bool] = False) -> Callable:
            
            if not is_last:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.Tanh()
                )
        

        self.model = nn.Sequential()

        self.model.add_module("latent_rep", build_upscale_layers(in_channels=latent_dim, out_channels=512, kernel_size=9, stride=2, padding=1)) # (7 x 7 x 512)
        self.model.add_module("layer_1_transpose", build_upscale_layers(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)) # (14 x 14 x 256)
        self.model.add_module("layer_2_transpose", build_upscale_layers(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)) # (28 x 28 x 128)
        self.model.add_module("layer_3_transpose", build_upscale_layers(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)) # (56 x 56 x 64)
        self.model.add_module("layer_4_transpose", build_upscale_layers(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)) # (112 x 112 x 32)
        self.model.add_module("layer_5_transpose", build_upscale_layers(in_channels=32, out_channels=1, kernel_size=2, 
                                                                        stride=2, padding=0, is_last=True)) # (224 x 224 x 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self): 
        super().__init__()

        def build_conv_layers(in_channels: int, out_channels: int, kernel_size: int|Tuple, stride: int|Tuple, padding: int|Tuple) -> Callable:

            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
        

        self.model = nn.Sequential()

        self.model.add_module("layer_1", build_conv_layers(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)) # (112 x 112 x 32)
        self.model.add_module("layer_2", build_conv_layers(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)) # (56 x 56 x 64)
        self.model.add_module("layer_3", build_conv_layers(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)) # (28 x 28 x 128)
        self.model.add_module("layer_4", build_conv_layers(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)) # (14 x 14 x 256)
        self.model.add_module("layer_5", build_conv_layers(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)) # (7 x 7 x 512)

        self.model.add_module("avg_pool", nn.AdaptiveAvgPool2d(output_size=(1, 1))) # (1 x 1 x 512)
        self.model.add_module("fc", nn.Linear(in_features=512, out_features=1))

        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.model.layer_1(x)
        x = self.model.layer_2(x)
        x = self.model.layer_3(x)
        x = self.model.layer_4(x)
        x = self.model.layer_5(x)
        avg_pool = self.model.avg_pool(x)
        flatten_x = self.flatten(avg_pool)
        out = self.model.fc(flatten_x)

        return out, avg_pool
    



class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder1 = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.encoder2 = Encoder(latent_dim=latent_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        z = self.encoder1(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)

        return z, x_hat, z_hat

