"""
@author: Adityam Ghosh
Date: 10-16-2023

"""

import torch

def init_weights(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.)