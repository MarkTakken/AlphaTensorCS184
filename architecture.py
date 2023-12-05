## Architecture of the model

import torch
from torch import nn

class Attention(nn.Module):
    # Can we just use nn.MultiheadAttention()?
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class AttentiveModes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class Torso(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

## Before implementing heads, read up on Torch head/transformer modules and how they work further. Unclear to me if their transformers do what we want. 
## Also, need to be careful with setting up training vs acting

class AlphaTensor184(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass
    
        