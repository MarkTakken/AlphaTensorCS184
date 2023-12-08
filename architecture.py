## Architecture of the model

import torch
from torch import nn

class Attention(nn.Module):
    
    def __init__(self, Nx, c1, Ny, c2, causal_mask = False, N_heads = 16, w = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm([Nx, c1])
        self.ln2 = nn.LayerNorm([Ny, c2])
        if not causal_mask:
            self.MAH = nn.MultiheadAttention(embed_dim=c1, kdim=c2, vdim=c2, num_heads=N_heads)
        else:
            mask = torch.triu(torch.ones((Nx, Ny)))
            self.MAH = nn.MultiheadAttention(embed_dim=c1, kdim=c2, vdim=c2, num_heads=N_heads, attn_mask=mask)
        self.ln3 = nn.LayerNorm([Nx, c1])
        self.l1 = nn.Linear(c1, c1*w)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(c1*w, c1)

    def forward(self, x, y):
        xn = self.ln1(x)
        yn = self.ln2(y)
        attn, _ = self.MAH(xn, yn, yn)
        x = x + attn
        x = x + self.l2(self.gelu(self.l1(self.ln3(x))))
        return x

class AttentiveModes(nn.Module):
    def __init__(self, s, c):
        super().__init__()
        self.attention = Attention(2 * s, c, 2 * s, c, N_heads = 8)
        self.s = s
        self.c = c

    def forward(self, x1, x2, x3):
        g = [x1, x2, x3]
        for m1, m2 in [(0, 1), (2, 0), (1, 2)]:
            a = torch.concatenate((g[m1], torch.transpose(g[m2], 0, 1)), axis=1)
            for i in range(self.s):
                c = self.attention(a[i, :, :], a[i, :, :])
                g[m1][i, :, :] = c[:self.s, :]
                g[m2][:, i, :] = c[self.s:, :]
        return g



class Torso(nn.Module):
    def __init__(self, s, c, i):
        super().__init__()
        self.l1 = nn.Linear(s, c)
        self.attentive_modes = nn.ModuleList([AttentiveModes(s, c) for _ in range(i)])
        self.s = s
        self.c = c
        self.i = i

    def forward(self, x):
        x1 = torch.permute(x, (0, 1, 2))
        x2 = torch.permute(x, (1, 2, 0))
        x3 = torch.permute(x, (2, 0, 1))

        x1 = self.l1(x1)
        x2 = self.l1(x2)
        x3 = self.l1(x3)
        
        for am in self.attentive_modes:
            x1, x2, x3 = am(x1, x2, x3)

        e = torch.reshape(torch.stack([x1, x2, x3], axis=1), (3 * self.s ** 2, self.c))
## Before implementing heads, read up on Torch head/transformer modules and how they work further. Unclear to me if their transformers do what we want. 
## Also, need to be careful with setting up training vs acting

class AlphaTensor184(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass
    
        