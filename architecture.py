## Architecture of the model

import torch
from torch import nn


## Adding batching?

class Attention(nn.Module):
    
    def __init__(self, Nx, c1, Ny, c2, causal_mask = False, N_heads = 16, w = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm([Nx, c1])
        self.ln2 = nn.LayerNorm([Ny, c2])
        self.causal_mask = causal_mask
        if not causal_mask:
            self.MAH = nn.MultiheadAttention(embed_dim=c1, kdim=c2, vdim=c2, num_heads=N_heads)
        else:
            self.mask = torch.triu(torch.ones((Nx, Ny)))
            self.MAH = nn.MultiheadAttention(embed_dim=c1, kdim=c2, vdim=c2, num_heads=N_heads)
        self.ln3 = nn.LayerNorm([Nx, c1])
        self.l1 = nn.Linear(c1, c1*w)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(c1*w, c1)

    def forward(self, x, y):
        xn = self.ln1(x)
        yn = self.ln2(y)
        if self.causal_mask:
            attn, _ = self.MAH(xn, yn, yn, attn_mask=self.mask)
        else:
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

class PolicyHead(nn.Module):
    def __init__(self, Nsteps, Nlogits, s, c, Nfeatures = 64, Nheads = 16, Nlayers = 2):
        super().__init__()
        self.Nlayers = Nlayers
        self.Nlogits = Nlogits
        self.Nsteps = Nsteps
        self.Nfeatures = Nfeatures
        self.Nheads = Nheads


        self.l1 = nn.Linear(Nlogits, Nfeatures * Nheads)
        self.ln = nn.LayerNorm([Nsteps, Nfeatures * Nheads])
        self.lookup = nn.Parameter(torch.empty((Nsteps, Nfeatures * Nheads)))
        nn.init.normal_(self.lookup, mean=0, std=1)

        self.ln = nn.LayerNorm([Nsteps, Nfeatures * Nheads])
        self.dropout = nn.Dropout(p=0.1)
        self.self_attention = nn.ModuleList([Attention(Nsteps, Nfeatures * Nheads, Nsteps, Nfeatures * Nheads, causal_mask=True, N_heads=Nheads) for _ in range(Nlayers)])
        self.cross_attention = nn.ModuleList([Attention(Nsteps, Nfeatures * Nheads, 3 * s ** 2, c, N_heads=Nheads) for _ in range(Nlayers)])
        
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(Nfeatures * Nheads, Nlogits)

    def predict_logits(self, a, e):
        x = self.l1(a)
        # x  = x + Learnable Position Encoding

        for i in range(self.Nlayers):
            x = self.ln(x)
            c = self.self_attention[i](x, x)
            c = self.dropout(c)
            x = x + c
            x = self.ln(x)
            c = self.cross_attention[i](x, e)
            c = self.dropout(c)
            x = x + c
        o = self.l2(self.relu(x))
        return o, x
    
    def forward(self, e, **kwargs):
        if self.training:
            g = kwargs['g']
            #I'm not entirely sure this is right -- need to think on tokens and what the null character is
            #g = torch.cat((torch.tensor([0]), g))
            #Not working at the moment, going to stick with this and not shifting, but maybe there's a shift or something needed?
            a = nn.functional.one_hot(g, self.Nlogits).float()
            o, z = self.predict_logits(a, e)
            return o, z
        
        else:
            Nsamples = kwargs['Nsamples']
            a = torch.zeros((Nsamples, self.Nsteps)).long()
            p = torch.ones(Nsamples)
            #z = torch.zeros((Nsamples, self.Nsteps, self.Nfeatures * self.Nheads))
            #Don't care about exporting Z anymore
            for j in range(Nsamples):
                for i in range(self.Nsteps):
                    encoded = nn.functional.one_hot(a[j, :], self.Nlogits)
                    o, _ = self.predict_logits(encoded.float(), e)
                    probs = torch.softmax(o[i, :], 0)
                    a[j, i] = torch.multinomial(probs, num_samples=1)
                    p = p * probs[a[j, i]]

            return a, p
            
        


class ValueHead(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c = c
        self.d = d
        
        self.l1 = nn.Linear(c, d)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(d, d)
        self.l3 = nn.Linear(d, d)
        self.lf = nn.Linear(d, 1)

    def forward(self, x):
        x = torch.mean(x, axis=0)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.lf(x)
        return x


## Before implementing heads, read up on Torch head/transformer modules and how they work further. Unclear to me if their transformers do what we want. 
## Also, need to be careful with setting up training vs acting

class AlphaTensor184(nn.Module):
    def __init__(self, s, c, d, Nlogits, Nsteps, Nsamples, torso_iterations = 8):
        super().__init__()
        self.s = s
        self.c = c
        self.Nlogits = Nlogits
        self.Nsteps = Nsteps
        self.Nsamples = Nsamples
        
        self.torso = Torso(s, c, torso_iterations)
        self.value_head = ValueHead(c, d) 
        self.policy_head = PolicyHead(Nsteps, Nlogits, s, c)
    
    def forward(self, x):
        if self.training:
            pass
        else:
            pass
    
        