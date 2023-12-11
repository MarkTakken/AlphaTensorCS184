## Architecture of the model

import torch
from torch import nn
import utilities
from collections import namedtuple

## Adding batching?  Mark: Done.
## Mark: I think the LayerNorm should be on the last axis only, so I've changed
##      that, but kept the original shape in comments on the side just in case

class Attention(nn.Module):
    
    def __init__(self, c1, c2, causal_mask = False, N_heads = 16, w = 4, device = torch.device('cuda')):
        super().__init__()
        self.ln1 = nn.LayerNorm([c1])  # [Nx, c1]
        self.ln2 = nn.LayerNorm([c2])  # [Ny, c2]
        self.causal_mask = causal_mask
        self.MAH = nn.MultiheadAttention(embed_dim=c1, kdim=c2, vdim=c2, num_heads=N_heads, batch_first=True)
        self.ln3 = nn.LayerNorm([c1])  # [Nx, c1]
        self.l1 = nn.Linear(c1, c1*w)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(c1*w, c1)
        self.device = device

    def forward(self, x, y):
        xn = self.ln1(x)
        yn = self.ln2(y)
        if self.causal_mask:
            mask = torch.triu(torch.ones(x.shape[1], y.shape[1], dtype=bool), diagonal=1).to(self.device)
            attn = self.MAH(xn, yn, yn, attn_mask = mask)[0]
        else:
            attn = self.MAH(xn, yn, yn)[0]
        x = x + attn
        x = x + self.l2(self.gelu(self.l1(self.ln3(x))))
        return x

class AttentiveModes(nn.Module):
    def __init__(self, s, c):
        super().__init__()
        self.attention = Attention(c, c, N_heads = 8)
        self.s = s
        self.c = c

    def forward(self, x1, x2, x3):
        g = [x1, x2, x3]
        for m1, m2 in [(0, 1), (2, 0), (1, 2)]:
            a = torch.concatenate((g[m1], torch.transpose(g[m2], 1, 2)), axis=2)
            aflat = a.flatten(0,1)
            c = self.attention(aflat, aflat).reshape_as(a)
            g[m1] = c[:, :, :self.s, :]
            g[m2] = c[:, :, self.s:, :].transpose(1,2)
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
        x1 = torch.permute(x, (0, 1, 2, 3))
        x2 = torch.permute(x, (0, 2, 3, 1))
        x3 = torch.permute(x, (0, 3, 1, 2))

        x1 = self.l1(x1)
        x2 = self.l1(x2)
        x3 = self.l1(x3)
        
        for am in self.attentive_modes:
            x1, x2, x3 = am(x1, x2, x3)

        e = torch.reshape(torch.stack([x1, x2, x3], axis=2), (-1, 3 * self.s ** 2, self.c))
        return e

class PolicyHead(nn.Module):
    # Currently assumes our implemented tokenization scheme
    # That is, Nstesp = s and Nlogits = range^3
    def __init__(self, Nsteps, elmnt_range, s, c, Nfeatures = 64, Nheads = 16, Nlayers = 2, device = torch.device('cuda')):
        super().__init__()
        self.Nlayers = Nlayers
        self.Nlogits = (elmnt_range[1]-elmnt_range[0]+1)**3
        self.tokenizer = utilities.Tokenizer(elmnt_range)
        self.Nsteps = Nsteps
        self.Nfeatures = Nfeatures
        self.Nheads = Nheads
        self.device = device

        self.tok_embedding = nn.Embedding(self.Nlogits+1, Nfeatures * Nheads)  #In principle more efficient than forming one-hot vectors and matrix multplying
        self.START_TOK = self.Nlogits
        self.pos_embedding = nn.Embedding(Nsteps, Nfeatures * Nheads)

        # I figure if we are keeping the weights in the LayerNorm, we might as well have
        #   a different one for each layer, but idk really
        self.ln1 = nn.ModuleList([nn.LayerNorm([Nfeatures * Nheads]) for _ in range(Nlayers)])  # [Nsteps, Nfeatures * Nheads]
        self.dropout = nn.Dropout(p=0.1)
        self.self_attention = nn.ModuleList([Attention(Nfeatures * Nheads, Nfeatures * Nheads, causal_mask=True, N_heads=Nheads) for _ in range(Nlayers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm([Nfeatures * Nheads]) for _ in range(Nlayers)])
        self.cross_attention = nn.ModuleList([Attention(Nfeatures * Nheads, c, N_heads=Nheads) for _ in range(Nlayers)])
        
        self.relu = nn.ReLU()
        self.lfinal = nn.Linear(Nfeatures * Nheads, self.Nlogits)

    def predict_logits(self, a, e):   # Assumes a is in tokenized, not one-hot form
        x = self.tok_embedding(a)
        positions = torch.arange(a.shape[1]).repeat((a.shape[0], 1)).to(self.device)
        x = x + self.pos_embedding(positions)
        for i in range(self.Nlayers):
            x = self.ln1[i](x)
            c = self.self_attention[i](x, x)
            c = self.dropout(c)  # Does not run if in evaluation mode
            x = x + c
            x = self.ln2[i](x)
            c = self.cross_attention[i](x, e)
            c = self.dropout(c)
            x = x + c
        o = self.lfinal(self.relu(x))
        return o    # Don't need x bc we are not feeding it to the value head
    
    def forward(self, e, **kwargs):
        if self.training:
            g = kwargs['g']
            #I'm not entirely sure this is right -- need to think on tokens and what the null character is
            #g = torch.cat((torch.tensor([0]), g))
            #Not working at the moment, going to stick with this and not shifting, but maybe there's a shift or something needed?
            # a = nn.functional.one_hot(g, self.Nlogits).float()
            # o, z = self.predict_logits(a, e)
            # return o, z
            a = torch.concatenate((torch.tensor(self.START_TOK).repeat(g.shape[0], 1), g[:, :-1]), axis=1).to(self.device)
            return self.predict_logits(a, e)
        
        else:
            Nsamples = kwargs['Nsamples']
            #a = torch.zeros((Nsamples, self.Nsteps)).long()
            a = [[self.START_TOK] for _ in range(Nsamples)]
            p = torch.ones(Nsamples)
            #z = torch.zeros((Nsamples, self.Nsteps, self.Nfeatures * self.Nheads))
            #Don't care about exporting Z anymore
            for j in range(Nsamples):
                for i in range(self.Nsteps):
                    # encoded = nn.functional.one_hot(a[j, :], self.Nlogits)
                    # o, _ = self.predict_logits(encoded.float(), e)
                    o = self.predict_logits(torch.tensor([a[j]]).to(self.device), e)
                    probs = torch.softmax(o[0, i, :], -1).to('cpu')
                    tok = torch.multinomial(probs, num_samples=1).item()
                    a[j].append(tok)
                    p[j] *= probs[tok]
            
            actions = self.tokenizer.batch_detokenize(torch.tensor(a)[:,1:])
            probs = p/p.sum()

            return namedtuple('Policy', 'actions probs')(actions, probs)
            

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
        x = torch.mean(x, axis=1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.lf(x)
        return x


## Before implementing heads, read up on Torch head/transformer modules and how they work further. Unclear to me if their transformers do what we want. 
## Also, need to be careful with setting up training vs acting

class AlphaTensor184(nn.Module):
    def __init__(self, s, c, d, elmnt_range, Nsteps, Nsamples, torso_iterations = 8):
        super().__init__()
        self.s = s
        self.c = c
        self.Nlogits = elmnt_range[1]-elmnt_range[0]+1
        self.Nsteps = Nsteps
        self.Nsamples = Nsamples
        
        self.torso = Torso(s, c, torso_iterations)
        self.value_head = ValueHead(c, d) 
        self.policy_head = PolicyHead(Nsteps, elmnt_range, s, c)
    
    def forward(self, x, g=None):
        e = self.torso(x)
        q = self.value_head(e)
        if g == None:  # Inference
            assert(not(self.training))
            policy = self.policy_head(e, Nsamples=self.Nsamples)
            return (q, policy)
        else: # Training
            assert(self.training)
            logits = self.policy_head(e, g=g)
            return (q, logits)