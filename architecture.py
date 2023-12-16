## Architecture of the model

import torch
from torch import nn
import utilities
from collections import namedtuple

## Adding batching?  Mark: Done.
## Mark: I think the LayerNorm should be on the last axis only, so I've changed
##      that, but kept the original shape in comments on the side just in case

class Attention(nn.Module):
    def __init__(self, c1, c2, causal_mask = False, N_heads = 16, w = 4, device = 'cuda'):
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
    def __init__(self, s, c, device = 'cuda'):
        super().__init__()
        self.device = device
        self.attention = Attention(c, c, N_heads = 8, device=device)
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
    def __init__(self, s, c, i, device = 'cuda'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(s, c)
        self.attentive_modes = nn.ModuleList([AttentiveModes(s, c, device=device) for _ in range(i)])
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
    def __init__(self, Nsteps, elmnt_range, s, c, Nfeatures = 64, Nheads = 16, Nlayers = 2, device = 'cuda'):
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
        self.self_attention = nn.ModuleList([Attention(Nfeatures * Nheads, Nfeatures * Nheads, causal_mask=True, N_heads=Nheads, device=device) for _ in range(Nlayers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm([Nfeatures * Nheads]) for _ in range(Nlayers)])
        self.cross_attention = nn.ModuleList([Attention(Nfeatures * Nheads, c, N_heads=Nheads, device=device) for _ in range(Nlayers)])
        
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
            a = torch.concatenate((torch.tensor(self.START_TOK).repeat(g.shape[0], 1).to(self.device), g[:, :-1].to(self.device)), axis=1).to(self.device)
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

## Mark: Removed duplication?

class AlphaTensor184(nn.Module):
    ## The main assembly of the model
    def __init__(self, s, c, d, elmnt_range, Nsteps, Nsamples, N_policy_features = 48, N_policy_heads = 12, torso_iterations = 8, device = 'cuda'):
        ## Takes arguments to build the neural network. Arguments:
        ## s as an int, which is the dimension of the cubic tensor
        ## c as an int, which is the number of features in the torso
        ## d as an int, which is the number of dimensions in the value head
        ## elmnt_range as a tuple[int, int], which is the range of elements in the actions
        ## Nsteps as an int, which is the number of steps of attention required in the policy head
        ## Nsamples as an int, which is the number of sampled actions to take in the policy head
        ## N_policy_features as an int, which is the number of features used in the attentive methods in the policy head
        ## N_policy_heads as an int, which is the number of heads used in the attentive methods in the policy head
        ## torso_iterations as an int, which is the number of iterations of the attentive methods in the torso
        ## device as a string, which is the device to use for the model

        super().__init__()
        self.s = s
        self.c = c
        self.Nlogits = elmnt_range[1]-elmnt_range[0]+1
        self.Nsteps = Nsteps
        self.Nsamples = Nsamples
        
        self.torso = Torso(s, c, torso_iterations, device=device)
        self.value_head = ValueHead(c, d) 
        self.policy_head = PolicyHead(Nsteps, elmnt_range, s, c, Nfeatures=N_policy_features, Nheads=N_policy_heads,  device=device)
    
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