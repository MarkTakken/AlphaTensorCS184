#Training is meant to control the training loop

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from architecture import *
from tqdm import tqdm

def loss_fn(pred_logits, true_tokens, pred_value, true_value, val_weight=1.0, device = 'cuda'):  # Ok to remove 'cuda' parameter?
    policy_loss = nn.functional.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]), true_tokens.flatten().type(torch.LongTensor).to(device))
    value_loss = ((pred_value - true_value)**2).mean()
    return policy_loss + val_weight*value_loss

def train(model, dataset, epochs, batch_size = 1024, lr=0.001, device = 'cuda'):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            states, actions, values = batch
            states = states.to(device).float()
            actions = actions.long().to(device)
            values = values.to(device).float()
            pred_value, pred_logits = model(states, g=actions)
            loss = loss_fn(pred_logits, actions, pred_value, values)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')