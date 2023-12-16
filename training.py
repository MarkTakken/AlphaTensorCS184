#Training is meant to control the training loop

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from architecture import *
from tqdm import tqdm

def loss_fn(pred_logits, true_tokens, pred_value, true_value, val_weight=.33, device = 'cuda'):
    # Computes the loss for the model. Is a combination of policy loss and value loss
    policy_loss = nn.functional.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]), true_tokens.flatten().type(torch.LongTensor).to(device))
    value_loss = ((pred_value - true_value)**2).mean()
    return policy_loss + val_weight*value_loss

def loss_reporter(pred_logits, true_tokens, pred_value, true_value, val_weight=1.0, device = 'cuda'):
    # Computes the loss for the model. Is a combination of policy loss and value loss. Returns each instead of together
    policy_loss = nn.functional.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]), true_tokens.flatten().type(torch.LongTensor).to(device))
    value_loss = ((pred_value - true_value)**2).mean()
    return policy_loss.to('cpu').item(), value_loss.to('cpu').item()

def train(model, dataset, epochs, batch_size = 1024, lr=0.001, val_weight = .33, device = 'cuda'):
    # Trains the model on the dataset for the specified number of epochs. Arguments:
    # model as a nn.Module, which is the model to train
    # dataset as a torch.utils.data.Dataset, which is the dataset to train on
    # epochs as an int, which is the number of epochs to train for
    # batch_size as an int, which is the batch size for training
    # lr as a float, which is the learning rate
    # val_weight as a float, which is the weight of the value loss in the total loss
    # device as a string, which is the device to use for the model

    #Returns a list of losses. Applies training directly to the model

    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []
    pl = []
    vl = []
    for epoch in range(epochs):
        running_loss = 0.0
        p = 0
        v = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            states, actions, values = batch
            states = states.to(device).float()
            actions = actions.long().to(device)
            values = values.to(device).float()
            pred_value, pred_logits = model(states, g=actions)
            loss = loss_fn(pred_logits, actions, pred_value, values, val_weight=val_weight, device=device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            lp, lv  = loss_reporter(pred_logits, actions, pred_value, values, val_weight=val_weight, device=device)
            p += lp
            v += lv
        losses.append(running_loss/len(dataloader))
        pl.append(p/len(dataloader))
        vl.append(v/len(dataloader))
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')
    return losses, pl, vl