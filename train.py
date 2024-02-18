import math
import torch
import torch.nn as nn
from torch.nn.utils.prune import l1_unstructured, random_unstructured, global_unstructured, L1Unstructured, RandomUnstructured
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn.functional as F


def train_step(model, train_data, optimizer, loss,  device, lr = 1.2 * 1e-3):
    
    x, Y = train_data
    x, Y = x.to(device), Y.to(device)
    optimizer = optimizer(model.parameters(), lr)
    logit_batch = model.forward(x)
    loss = loss(logit_batch, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return torch.sum(logit_batch.argmax(dim=-1) == Y)

def evaluate(model, xy_batch, device):
    
    with torch.no_grad():
        
        x, Y = xy_batch
        x, Y = x.to(device), Y.to(device)           
        n_samples = Y.shape
        x_pred = model(x).argmax(dim=-1)
        return torch.sum(x_pred == Y) 

def train_epoch(model, dataloader, optimizer, loss, device, lr =  1.2 * 1e-3):


    train_correct_pred = 0
    
    for batch in dataloader:
        train_correct_pred += train_step(model, batch, optimizer, loss, device = device, lr = lr)
        
    epoch_acc = train_correct_pred/len(dataloader.dataset)
    print("Train accuracy this epoch = {0}".format(epoch_acc))

def val_epoch(model, dataloader, device, loss):
    val_correct_pred = 0
    val_loss = 0
    for batch in dataloader:
        with torch.no_grad():
        
            x, Y = batch
            x, Y = x.to(device), Y.to(device)           
            n_samples = Y.shape
            logit_batch = model.forward(x)
            x_pred = logit_batch.argmax(dim=-1)
            val_loss += loss(logit_batch, Y)*Y.size(0)
            val_correct_pred += torch.sum(x_pred == Y) 
        
    epoch_val_acc = val_correct_pred/len(dataloader.dataset)
    epoch_loss = val_loss/len(dataloader.dataset)
    print("Validation accuracy this epoch = {0}".format(epoch_val_acc))
    print("Validation loss this epoch = {0}".format(epoch_loss))


def train(model, dataloader_train, dataloader_val, device, loss, optimizer, epochs = 10, lr =  1.2 * 1e-3):

    for iteration in range(epochs):
        train_epoch(model, dataloader_train, optimizer, loss, device, lr)
        val_epoch(model, dataloader_val, device, loss)
    
        
 