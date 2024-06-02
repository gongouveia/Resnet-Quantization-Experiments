import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


from util.dataset_util import dataset_load
from util.model_util import compute_accuracy

from hyperparameters import NUM_CLASSES, RANDOM_SEED, DEVICE, LEARNING_RATE, NUM_EPOCHS, NUM_FEATURES, BATCH_SIZE

from model import resnet18


if torch.cuda.is_available():
    print('CUDA available')
    torch.backends.cudnn.deterministic = True


train_dataset, test_dataset, train_loader, test_loader = dataset_load()
device = torch.device(DEVICE)

for epoch in range(2):
    for batch_idx, (x, y) in enumerate(train_loader):
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)
        break

torch.manual_seed(RANDOM_SEED)

model = resnet18(NUM_CLASSES, True)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 


start_time = time.time()
for epoch in range(1):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, NUM_EPOCHS, 
              compute_accuracy(model, train_loader, device=DEVICE)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))




with torch.no_grad(): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))

model.eval()
logits, probas = model(features.to(device)[0, None])



with torch.no_grad():
    scripted_model = torch.jit.script(model)
    scripted_model.save('models/resnet18_trained_f32.pt')
