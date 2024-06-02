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

import util.train_util


if torch.cuda.is_available():
    print('CUDA available')
    torch.backends.cudnn.deterministic = True

torch.manual_seed(RANDOM_SEED)

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


model = resnet18(NUM_CLASSES, True)
model.to(DEVICE)


model = util.train_util.train_model(model, train_loader, NUM_EPOCHS, DEVICE)

with torch.no_grad(): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))

# model.eval()
# logits, probas = model(features.to(device)[0, None])



with torch.no_grad():
    scripted_model = torch.jit.script(model)
    scripted_model.save('models/resnet18_trained_f32.pt')
