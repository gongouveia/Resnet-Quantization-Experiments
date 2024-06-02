


import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from util.model_util import compute_accuracy

from  hyperparameters import LEARNING_RATE




def train_model(model, data,  num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    start_time = time.time()
    for epoch in range(num_epochs):
        
        model.train()
        for batch_idx, (features, targets) in enumerate(data):
            
            features = features.to(device)
            targets = targets.to(device)
                
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
                    %(epoch+1, num_epochs, batch_idx, 
                        len(data), cost))

            

        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                epoch+1, num_epochs, 
                compute_accuracy(model, data, device=device)))
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    return model