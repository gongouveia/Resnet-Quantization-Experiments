import torch
import torch.optim as optim
import torch.quantization as tq
import util.model_util as model_util
from hyperparameters import NUM_CLASSES, GRAYSCALE, NUM_EPOCHS, BATCH_SIZE
from model import quantizable_resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Instantiate quantizable model
model = quantizable_resnet18(NUM_CLASSES, GRAYSCALE)

# Load pretrained model
pretrained_script_model = torch.jit.load(r'models\resnet_model_trained.pt')
pretrained_script_model.to('cpu')
pretrained_state_dict = pretrained_script_model.state_dict()
model.load_state_dict(pretrained_state_dict)

# Set quantization configuration
quant_config = tq.get_default_qat_qconfig('fbgemm')
model.qconfig = quant_config

# Prepare the model for quantization-aware training
model_prepared = tq.prepare_qat(model)

# Define loss function and optimizer for training
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model_prepared.parameters(), lr=0.001, momentum=0.9)

# Data preparation
train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = 'cpu'

import torch.nn.functional as F


for epoch in range(NUM_EPOCHS):
    
    model_prepared.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
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
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

        


# Convert the model to a quantized version
model_quantized = tq.convert(model_prepared)

# Evaluate the quantized model
model_quantized.eval()  # Set model to evaluation mode
with torch.no_grad():
    print('Test accuracy: %.2f%%' % (model_util.compute_accuracy(model_quantized, test_loader, device='cpu')))

# Save the quantized model
try:
    model_util.save_model_script(model_quantized.half(), r'models\resnet_trained_QAT.pt')
    print('Quantized model saved')
except:
    print('Error saving quantized model')
