import torch
import torch.optim as optim
import torch.quantization as tq
from hyperparameters import NUM_CLASSES, GRAYSCALE, NUM_EPOCHS
from model import quantizable_resnet18
from torch.utils.data import DataLoader
import util.dataset_util
import util.model_util
import util.train_util

# Instantiate the quantizable model
model = quantizable_resnet18(NUM_CLASSES, GRAYSCALE)

# Load the pretrained model
pretrained_script_model = torch.jit.load(r'models\resnet_model_trained.pt').to('cpu')
pretrained_state_dict = pretrained_script_model.state_dict()
model.load_state_dict(pretrained_state_dict)

# Set the quantization configuration
quant_config = tq.get_default_qat_qconfig('fbgemm')
model.qconfig = quant_config

# Prepare the model for quantization-aware training
model_prepared = tq.prepare_qat(model)

# Define the loss function and optimizer for training
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model_prepared.parameters(), lr=0.001, momentum=0.9)

# Load the datasets and create data loaders
train_dataset, test_dataset, train_loader, test_loader = util.dataset_util.dataset_load()

device = 'cpu'

# Train the model
model = util.train_util.train_model(model, train_loader, NUM_EPOCHS, device)

# Convert the model to a quantized version
model_quantized = tq.convert(model_prepared)

# Evaluate the quantized model
model_quantized.eval()  # Set model to evaluation mode
with torch.no_grad():
    test_accuracy = util.model_util.compute_accuracy(model_quantized, test_loader, device=device)
    print('Test accuracy: %.2f%%' % test_accuracy)

# Save the quantized model
try:
    util.model_util.save_model_script(model_quantized.half(), r'models\resnet18_trained_QAT.pt')
    print('Quantized model saved')
except:
    print('Error saving quantized model')
