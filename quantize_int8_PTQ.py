import util.model_util as model_util
import util.dataset_util 

import torch


from hyperparameters import  NUM_CLASSES, GRAYSCALE
from model import quantizable_resnet18, resnet18
import torch.quantization as tq



try:

    pretrained_script_model = torch.jit.load(r'models\resnet_model_trained.pt')
    pretrained_script_model.to('cpu')

    # Extract the state dictionary from the TorchScript model
    pretrained_state_dict = pretrained_script_model.state_dict()
except:
    print('Error loading pre trained model')
# Prepare the quantizable model
model = quantizable_resnet18(NUM_CLASSES, GRAYSCALE)

# Load the state dictionary into the quantizable model
model.load_state_dict(pretrained_state_dict)


# Set the model to evaluation mode
model.eval()

# Fuse Conv, BN, and ReLU layers
model_fused = torch.quantization.fuse_modules(model, 
                                              [['conv1', 'bn1', 'relu'],
                                               ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'],
                                               ['layer1.0.conv2', 'layer1.0.bn2'],
                                               ['layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu'],
                                               ['layer1.1.conv2', 'layer1.1.bn2'],
                                               ['layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu'],
                                               ['layer2.0.conv2', 'layer2.0.bn2'],
                                               ['layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu'],
                                               ['layer2.1.conv2', 'layer2.1.bn2'],
                                               ['layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu'],
                                               ['layer3.0.conv2', 'layer3.0.bn2'],
                                               ['layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu'],
                                               ['layer3.1.conv2', 'layer3.1.bn2'],
                                               ['layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu'],
                                               ['layer4.0.conv2', 'layer4.0.bn2'],
                                               ['layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu'],
                                               ['layer4.1.conv2', 'layer4.1.bn2']])

# Quantize the model
model_prepared = tq.prepare(model_fused)
model_quantized = tq.convert(model_prepared)



# Define the quantization configuration
quant_config = tq.get_default_qconfig('fbgemm')
model_fused.qconfig = quant_config



train_dataset, test_dataset, train_loader, test_loader = util.dataset_util.dataset_load()



with torch.no_grad(): # save memory during inference
    print('Test accuracy: %.2f%%' % (model_util.compute_accuracy(model_quantized, test_loader, device='cpu')))


try:
    model_util.save_model_script(model.half(), r'models\resnet18_trained_PTQ.pt')
    print('model 8bit saved')

except:
    print('ERROR Saving')
