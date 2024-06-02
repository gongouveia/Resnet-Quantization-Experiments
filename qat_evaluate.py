
import util.model_util as model_util
import util.dataset_util as dataset_util



import torch
from model import quantizable_resnet18

from hyperparameters import DEVICE

model_path = r'models\quantized_resnet18.pth'



def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Load the quantized model
quantized_model_path = "quantized_resnet18.pth"
loaded_model = load_model(lambda: quantizable_resnet18(10,True), quantized_model_path)

# Evaluate the quantized model on the test dataset
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Move the model to the appropriate device
loaded_model.to(DEVICE)


train_dataset, test_dataset, train_loader, test_loader = dataset_util.dataset_load()


# Evaluate the model
evaluate_model(loaded_model, test_loader)