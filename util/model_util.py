
import torch



def save_model_entire(model, file_path):
    """
    Saves the entire PyTorch model to a file.

    Args:
        model (torch.nn.Module): The model to be saved.
        file_path (str): The file path where the model should be saved.
    """
    torch.save(model, file_path)
    print(f"Model saved to {file_path}")



# This is better beacuse it saves network graph
def save_model_script(model, file_path):
    """
    Saves a PyTorch model using torch.jit.script.

    Args:
        model (torch.nn.Module): The model to be saved.
        file_path (str): The file path where the model should be saved.
    """
    try:
        with torch.no_grad():
            scripted_model = torch.jit.script(model)
            scripted_model.save(file_path)

        print(f"Scripted model saved to {file_path}")

    except:
        print(" SomeError")


def load_model_entire(file_path):
    """
    Loads a PyTorch model from a file.

    Args:
        file_path (str): The file path from where the model should be loaded.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = torch.load(file_path)
    print(f"Model loaded from {file_path}")
    return model



def load_pytorch_script_model(model_path):
    """
    Load a PyTorch Script model from a file.
    
    Parameters:
    - model_path (str): The file path to the TorchScript model.
    
    Returns:
    - torch.jit.ScriptModule: The loaded PyTorch Script model.
    """
    try:
        # Load the model
        model = torch.jit.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None
    


    
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    


    
def compute_accuracy_half(model, data_loader, device):
    model = model.half()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features.half())
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

