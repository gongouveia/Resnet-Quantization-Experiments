
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


BATCH_SIZE = 128


def dataset_load():

    train_dataset = datasets.MNIST(root='data', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = datasets.MNIST(root='data', 
                                train=False, 
                                transform=transforms.ToTensor())


    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

    # Checking the dataset
    for images, labels in train_loader:  
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break


    return train_dataset, test_dataset, train_loader, test_loader


