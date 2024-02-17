import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset, random_split
import torchvision.transforms as transforms

"""
Write Code for Downloading Image and Audio Dataset Here
"""
# Image Downloader
train_image_dataset_downloader = torchvision.datasets.CIFAR10(
    root='./data',  # Specify the root directory where the dataset will be downloaded
    train=True,      # Set to True for the training set, False for the test set
    download=True,   # Set to True to download the dataset if not already downloaded
    transform=transforms.ToTensor()    
)

test_image_dataset_downloader = torchvision.datasets.CIFAR10(
    root='./data',  # Specify the root directory where the dataset will be downloaded
    train=False,      # Set to True for the training set, False for the test set
    download=True,   # Set to True to download the dataset if not already downloaded
    transform=transforms.ToTensor()    
)

# Audio Downloader
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data',      # Specify the root directory where the dataset will be downloaded
    url='speech_commands_v0.02',  # Specify the version of the dataset
    download=True,       # Set to True to download the dataset if not already downloaded
)



class ImageDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split

        # full_dataset = torchvision.datasets.CIFAR10(
        #     root='./data',
        #     train=True,
        #     download=False,
        #     transform=transforms.ToTensor()
        # )

        # Calculate the number of samples for training and validation
        num_samples = len(train_image_dataset_downloader)
        num_val_samples = int(0.1 * num_samples)
        num_train_samples = num_samples - num_val_samples

        # Perform train-validation split
        train_dataset, val_dataset = random_split(train_image_dataset_downloader, [num_train_samples, num_val_samples])

        # Assign the appropriate dataset based on the split
        if split == "train":
            self.dataset = train_dataset
        elif split == "val":
            self.dataset = val_dataset
        elif split == "test":
            self.dataset = test_image_dataset_downloader

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # You can perform additional processing or transformations here if needed
        return image, label
        # pass
        # """
        # Write your code here
        # """

class AudioDataset(Dataset):
    def __init__(self, split:str="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split

        num_samples = len(audio_dataset_downloader)
        num_train_samples = int(0.75 * num_samples)
        num_val_test_samples = num_samples - num_train_samples

        num_val_samples = int(0.10 * num_val_test_samples)
        num_test_samples = num_val_test_samples - num_val_samples

        # Perform train-validation split
        train_dataset, val_test_dataset = random_split(audio_dataset_downloader, [num_train_samples, num_val_test_samples])
        val_dataset, test_dataset = random_split(val_test_dataset, [num_val_samples, num_test_samples])

        

        pass
        """
        Write your code here
        """

class Resnet_Q1(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        
class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        
class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        
class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    network = network.to(device)
    
    # Write your code here
    for epoch in range(EPOCH):
        pass
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """ 

def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    network = network.to(device)
    
    # Write your code here
    for epoch in range(EPOCH):
        pass
    """
    Only use this print statement to print your epoch loss, accuracy
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def evaluator(dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    # Write your code here
    for epoch in range(EPOCH):
        pass
    """
    Only use this print statement to print your loss, accuracy
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """ 
    
    