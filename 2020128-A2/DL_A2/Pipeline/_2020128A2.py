import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
import torch.nn.functional as F
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

        # Assign the appropriate dataset based on the split
        if split == "train":
            self.dataset = train_dataset
        elif split == "val":
            self.dataset = val_dataset
        elif split == "test":
            self.dataset = test_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        # You can perform additional processing or transformations here if needed
        return waveform, label
        # pass
        # """
        # Write your code here
        # """

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_1d=False):
        super(ResnetBlock, self).__init__()

        # Convolution layer
        if is_1d:
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Batch normalization layer
        bn_layer = nn.BatchNorm1d(out_channels) if is_1d else nn.BatchNorm2d(out_channels)

        self.block = nn.Sequential(
            conv_layer,
            bn_layer,
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1) if is_1d else nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels) if is_1d else nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Resnet_Q1(nn.Module):
    def __init__(self, num_blocks=18, is_1d=True, *args, **kwargs) -> None:
        super(Resnet_Q1, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1) if is_1d else nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64) if is_1d else nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Create 18 ResNet blocks
        self.resnet_blocks = nn.Sequential(
            *[ResnetBlock(64, 64, is_1d=is_1d) for _ in range(num_blocks)]
        )

        
        # Add a global average pooling layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1) if not is_1d else nn.AdaptiveAvgPool1d(1)

        # Final fully connected layer
        self.fc = nn.Linear(64, 10) if not is_1d else nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resnet_blocks(x)

        # Global average pooling
        x = self.global_avg_pooling(x)

        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)

        # Final fully connected layer
        x = self.fc(x)

        return x
    
        # """
        # Write your code here
        # """
        
class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc_layers(x)
        return x
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
    # network.train()  # Set the network to training mode
    print("in the train function about to train")
    for epoch in range(EPOCH):
        print("inside the loop now")
        print(EPOCH)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        optimizer.zero_grad()
        for inputs, labels in dataloader:
            print("inside dataloader loop")
            inputs, labels = inputs.to(device), labels.to(device)
            print("labels: ", labels)


            outputs = network(inputs)
            # if len(outputs.shape) > 1:
            #     outputs = outputs.squeeze()
            print("output: ", outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            accuracy = correct_predictions / total_samples
            epoch_loss = running_loss / len(dataloader)

            print("Training Epoch: {}, [Loss: {:.4f}, Accuracy: {:.4f}]".format(epoch, epoch_loss, accuracy))

        accuracy = correct_predictions / total_samples
        epoch_loss = running_loss / len(dataloader)

        print("Training Epoch: {}, [Loss: {:.4f}, Accuracy: {:.4f}]".format(epoch, epoch_loss, accuracy))

    # # Write your code here
    # for epoch in range(EPOCH):
    #     pass
    # """
    # Only use this print statement to print your epoch loss, accuracy
    # print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
    #     epoch,
    #     loss,
    #     accuracy
    # ))
    # """ 

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
    
    