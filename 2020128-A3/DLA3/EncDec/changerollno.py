import os
import torch
from torchvision import transforms
from PIL import Image
from torch import nn


class AlteredMNIST():
    """
    AlteredMNIST dataset loader.
    """

    def __init__(self, root_dir="d:\\Deep Learning\\Deep-Learning-Assignments\\2020128-A3\\DLA3\\Data\\", transform=transforms.ToTensor()):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the root directory containing 'aug' and 'clean' folders.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.clean_image_paths = self._get_image_paths('clean')
        self.noisy_image_paths = self._get_image_paths('aug')

    def _get_image_paths(self, folder_name):
        image_paths = []
        folder_path = os.path.join(self.root_dir, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                image_paths.append(os.path.join(folder_path, filename))
        return image_paths

    def __len__(self):
        return len(self.clean_image_paths)

    def __getitem__(self, idx):
        clean_image_path = self.clean_image_paths[idx]
        clean_image = Image.open(clean_image_path).convert('L')

        # print("clean imagepath: ", clean_image_path)

        # Find corresponding noisy images
        noisy_images = []
        clean_images = []
        for noisy_image_path in self.noisy_image_paths:
            # print("weell: ", noisy_image_path.split('_')[1])
            if noisy_image_path.split('_')[1] == clean_image_path.split('_')[1]:
                noisy_image = Image.open(noisy_image_path).convert('L')
                if self.transform:
                    noisy_image = self.transform(noisy_image)
                    clean_image = self.transform(clean_image)
                # print("clean image: ", clean_image)
                # print("noisy image: ", noisy_image)
                noisy_images.append(noisy_image)
                clean_images.append(clean_image)
        # print("Printing noisy images")
        # print(noisy_image.shape)

        # print("Printing clean images")
        # print(clean_image.shape)

        
        # print("noisy len:",len(noisy_images))
        # print("clean len:",len(clean_images))
        # print(" ")
        return noisy_images, clean_images


class Encoder(nn.Module):
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    def __init__(self, input_size=28*28, hidden_size1=128, hidden_size2=16, z_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1 , hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, z_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(nn.Module):
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self, output_size=28*28, hidden_size1=128, hidden_size2=16, z_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2 , hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, output_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class AELossFn:
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self):
        super(AELossFn, self).__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input, target):
        return self.loss_fn(input, target)

class VAELossFn:
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    pass

def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train
    """
    return list(E.parameters()) + list(D.parameters())


class AETrainer:
    """
    Write code for training AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self,dataloader, encoder, decoder, loss_func, optimizer, device):
        """
        Initialize the AETrainer.

        Args:
            encoder (nn.Module): Encoder network.
            decoder (nn.Module): Decoder network.
            criterion (nn.Module): Loss function.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = loss_func
        # self.learning_rate = learning_rate
        self.device = device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.dataloader = dataloader
        print("yehs")
        # Initialize optimizer
        self.optimizer = optimizer
        self.train(dataloader, 50)
        print("train hogya shyd")


    def train_step(self, noisy_inputs, clean_targets):
        """
        Perform a single training step.

        Args:
            noisy_inputs (torch.Tensor): Noisy input data.
            clean_targets (torch.Tensor): Corresponding clean data.

        Returns:
            float: Loss value for the current step.
        """
        print("kmswksm")
        self.encoder.train()
        self.decoder.train()
        noisy_inputs = noisy_inputs.to(self.device)
        clean_targets = clean_targets.to(self.device)

        # Forward pass
        latent_representation = self.encoder(noisy_inputs)
        reconstructed_data = self.decoder(latent_representation)

        # Calculate loss
        loss = self.criterion(reconstructed_data, clean_targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataloader, num_epochs):
        print("in the train function")
        """
        Train the autoencoder.

        Args:
            dataloader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of epochs for training.
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            print("epcohs tk")
            # data = dataloader.dataset.data 
            # shape = dataloader.dataset.data.shape  
            # datatype = dataloader.dataset.data.dtype
            # print(data)
            # print("hello1")
            # print(shape)
            # print("hello2")
            # print(datatype)
            # print(dataloader)
            print()
            for batch_idx, (noisy_inputs, clean_targets) in enumerate(dataloader):
                print("what the hell")
                print("here i am")
                # loss = self.train_step(noisy_inputs, clean_targets)
                # epoch_loss += loss
                break

            epoch_loss /= len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    pass


class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    pass

    def from_path(sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        pass

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    pass

    def from_path(sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        pass

class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass

class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()