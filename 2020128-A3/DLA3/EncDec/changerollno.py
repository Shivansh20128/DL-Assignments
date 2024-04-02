import os
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np

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
            image_paths.append(os.path.join(folder_path, filename))
        return image_paths

    def __len__(self):
        return len(self.clean_image_paths)

    def __getitem__(self, idx):
        clean_image_path = self.clean_image_paths[idx]
        # print("clean image path:   ",clean_image_path)
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

            
                # print("clean image: ", clean_image.shape)
                # print("noisy image: ", type(noisy_image))
                noisy_images.append(noisy_image)
                clean_images.append(clean_image)
        # print("Printing noisy images")
        # print(noisy_image.shape)
        if len(noisy_images)!=0:
            noisy_images = torch.stack(noisy_images).squeeze().unsqueeze(0)
            clean_images = torch.stack(clean_images).squeeze().unsqueeze(0)
        else:
            # print("isme")
            noisy_images = torch.zeros(1,28,28)
            clean_images = torch.zeros(1,28,28)

        # print("noisy len:",len(noisy_images))
        # print("clean len:",len(clean_images))
        # print(" ")
        
        return noisy_images, clean_images


def plot_tsne_embeddings(encoder, data_loader, n_epochs):
    all_embeddings = []
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            logits = encoder(inputs)  # Assuming your encoder provides embeddings/logits
            all_embeddings.append(logits.cpu().numpy().reshape(inputs.size(0), -1))  # Flatten the embeddings
            labels.append(targets.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    embeddings_3d = tsne.fit_transform(all_embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label
        ax.scatter(embeddings_3d[indices][:, 0], embeddings_3d[indices][:, 1], embeddings_3d[indices][:, 2], label=label)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(f't-SNE Embeddings After {n_epochs} Epochs')
    ax.legend()
    plt.show()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.residual1 = ResidualBlock(64, 64)
        self.residual2 = ResidualBlock(64, 64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.residual3 = ResidualBlock(32, 32)
        self.residual4 = ResidualBlock(32, 32)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.residual3(x)
        x = self.residual4(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.residual1 = ResidualBlock(64, 64)
        self.residual2 = ResidualBlock(64, 64)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.conv2(x)
        return x

class AELossFn:
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self):
        super(AELossFn, self).__init__()
        # print("this is the loss function")
        self.loss_fn = torch.nn.MSELoss()
        # print("this is the loss function after calling MSE loss")

    def forward(self, input, target):
        # print("this is inside the forward function")
        value_loss = self.loss_fn(input, target)
        # print("value loss:  ",value_loss)
        return value_loss

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
        # print("yehs")
        # Initialize optimizer
        self.optimizer = optimizer
        self.train(self.dataloader, 1)
        # print("train hogya shyd")


    def train_step(self, noisy_inputs, clean_targets):
        """
        Perform a single training step.

        Args:
            noisy_inputs (torch.Tensor): Noisy input data.
            clean_targets (torch.Tensor): Corresponding clean data.

        Returns:
            float: Loss value for the current step.
        """
        # print("kmswksm")
        self.encoder.train()
        self.decoder.train()
        noisy_inputs = noisy_inputs.to(self.device)
        clean_targets = clean_targets.to(self.device)

        # Forward pass
        latent_representation = self.encoder(noisy_inputs)
        reconstructed_data = self.decoder(latent_representation)
        # print("clean targets: ", latent_representation)
        # print("recoinstructed data: ", reconstructed_data)
        # print("recoinstructed data: ")
        # Calculate loss
        # print(self.criterion)
        loss = self.criterion.forward(reconstructed_data, clean_targets)
        print("loss:",loss)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataloader, num_epochs):
        # print("in the train function")
        """
        Train the autoencoder.

        Args:
            dataloader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of epochs for training.
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # print("epcohs tk")
            # data = dataloader.dataset.data 
            # shape = dataloader.dataset.data.shape  
            # datatype = dataloader.dataset.data.dtype
            # print(data)
            # print("hello1")
            # print(shape)
            # print("hello2")
            # print(datatype)
            # print(dataloader)

            for batch_idx, (noisy_inputs, clean_targets) in enumerate(dataloader):
                # print("what the hell")
                # print("here i am")
                loss = self.train_step(noisy_inputs, clean_targets)
                epoch_loss += loss
                # break

            epoch_loss /= len(dataloader)
            torch.save({
            'epoch': epoch,
            'model_state_dict_encoder': self.encoder.state_dict(),
            'model_state_dict_decoder': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': epoch_loss,
            }, 'checkpoint.pth')
            # plot_tsne_embeddings(self.encoder, dataloader, n_epochs=epoch+1)
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

def ssim(img1, img2):
    # Convert images to tensors
    img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)  # Assuming input images are in numpy format
    img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    
    # Calculate SSIM
    ssim_value = F.msssim(img1_tensor, img2_tensor, data_range=1, size_average=True)
    
    return ssim_value.item()

class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self, gpu_status):
        self.encoder = Encoder()
        self.decoder = Decoder()
        

    def from_path(self,sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        checkpoint = torch.load("checkpoint.pth")
        
        self.encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
        self.decoder.load_state_dict(checkpoint['model_state_dict_decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        with torch.no_grad():
            encoded = self.encoder(sample)
            decoded = self.decoder(encoded)

        # Convert tensors to numpy arrays
        sample_np = original.squeeze(0).permute(1, 2, 0).numpy()
        decoded_np = decoded.squeeze(0).permute(1, 2, 0).numpy()

        print("sample np: ",sample_np)
        print("decoded np: ",decoded_np)

        # Compute SSIM score
        score = ssim(sample_np, decoded_np)

        return float(score)

        

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """


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