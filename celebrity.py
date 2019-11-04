
#############
# Libraries #
#############

import os
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from tqdm import tqdm
from numpy.random import randint
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import leaky_relu, relu, tanh
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, BCELoss
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Normalize, Resize, CenterCrop
warnings.filterwarnings("ignore")

##################
# Data generator #
##################

train_df = os.listdir("./input/train/") # list of img IDs

class CelebrityDataset(Dataset):    
    def __init__(self, df, img_path, augmentations = None):    
        self.df = df
        self.img_path = img_path
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.df[index]
        img = Image.open(self.img_path + img_id)
        
        if self.augmentations is not None:
            img = self.augmentations(img)
        
        return img

augs = Compose([
        Resize(64), # if h > w, rescale to (64 * h / w, 64)
        CenterCrop(64),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])
                                 
train_dataset = CelebrityDataset(train_df, "./input/train/", augmentations = augs)

batch_size = 64

train_gen = DataLoader(train_dataset, batch_size, shuffle = True)

###########################
# Visualize training data #
###########################

imgs = next(iter(train_gen))
imgs = imgs.numpy().transpose(0, 2, 3, 1) # reshape for plotting
    
fig = plt.figure(figsize = (11, 5))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.imshow((imgs[i] + 1) / 2) # un-normalize from -1...1 to 0...1
    plt.axis("off")
plt.tight_layout() 

#################
# Specify model #
#################

device = torch.device("cuda:0")

latent_dim = 128 # latent vector (i.e. size of generator input)

class Generator(nn.Module):   
    def __init__(self, latent_dim = 128):
        super(Generator, self).__init__()        
        self.latent_dim = latent_dim
        self.convt1 = ConvTranspose2d(self.latent_dim, 1024, 4, stride = 1, padding = 0, bias = False)
        self.convt2 = ConvTranspose2d(1024, 512, 4, stride = 2, padding = 1, bias = False)
        self.convt3 = ConvTranspose2d(512, 256, 4, stride = 2, padding = 1, bias = False)
        self.convt4 = ConvTranspose2d(256, 128, 4, stride = 2, padding = 1, bias = False)
        self.convt5 = ConvTranspose2d(128, 64, 4, stride = 2, padding = 1, bias = False)
        self.convt6 = ConvTranspose2d(64, 3, 3, stride = 1, padding = 1, bias = True)
        self.bn1 = BatchNorm2d(1024)
        self.bn2 = BatchNorm2d(512)
        self.bn3 = BatchNorm2d(256)
        self.bn4 = BatchNorm2d(128)
        self.bn5 = BatchNorm2d(64)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1) # reshape to (n, 128, 1, 1)
        x = relu(self.bn1(self.convt1(x)), inplace = True) # in-place saves GPU memory
        x = relu(self.bn2(self.convt2(x)), inplace = True)
        x = relu(self.bn3(self.convt3(x)), inplace = True)
        x = relu(self.bn4(self.convt4(x)), inplace = True)
        x = relu(self.bn5(self.convt5(x)), inplace = True)
        x = tanh(self.convt6(x))
        return x

class Discriminator(nn.Module):      
    def __init__(self):
        super(Discriminator, self).__init__()        
        self.conv1 = Conv2d(3, 32, 4, stride = 2, padding = 1, bias = False)
        self.conv2 = Conv2d(32, 64, 4, stride = 2, padding = 1, bias = False)
        self.conv3 = Conv2d(64, 128, 4, stride = 2, padding = 1, bias = False)
        self.conv4 = Conv2d(128, 256, 4, stride = 2, padding = 1, bias = False)
        self.conv5 = Conv2d(256, 1, 4, stride = 1, padding = 0, bias = False)
        self.bn1 = BatchNorm2d(128)
        self.bn2 = BatchNorm2d(256)


    def forward(self, x):
        x = leaky_relu(self.conv1(x), 0.2, inplace = True)
        x = leaky_relu(self.conv2(x), 0.2, inplace = True)
        x = leaky_relu(self.bn1(self.conv3(x)), 0.2, inplace = True)
        x = leaky_relu(self.bn2(self.conv4(x)), 0.2, inplace = True)
        x = self.conv5(x)
        x = x.view(-1, 1) # reshape to (n, 1)
        return x
    
netG = Generator(latent_dim).to(device)
netD = Discriminator().to(device)

criterion = BCELoss()

optimizerD = Adam(netD.parameters(), lr = 1e-3, betas = (0.5, 0.999))
optimizerG = Adam(netG.parameters(), lr = 1e-3, betas = (0.5, 0.999))

#############
# Fit model #
#############

num_epochs = 60

fixed_noise = torch.randn((1, latent_dim, 1, 1), device = device) # for visualization from same point in latent space
G_losses = []
D_losses = []

for epoch_i in range(num_epochs):    
    for batch_i, imgs in enumerate(tqdm(train_gen)):
        
        # ----------- Classify real images -----------
        netD.zero_grad()
        real_imgs = imgs.to(device)
        b_size = imgs.shape[0]
        labels = torch.full((b_size, 1), 0.5, device = device) # label vector filled with 0.5
        out_real = netD(real_imgs)
        
        # ----------- Generate and classify fake images -----------
        noise = torch.randn((b_size, latent_dim, 1, 1), device = device)
        fake_imgs = netG(noise)
        out_fake = netD(fake_imgs.detach()) # detach from graph
        
        # ----------- Update discriminator -----------
        errD = (torch.mean((out_real - torch.mean(out_fake) - labels) ** 2) +
                torch.mean((out_fake - torch.mean(out_real) + labels) ** 2)) / 2
        errD.backward(retain_graph = True)
        optimizerD.step()
        
        # ----------- Classify fake images -----------
        netG.zero_grad()
        out_fake = netD(fake_imgs)   
        
        # ----------- Update generator -----------
        errG = (torch.mean((out_real - torch.mean(out_fake) + labels) ** 2) +
                torch.mean((out_fake - torch.mean(out_real) - labels) ** 2)) / 2
        errG.backward()
        optimizerG.step()
        
        # ----------- Save errors -----------
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
    # ----------- Visualization -----------
    with torch.no_grad():
        img = netG(fixed_noise).cpu().squeeze() # send to host memory and drop batch dim
        img = img.numpy().transpose(1, 2, 0)
        img = (img + 1) / 2        
        plt.imshow(img)
        plt.show()
    
    print(f"\n{epoch_i + 1} | Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")

#########################
# Visualize fake images #
#########################

# ----------- Generate a few images -----------
with torch.no_grad():   
    noise = torch.randn((18, latent_dim, 1, 1), device = device)
    imgs = netG(noise).cpu() # copy to to host memory
    imgs = (imgs + 1) / 2
    imgs = imgs.numpy().transpose(0, 2, 3, 1)

fig = plt.figure(figsize = (11, 5))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.imshow(imgs[i])
    plt.axis("off")
plt.tight_layout() 
    
# ----------- Save large batch of images -----------
with torch.no_grad():
    noise = torch.randn((100, latent_dim, 1, 1), device = device) # generate 100 images
    imgs = netG(noise)
    imgs = (imgs + 1) / 2
    for i in range(100):
        save_image(imgs[i], f"./output/{randint(0, 999999):06d}.png")
        
