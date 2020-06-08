#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
import torch.optim as optim

#define a function for sampling from real data distribution
def generate_real_data(n):
    x1 = np.random.uniform(-0.5,0.5,size=(n,1)) 
    x2 = x1**2                                
    return torch.from_numpy(np.hstack((x1, x2))).float()


#define discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 25)
        self.output = nn.Linear(25, 1)
        
    def forward(self,x):
        #forward propagate through discriminator
        out = F.relu(self.fc1(x))
        out = self.output(out)
        
        return out
        

#define generator model 
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, 15)
        self.out = nn.Linear(15, 2)
        
    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = self.out(out)
        
        return out
        

#define real loss
def real_loss(d_out) :
    batch_size = d_out.size(0)
    
    #create labels for real data
    labels = torch.ones(batch_size, 1)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(d_out, labels)
    return loss

#define fake loss 
def fake_loss(d_out):
    batch_size = d_out.size(0)
    
    #create labels for fake data
    labels = torch.zeros(batch_size, 1)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(d_out, labels)
    return loss

#define latent points generator function
def generate_latent_points(n_samples, latent_dim):
    #generate latent samples from a standard normal
    latent = np.random.normal(0,1,size=(n_samples, latent_dim))
    latent = torch.from_numpy(latent).float()
    return latent


#set hyperparametrs
latent_dim = 5
lr = 0.001


#instantiate Generator and Descriminator
D = Discriminator(2)
G = Generator(latent_dim, 2)

D.train(), G.train()

#define optimizers for discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, betas=(0.5,0.999))
g_optimizer = optim.Adam(G.parameters(), lr, betas=(0.5,0.999))

epochs = 10000
batch_size = 128
show_every = 100

gen_losses = []
dis_losses = []

#implement training loop
for i in range(1 ,epochs+1):

    #=========================================
    #        Train Discriminator
    #=========================================

    d_optimizer.zero_grad()
    #get a batch from real distribution
    real_data  = generate_real_data(batch_size)

    #calculate loss on real samples
    real_output = D(real_data)
    d_r_loss = real_loss(real_output)

    #generate latent samples from a standard normal
    latent = generate_latent_points(batch_size, latent_dim)
    fake_data = G(latent)

    fake_output = D(fake_data)
    d_f_loss = fake_loss(fake_output)

    #accumilate losses
    d_loss = d_r_loss + d_f_loss

    dis_losses.append(d_loss)

    d_loss.backward()
    d_optimizer.step()

    #===========================================
    #         Train Generator
    #===========================================
    g_optimizer.zero_grad()

    #generate latent samples for generator
    latent = generate_latent_points(batch_size, latent_dim)
    fake_data = G(latent)

    fake_output = D(fake_data)
    g_loss = real_loss(fake_output)

    gen_losses.append(g_loss)

    g_loss.backward()
    g_optimizer.step()

    if i%show_every == 0:
       print(f'Epoch: {i} | d_loss: {d_loss} | g_loss: {g_loss}')