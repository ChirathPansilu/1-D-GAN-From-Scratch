#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
import torch.optim as optim

#generate real distribution data 
def generate_real_data(n):
    x1 = np.random.uniform(-50,50,size=(n,1)) 
    x2 = x1**2                                
    return torch.from_numpy(np.hstack((x1, x2))).float()


#define discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        self.output = nn.Linear(2, 1)
        
    def forward(self,x):
        #forward propagate through discriminator
        out = F.leaky_relu(self.fc1(x))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        out = self.output(out)
        
        return out
        

#define generator model 
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 2)
        
    def forward(self,x):
        out = F.leaky_relu(self.fc1(x))
        out = F.leaky_relu(self.fc2(out))
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


#set hyperparametrs
latent_dim = 10
lr = 0.001
hidden_dim= 25

#instantiate Generator and Descriminator
D = Discriminator(2, 16)
G = Generator(latent_dim, 16, 2)

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
   latent = np.random.normal(-50,50,size=(batch_size, latent_dim))
   latent = torch.from_numpy(latent).float()
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
   latent = np.random.normal(-50,50, size=(batch_size, latent_dim))
   latent = torch.from_numpy(latent).float()
   fake_data = G(latent)
   
   fake_output = D(fake_data)
   g_loss = real_loss(fake_output)
   
   gen_losses.append(g_loss)
   
   g_loss.backward()
   g_optimizer.step()
   
   if i%show_every == 0:
       print(f'Epoch: {i} | d_loss: {d_loss} | g_loss: {g_loss}')
   
   

   
   
   
   
    


    
        
        
        
        
        
        
        
        
