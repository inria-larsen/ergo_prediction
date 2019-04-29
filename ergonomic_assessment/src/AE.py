import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(66, 2),
			nn.ReLU(True),
			nn.Linear(10, 2),
			nn.ReLU(True))
		self.decoder = nn.Sequential(
			nn.Linear(2, 66),
			nn.ReLU(True),
			nn.Linear(10, 66),
			nn.Sigmoid())

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded


class VariationalAutoencoder(nn.Module):
	def __init__(self, latent_variable_dim):
		super(VAE, self).__init__()
		self.fc1 = nn.Linear(66, 10)
		self.fc2m = nn.Linear(10, latent_variable_dim) # use for mean
		self.fc2s = nn.Linear(10, latent_variable_dim) # use for standard deviation
		
		self.fc3 = nn.Linear(latent_variable_dim, 10)
		self.fc4 = nn.Linear(10, 66)
		
	def reparameterize(self, log_var, mu):
		s = torch.exp(0.5*log_var)
		eps = torch.rand_like(s) # generate a iid standard normal same shape as s
		return eps.mul(s).add_(mu)
		
	def forward(self, input):
		x = input.view(-1, 66)
		x = torch.relu(self.fc1(x))
		log_s = self.fc2s(x)
		m = self.fc2m(x)
		z = self.reparameterize(log_s, m)
		
		x = self.decode(z)
		
		return x, m, log_s
	
	def decode(self, z):
		x = torch.relu(self.fc3(z))
		x = torch.sigmoid(self.fc4(x))
		return x
