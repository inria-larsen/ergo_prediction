import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AutoEncoder(nn.Module):
	def __init__(self, input_dim, latent_variable_dim, hidden_dim):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim)
			,
			nn.ReLU(True),
			nn.Linear(hidden_dim, latent_variable_dim),
			nn.ReLU(True))
		self.decoder = nn.Sequential(
			nn.Linear(latent_variable_dim, hidden_dim)
			,
			nn.ReLU(True),
			nn.Linear(hidden_dim, input_dim),
			nn.Sigmoid())

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded


class VariationalAutoencoder(nn.Module):
	def __init__(self, input_dim, latent_variable_dim, hidden_dim):
		super(VariationalAutoencoder, self).__init__()
		self.input_dim = input_dim
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2m = nn.Linear(hidden_dim, latent_variable_dim) # use for mean
		self.fc2s = nn.Linear(hidden_dim, latent_variable_dim) # use for standard deviation
		
		self.fc3 = nn.Linear(latent_variable_dim, hidden_dim)
		self.fc4 = nn.Linear(hidden_dim, input_dim)
		
	def reparameterize(self, log_var, mu):
		s = torch.exp(0.5*log_var)
		eps = torch.rand_like(s)
		return eps.mul(s).add_(mu)
		
	def forward(self, input):
		x = input.view(-1, self.input_dim)
		x = torch.relu(self.fc1(x))
		log_s = self.fc2s(x)
		m = self.fc2m(x)
		z = self.reparameterize(log_s, m)
		
		x = self.decode(z)
		
		return x, m
	
	def decode(self, z):
		x = torch.relu(self.fc3(z))
		x = torch.sigmoid(self.fc4(x))
		return x