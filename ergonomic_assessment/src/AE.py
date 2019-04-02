import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(66, 2)
		)
		self.decoder = nn.Sequential(
			nn.Linear(2, 66)
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded