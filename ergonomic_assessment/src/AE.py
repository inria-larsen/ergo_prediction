import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
import configparser
import tools
import numpy as np
import json

from Skeleton import Skeleton
from HumanPosture import HumanPosture
from ErgoAssessment import ErgoAssessment

import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import visualization_tools as vtools

import multiprocessing as mp




# def compute_loss_function(input_data, output_data):
# 		loss_score[metric].append(np.sqrt(np.square(input_data - output_data).mean()))
# 		return loss


class AutoEncoder(nn.Module):
	def __init__(self, input_dim, latent_variable_dim, hidden_dim):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim)
			,
			# nn.ReLU(True)
			# nn.Linear(hidden_dim, latent_variable_dim),
			# nn.ReLU(True)
			# nn.Sigmoid()
			)
		self.decoder = nn.Sequential(
			# nn.Linear(latent_variable_dim, hidden_dim)
			# ,
			# nn.ReLU(True),
			nn.Linear(hidden_dim, input_dim),
			# nn.ReLU(True)
			# nn.Sigmoid()
			)

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
		
		return m, x
	
	def decode(self, z):
		x = torch.relu(self.fc3(z))
		x = torch.sigmoid(self.fc4(x))
		return x


class ModelAutoencoder():
	"""docstring for ModelAutoencoder"""
	def __init__(self, parser):
		parser.add_argument('--file', '-f', help='Configuration file', default="config/config_AE.json")
		parser.add_argument('--config', '-c', help='Configuration type', default="DEFAULT")
		args=parser.parse_args()
		config_file = args.file
		config_type = args.config

		self.initilization(config_file, config_type)
		self.load_data(self.path)
		self.config_model()

	def initilization(self, config_file, config_type = 'DEFAULT'):
		# Parameters configuration
		with open(config_file, 'r') as f:
			param_init = json.load(f)

		self.path = param_init["path_data"]

		self.config = param_init[config_type]
		
		self.BATCH_SIZE = 64
		self.LR = 0.005		 # learning rate
		self.N_TEST_IMG = 5
		self.tracks = ['details']
		self.ratio_split = self.config["ratio_split_sets"]
		self.input_type = self.config["input_type"]
		
	def config_model(self):
		self.nbr_epoch = int(self.config["epoch"])
		self.latent_dim = self.config["latent_dim"]
		self.hidden_dim = self.config["hidden_dim"]
		self.type_AE = self.config["type_AE"]

		if self.type_AE == 'AE':
			self.autoencoder = AutoEncoder(self.input_dim, self.latent_dim, self.hidden_dim)
		elif self.type_AE == 'VAE':
			self.autoencoder = VariationalAutoencoder(self.input_dim, self.latent_dim, self.hidden_dim)

		self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.LR)
		self.loss_func = nn.MSELoss()

	def change_config(self, key_config, new_value):
		self.config[key_config] = new_value
		self.config_model()

	def load_data(self, path):
		self.list_features, self.data_np, self.real_labels, self.timestamps, list_states = tools.load_data(path, self.tracks, self.input_type + '_')
		self.list_states = list_states[0]

		self.seq_data_train, seq_labels_train, self.seq_data_test, seq_labels_test, seq_id_train, seq_id_test = tools.split_data_base(self.data_np, self.real_labels[0], self.ratio_split)

		self.data_train = []
		self.data_test = []

		for data_joint in self.seq_data_train:
			for d_joint in data_joint:
				self.data_train.append(d_joint)

		for data_joint in self.seq_data_test:
			for d_joint in data_joint:
				self.data_test.append(d_joint)

		x_joint = np.asarray(self.data_train)

		x_norm = np.copy(x_joint)
		x_joint = x_joint.astype(np.float32)
		self.mean_norm = np.mean(x_joint, axis = 0)
		self.var_norm = np.std(x_joint, axis = 0)

		size_data, self.input_dim = np.shape(x_joint)
		for i in range(self.input_dim):
			x_norm[:,i] = (x_joint[:,i] - self.mean_norm[i])/self.var_norm[i]
		x_norm = x_norm.astype(np.float32)

		self.train_loader = torch.from_numpy(x_norm)

	def train_model(self, nbr_epoch = 100, list_metric = ['jointAngle']):
		loss_score = {}
		for metric in list_metric:
			loss_score[metric] = []

		skeleton = Skeleton('dhm66_ISB_Xsens.urdf')

		b_x = self.train_loader.view(-1, self.input_dim)
		b_y = self.train_loader.view(-1, self.input_dim)

		input_data = np.copy(b_x.detach().numpy())

		for i in range(self.input_dim):
			input_data[:,i] = np.rad2deg(input_data[:,i]*self.var_norm[i] + self.mean_norm[i])

		# if self.input_type == 'jointAngle':
		# 	input_data = np.rad2deg(input_data[:,i])

		if 'position' in list_metric:
			# input_position = np.zeros((len(input_data), 12))
			output_position = np.zeros((len(input_data), 12))
			end_effectors = ['Right Hand', 'Left Hand', 'Right Foot', 'Left Foot']

			pool = mp.Pool(mp.cpu_count())
			# position = [pool.apply(skeleton.update_posture, args=(data, True)) for data in input_data]
			# position = pool.starmap(skeleton.update_posture, [(data, True) for data in input_data])

			result_objects = [pool.apply_async(skeleton.update_posture, args=(data, True, i)) for i, data in enumerate(input_data)]
			input_position = [r.get() for r in result_objects]

			pool.close()
			pool.join()

			# for i in range(len(input_data)):
			# 	skeleton.update_posture(input_data[i])
				# for num_link, linkname in enumerate(end_effectors):
				# 	input_position[i,num_link*3:num_link*3+3] = skeleton.get_segment_position(linkname)[0:3,3]

		if 'ergo_score' in list_metric:
			score_total = 'RULA_SCORE'
			human_posture = HumanPosture('config/mapping_joints.json')
			ergo_assessment = ErgoAssessment('config/rula_config.json')

			input_ergo = np.zeros((len(input_data), 1))
			output_ergo = np.zeros((len(input_data), 1))

			for i in range(len(input_data)):
				human_posture.update_posture(input_data[i])
				ergo_assessment.compute_ergo_scores(human_posture)
				input_ergo[i] = ergo_assessment[score_total]


		list_loss = []
		for epoch in range(self.nbr_epoch):

			encoded, decoded = self.autoencoder(b_x)

			loss = self.loss_func(decoded, b_y)	  # mean square error

			self.optimizer.zero_grad()			   # clear gradients for this training step
			loss.backward()					 # backpropagation, compute gradients
			self.optimizer.step()					# apply gradients
			
			output_data = np.copy(decoded.detach().numpy())

			for i in range(self.input_dim):
				output_data[:,i] = np.rad2deg(output_data[:,i]*self.var_norm[i] + self.mean_norm[i])

			for metric in list_metric:
				if metric == 'jointAngle':
					loss_score[metric].append(np.sqrt(np.square(input_data - output_data).mean()))
				if metric == 'ergo_score':
					for i in range(len(input_data)):
						human_posture.update_posture(output_data[i])
						ergo_assessment.compute_ergo_scores(human_posture)
						output_ergo[i] = ergo_assessment[score_total]
					loss_score[metric].append(np.sqrt(np.square(input_ergo - output_ergo).mean()))

				elif metric == 'position':
					for i in range(len(input_data)):
						skeleton.update_posture(output_data[i])
						for num_link, linkname in enumerate(end_effectors):
							output_position[i,num_link*3:num_link*3+3] = skeleton.get_segment_position(linkname)[0:3,3]

					loss_score[metric].append(np.sqrt(np.square(input_position - output_position).mean()))

				if epoch%100 == 0:
					print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

				list_loss.append(loss_score[metric][epoch]) 
				if len(list_loss) > 500:
					del list_loss[0]
					if np.std(list_loss) < 0.005:
						break

		return loss_score

	def test_model(self, data):
		x_joint = np.asarray(data)

		x = x_joint
		x_norm = x_joint

		for i in range(self.input_dim):
		 	x_norm[:,i] = (x[:,i] - self.mean_norm[i])/self.var_norm[i]
		x_norm = x_norm.astype(np.float32)

		train_loader = torch.from_numpy(x_norm)

		b_x = train_loader.view(-1, self.input_dim)

		encoded, decoded = self.autoencoder(b_x)

		decoded_joint = decoded.detach().numpy()

		for i in range(self.input_dim):
			decoded_joint[:,i] = decoded_joint[:,i]*self.var_norm[i] + self.mean_norm[i]

		return decoded_joint

	def get_data_train(self):
		return self.data_train

	def get_data_test(self):
		return self.data_test



	

		



