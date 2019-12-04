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


class AutoEncoderSimple(nn.Module):
	def __init__(self, input_dim, latent_variable_dim, output_dim):
		super(AutoEncoderSimple, self).__init__()

		self.encoder = nn.Sequential(nn.Linear(input_dim, latent_variable_dim), nn.Sigmoid())
		self.decoder = nn.Sequential(nn.Linear(latent_variable_dim, output_dim), nn.Sigmoid())

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded

	def encode_data(self, x):
		encoded = self.encoder(x)
		return encoded

	def decode_data(self, x):
		decoded = self.decoder(x)
		return decoded


class AutoEncoder(nn.Module):
	def __init__(self, input_dim, latent_variable_dim, hidden_dim, output_dim):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.Sigmoid(),
			nn.Linear(hidden_dim, latent_variable_dim),
			nn.Sigmoid()
			)
		self.decoder = nn.Sequential(
			nn.Linear(latent_variable_dim, hidden_dim),
			nn.Sigmoid(),
			nn.Linear(hidden_dim, output_dim),
			nn.Sigmoid()
			)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded

	def encode_data(self, x):
		encoded = self.encoder(x)
		return encoded

	def decode_data(self, x):
		decoded = self.decoder(x)
		return decoded

class VariationalAutoencoder(nn.Module):
	def __init__(self, input_dim, latent_variable_dim, hidden_dim, output_dim):
		super(VariationalAutoencoder, self).__init__()
		self.input_dim = input_dim
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2m = nn.Linear(hidden_dim, latent_variable_dim) # use for mean
		self.fc2s = nn.Linear(hidden_dim, latent_variable_dim) # use for standard deviation
		
		self.fc3 = nn.Linear(latent_variable_dim, hidden_dim)
		self.fc4 = nn.Linear(hidden_dim, output_dim)
		
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

	def __init__(self, parser, path_data = ""):
		args=parser.parse_args()
		config_file = args.file
		config_type = args.config

		self.initilization(config_file, config_type, path_data = path_data)
		self.load_data(self.path)
		self.config_model()

	def initilization(self, config_file, config_type = 'DEFAULT', path_data = ""):
		# Parameters configuration
		with open(config_file, 'r') as f:
			param_init = json.load(f)

		self.path = path_data + param_init["path_data"]

		self.config = param_init[config_type]
		
		self.BATCH_SIZE = 64
		self.LR = 0.0005		 # learning rate
		self.tracks = ['details']
		self.ratio_split = self.config["ratio_split_sets"]
		self.input_type = self.config["input_type"]
		self.list_metric = self.config["list_metric"]
		self.output_type = self.config["output_type"]
		
	def config_model(self):
		self.nbr_epoch = int(self.config["epoch"])
		self.latent_dim = self.config["latent_dim"]
		self.hidden_dim = self.config["hidden_dim"]
		self.type_AE = self.config["type_AE"]
		self.stop_criterion = self.config["stop_criterion"]

		if 'ergo_score' in self.list_metric:
			# self.output_dim = len(self.list_ergo_score)
			self.output_dim = self.input_dim

		elif self.output_type == 'ergo_posture':
			self.ergo_assessment = ErgoAssessment('config/rula_config.json')
			self.list_ergo_score = self.ergo_assessment.get_list_score_name()
			self.list_ergo_score.sort()
			self.output_dim = len(self.list_ergo_score) + self.input_dim

		else:
			self.output_dim = self.input_dim

		if self.type_AE == 'AE':
			self.autoencoder = AutoEncoder(self.input_dim, self.latent_dim, self.hidden_dim, self.output_dim)
		elif self.type_AE == 'AE_simple':
			self.autoencoder = AutoEncoderSimple(self.input_dim, self.latent_dim, self.output_dim)
		elif self.type_AE == 'VAE':
			self.autoencoder = VariationalAutoencoder(self.input_dim, self.latent_dim, self.hidden_dim, self.output_dim)

		self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.LR)
		self.loss_func = nn.MSELoss()

	def change_config(self, key_config, new_value):
		self.config[key_config] = new_value
		self.config_model()

	def load_data(self, path):
		self.list_features, self.data_np, self.real_labels, self.timestamps, list_states = tools.load_data(path, self.tracks, self.input_type + '_')
		self.list_states = list_states

		

		self.seq_data_train, seq_labels_train, self.seq_data_test, seq_labels_test, val_set, label_val, seq_id_train, seq_id_test, val_id = tools.split_data_base(self.data_np, self.real_labels[0], self.ratio_split)

		self.data_train = []
		self.data_test = []

		self.labels_train = []
		self.labels_test = []

		for data_joint in self.seq_data_train:
			for d_joint in data_joint:
				self.data_train.append(d_joint)
		self.data_train = np.asarray(self.data_train)

		for data_joint in self.seq_data_test:
			for d_joint in data_joint:
				self.data_test.append(d_joint)
		self.data_test = np.asarray(self.data_test)

		# x_norm, self.input_mean, self.input_var = tools.standardization(self.data_train)

		if 'posture' in self.list_metric:
			self.data_train = self.prepare_data('posture', self.data_train)
			self.data_test = self.prepare_data('posture', self.data_test)

		size_data, self.input_dim = np.shape(self.data_train)
		self.data_min = np.ones((self.input_dim, 1))*(-np.pi)
		self.data_max = np.ones((self.input_dim, 1))*np.pi

		x_norm = tools.normalization(self.data_train, self.data_min, self.data_max)
		self.train_loader = torch.from_numpy(x_norm)

		if 'ergo_score' in self.list_metric:
			self.data_ergo = self.prepare_data('ergo_score', self.data_train)
			self.dim_loss = len(self.list_ergo_score)

			self.loss_data_min = np.zeros((self.dim_loss, 1))
			self.loss_data_max = np.ones((self.dim_loss, 1))*7

			x_ergo = tools.normalization(self.data_ergo, self.loss_data_min, self.loss_data_max)

			self.data_loss = torch.from_numpy(x_ergo)
			# data_ergo2 = tools.denormalization(x_ergo, self.loss_data_min, self.loss_data_max)

		elif self.output_type == 'ergo_posture':
			self.data_ergo = self.prepare_data('ergo_score', self.data_train)
			x_ergo, self.loss_data_mean, self.loss_data_var = tools.standardization(self.data_ergo)
			self.dim_loss = len(self.list_ergo_score) + self.input_dim

			self.loss_data_min = np.zeros((self.input_dim, 1))
			self.loss_data_max = np.ones((self.input_dim, 1))*7

			self.loss_data_min = np.concatenate((self.loss_data_min, np.ones((16, 1))))
			self.loss_data_max = np.concatenate((self.loss_data_max, np.ones((16, 1))*7))

			data_combine = np.concatenate((x_norm, x_ergo), axis=1)

			self.data_loss = torch.from_numpy(data_combine)

		else:
			self.data_loss = torch.from_numpy(x_norm)
			self.loss_data_min = self.data_min
			self.loss_data_max = self.data_max
			self.dim_loss = len(self.data_min)

		x_norm = tools.normalization(self.data_test, self.data_min, self.data_max)
		self.test_loader = torch.from_numpy(x_norm)

	def train_model(self, nbr_epoch = 100, list_metric = ['jointAngle']):
		loss_score = []

		skeleton = Skeleton('dhm66_ISB_Xsens.urdf')

		b_x = self.train_loader.view(-1, self.input_dim)
		b_y = self.data_loss.view(-1, self.dim_loss)

		input_data = np.copy(b_y.detach().numpy())
		input_data = tools.denormalization(input_data, self.loss_data_min, self.loss_data_max)

		list_loss = []
		for epoch in range(self.nbr_epoch):

			encoded, decoded = self.autoencoder(b_x)

			# if 'ergo_score' in self.list_metric:
			# 	data_eval = self.prepare_data('ergo_score', decoded)

			loss = self.loss_func(decoded, b_y)	  # mean square error

			self.optimizer.zero_grad()			   # clear gradients for this training step
			loss.backward()					 # backpropagation, compute gradients
			self.optimizer.step()					# apply gradients
			
			output_data = np.copy(decoded.detach().numpy())

			if self.output_type == 'ergo':
				output_data = tools.denormalization(output_data, self.loss_data_min, self.loss_data_max)

			elif self.output_type == 'ergo_posture':
				output_data = tools.denormalization(output_data, self.loss_data_min, self.loss_data_max)

				score1 = self.evaluate_model(input_data[0:66], output_data[0:66], 'jointAngle')
				score2 = self.evaluate_model(input_data[66:], output_data[66:], '')

			else:
				output_data = tools.denormalization(output_data, self.loss_data_min, self.loss_data_max)

			score = self.evaluate_model(input_data, output_data, list_metric[0])

			# loss_score.append(np.sqrt(np.square(input_data - output_data).mean()))
			loss_score.append(score)

			list_loss.append(score) 
			if epoch%100 == 0:
				print('Epoch: ', epoch, '| train loss:', loss.data.numpy(), score)

			if len(list_loss) > 500:
				del list_loss[0]
				if np.std(list_loss) < self.stop_criterion:
					return loss_score

		return loss_score

	def test_model(self, data = [], metric = ''):

		if len(data) == 0:
			b_x = self.train_loader.view(-1, self.input_dim)
			b_y = self.data_loss.view(-1, self.dim_loss)
			# b_x = self.test_loader.view(-1, self.input_dim)
			# b_y = self.test_loader.view(-1, self.input_dim)

		else:
			data_norm = tools.normalization(data, self.loss_data_min, self.loss_data_max)
			# data = np.asarray(data)
			# data_norm = np.copy(data)
			# data = data.astype(np.float32)
			# size_data, self.input_dim = np.shape(data)

			# for i in range(self.input_dim):
			# 	data_norm[:,i] = (data[:,i] - self.input_mean[i])/self.input_var[i]
			# data_norm = data_norm.astype(np.float32)

			b_x = torch.from_numpy(data_norm)
			b_y = torch.from_numpy(data_norm)

		input_data = np.copy(b_x.detach().numpy())

		encoded, decoded = self.autoencoder(b_x)
		decoded_joint = decoded.detach().numpy()
		decoded_joint = tools.denormalization(decoded_joint, self.loss_data_min, self.loss_data_max)

		score = self.evaluate_model(input_data, decoded_joint, metric)
	
		return decoded_joint, encoded.detach().numpy(), score


	def encode_data(self, input_data):
		data_norm = tools.normalization(input_data, self.loss_data_min, self.loss_data_max)
		b_x = torch.from_numpy(data_norm)

		encoded = self.encode_data(b_x)

		encoded_data = encoded.detach().numpy()
		encoded_data = tools.denormalization(encoded_data, self.loss_data_min, self.loss_data_max)

		return encoded_data


	def get_data_train(self):
		return self.data_train

	def get_data_test(self):
		return self.data_test

	def get_list_metric(self):
		return self.list_metric

	def get_config(self):
		return self.config

	def get_all_data(self):
		return self.data_np

	def get_all_labels(self):
		return self.real_labels

	def get_list_states(self):
		return self.list_states

	def decode_data(self, data, type_data = 'jointAngle'):
		data = data.astype(np.float32)
		data = torch.from_numpy(data)
		decoded_data = self.autoencoder.decode_data(data)
		decoded_data = decoded_data.detach().numpy()
		decoded_data = tools.denormalization(decoded_data, self.loss_data_min, self.loss_data_max)
		return decoded_data

	def prepare_data(self, metric, input_data):
		if metric == 'posture':
			human_pose = HumanPosture('config/mapping_joints.json')
			pool = mp.Pool(mp.cpu_count())
			result_objects = [pool.apply_async(human_pose.update_posture, args=(data, i)) for i, data in enumerate(input_data)]
			data_eval = np.asarray([r.get() for r in result_objects])

			pool.close()
			pool.join()

		elif metric == 'position':
			end_effectors = ['Right Hand', 'Left Hand', 'Right Foot', 'Left Foot']

			pool = mp.Pool(mp.cpu_count())

			result_objects = [pool.apply_async(skeleton.update_posture, args=(data, True, i)) for i, data in enumerate(input_data)]
			data_eval = [r.get() for r in result_objects]

			pool.close()
			pool.join()

		elif metric == 'ergo_score':
			self.ergo_assessment = ErgoAssessment('config/rula_config.json')
			self.list_ergo_score = self.ergo_assessment.get_list_score_name()
			self.list_ergo_score.sort()

			if(type(input_data) == torch.Tensor):

				for i, data in enumerate(input_data[0:1]):
					data_eval = tools.compute_sequence_ergo(data, i, self.list_ergo_score)

			elif(type(input_data) == np.ndarray):
				pool = mp.Pool(mp.cpu_count())
				
				result_objects = [pool.apply_async(tools.compute_sequence_ergo, args=(data, i, self.list_ergo_score)) for i, data in enumerate(input_data)]
				data_eval = [r.get() for r in result_objects]
				data_eval = np.asarray(data_eval)

				pool.close()
				pool.join()

		else:
			data_eval = np.copy(input_data)

		return data_eval


	def evaluate_model(self, input_data, output_data, metric):
		if metric in ['jointAngle', 'posture'] :
			input_joint = np.rad2deg(input_data)
			output_joint = np.rad2deg(output_data)
			score = np.sqrt(np.square(input_joint - output_joint).mean())

		elif metric == 'ergo_score':
			list_ergo_score = self.ergo_assessment.get_list_score_name()
			pool = mp.Pool(mp.cpu_count())

			result_objects = [pool.apply_async(tools.compute_sequence_ergo, args=(data, i, list_ergo_score)) for i, data in enumerate(input_data)]
			output_ergo = [r.get() for r in result_objects]

			pool.close()
			pool.join()

			output_ergo = np.asarray(output_ergo)

			score = np.sqrt(np.square(input_ergo - output_ergo).mean())

		elif metric == 'end_effectors':
			for i in range(len(input_data)):
				pool = mp.Pool(mp.cpu_count())

				result_objects = [pool.apply_async(skeleton.update_posture, args=(data, True, i)) for i, data in enumerate(output_data)]
				output_position = [r.get() for r in result_objects]

				pool.close()
				pool.join()

			score = np.sqrt(np.square(input_position - output_position).mean())


		elif metric == 'position':
			for i in range(len(input_data)):
				pool = mp.Pool(mp.cpu_count())

				result_objects = [pool.apply_async(skeleton.update_posture, args=(data, True, i)) for i, data in enumerate(output_data)]
				output_position = [r.get() for r in result_objects]

				pool.close()
				pool.join()

			score = np.sqrt(np.square(input_position - output_position).mean())

		else:
			score = np.sqrt(np.square(input_data - output_data).mean())
		return score



	

		



