from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture
from Skeleton import Skeleton
import AE
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import torch
import torch.nn as nn
import tools
import random
import pandas as pd

from discrete_promp import DiscretePROMP
from linear_sys_dyn import LinearSysDyn
from promp_ctrl import PROMPCtrl

import os
import argparse
import configparser

from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

def plot_mean_and_sigma(mean, lower_bound, upper_bound, color_mean=None, color_shading=None):
	# plot the shaded range of the confidence intervals
	plt.fill_between(range(mean.shape[0]), lower_bound, upper_bound, color=color_shading, alpha=.5)
	# plot the mean on top
	plt.plot(mean, color_mean)

if __name__ == '__main__':
	local_path = os.path.dirname(os.path.abspath(__file__))
	# Get arguments
	parser=argparse.ArgumentParser()
	parser.add_argument('--file', '-f', help='Configuration file', default="config/config_AE.json")
	parser.add_argument('--config', '-c', help='Configuration type', default="DEFAULT")

	tracks = ['general_posture']

	path_data = 'database/win_size250/'

	list_features, all_data, all_labels, timestamps, list_states = tools.load_data(path_data, tracks)

	list_states = list_states.tolist()

	data_tasks = [[]]

	for id_state, state in enumerate(list_states):
		data_tasks.append([])
		data_tasks[id_state].append([])
	del data_tasks[-1]

	for data_seq, labels_seq in zip(all_data, all_labels):
		prev_state = labels_seq[0]
		for data, label in zip(data_seq, labels_seq):
			id_state = list_states.index(label)

			if not(label == prev_state):
				id_prev_state = list_states.index(prev_state)
				if len(data_tasks[id_prev_state][-1]) < 10:
					del data_tasks[id_prev_state][-1]
				data_tasks[id_prev_state].append([])

			data_tasks[id_state][-1].append(data)

			prev_state = label

	config_ergo = 'rula'
	metric = 'jointAngle'
	if config_ergo == 'reba':
		ergo_name = ['TABLE_REBA_C']
	else:
		ergo_name = ['RULA_SCORE']

	size_latent = 2
	dx = 0.02

	loss = [[]]
	autoencoder = []

	all_score = []
	all_size = []
	type_data = []

	# path = "save/AE/" + metric + "/" + str(size_latent) + '/'
	# list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	# list_files.sort()
	# file = list_files[0]

	file = '/home/amalaise/Documents/These/code/ergo_prediction/ergonomic_assessment/src/save/AE/jointAngle/2/autoencoder_2_0.pkl'

	with open(file, 'rb') as input:
		autoencoder = pickle.load(input)

	# data_output, encoded_data, score = autoencoder.test_model(input_data)

	# Compute ergo score
	ergo_assessment = ErgoAssessment('config/' + config_ergo + '_config.json')
	list_ergo_score = ergo_assessment.get_list_score_name()
	list_ergo_score.sort()

	reduce_posture = HumanPosture('config/mapping_joints.json')
	posture = Skeleton('dhm66_ISB_Xsens.urdf')

	X = np.arange(0.0, 1.0 + dx, dx)

	path_src = "/home/amalaise/Documents/These/code/ergo_prediction/ergonomic_assessment/src/"

	pd_ergo_grid = pd.read_csv(config_ergo + '_grid.csv', index_col=False)
	ergo_grid = np.asarray(pd_ergo_grid)


	fig = plt.figure()
	# ax1 = Axes3D(fig)
	
	# ax1 = fig.add_subplot(122, projection='3d')
	# plt.imshow(ergo_grid, extent = [X[0] , X[-1]-X[0], X[-1]-X[0] , X[0]])
	# plt.imshow(ergo_grid)

	pd_ergo_grid = pd.read_csv(config_ergo + '_grid.csv', index_col=False)
	ergo_grid = np.asarray(pd_ergo_grid)

	# cax = ax.matshow(ergo_grid, cmap=plt.cm.Reds)

	colour = [ "red", "black", "green", "yellow", "purple", "orange", "b", "c" ]

	data_latent = []

	min_len = 500

	for id_state, state in enumerate(list_states):
		if not(state=='Re'):
			continue

		ax = fig.add_subplot(len(list_states)/2,2,id_state+1)

		# id_samples = random.sample(range(0, len(data_tasks[id_state])), len(data_tasks[id_state]))
		id_samples = random.sample(range(0, len(data_tasks[id_state])), len(data_tasks[id_state]))
		# id_samples = [0]

		#create a promb object by passing the data
		# d_promp = DiscretePROMP(data=data_tasks[id_state])
		# d_promp.train()
		
		for sample in id_samples:
			id_state = list_states.index(state)
			if len(data_tasks[id_state][sample]) == 0:
				continue
			size_data, input_dim = np.shape(data_tasks[id_state][sample])
			data_min = np.ones((input_dim, 1))*(-np.pi)
			data_max = np.ones((input_dim, 1))*np.pi

			data_output, encoded_data, score = autoencoder.test_model(np.asarray(data_tasks[id_state][sample]))

			encoded_data[:,0]=encoded_data[:,0]
			encoded_data[:,1]=1-encoded_data[:,1]

			if state == 'Re':
				data_latent.append(encoded_data[:,0])
				if len(data_latent[-1]) < min_len:
					min_len = len(data_latent[-1])

			plt.plot(encoded_data[:,0], encoded_data[:,1], colour[id_state], label = state)
			plt.scatter(encoded_data[0,0], encoded_data[0,1], color = 'g', marker = '^')
			plt.scatter(encoded_data[-1,0], encoded_data[-1,1], color = 'r', marker = 's')
			plt.xlim(0,1)
			plt.ylim(0,1)
			ax.set_title(state)


	data_norm = np.zeros((len(data_latent), min_len))

	plt.figure()
	for i, trj_i in enumerate(data_latent):
		delta = (len(trj_i)-1)/(min_len-1)
		for j in range(min_len):
			data_norm[i,j] = trj_i[int(j*delta)]

		plt.plot(data_norm[i])

	d_promp = DiscretePROMP(data=data_norm)
	d_promp.train()

	d_promp.set_start(data_norm[0][0])
	d_promp.set_goal(data_norm[0][-1])

	lsd = LinearSysDyn()

	promp_ctl = PROMPCtrl(promp_obj=d_promp)
	promp_ctl.update_system_matrices(A=lsd._A, B=lsd._B)

	ctrl_cmds_mean, ctrl_cmds_sigma = promp_ctl.compute_ctrl_traj(state_list=['Re'])

	for k in range(lsd._action_dim):
		
		mean		= ctrl_cmds_mean[:, k]
		lower_bound = mean - 3.*ctrl_cmds_sigma[:, k, k]
		upper_bound = mean + 3*ctrl_cmds_sigma[:, k, k]

		plot_mean_and_sigma(mean=mean, lower_bound=lower_bound, upper_bound=upper_bound, color_mean='g', color_shading='g')

	plt.plot(action, 'r')



	# plt.figure()
	# plt.plot(data_latent)


	# plt.legend()
	# plt.title('Trajectories in latent space')

	plt.show()
