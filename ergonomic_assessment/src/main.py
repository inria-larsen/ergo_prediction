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

import os
import argparse
import configparser

from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

if __name__ == '__main__':
	local_path = os.path.dirname(os.path.abspath(__file__))
	# Get arguments
	parser=argparse.ArgumentParser()
	autoencoder = AE.ModelAutoencoder(parser, local_path)

	list_metric = autoencoder.get_list_metric()

	size_list = [2, 3, 5, 7, 10]
	loss = []

	all_data_test = autoencoder.get_data_test()

	metric = 'jointAngle'

	for i, size in enumerate(size_list):
		path = local_path + "/save/AE/" + metric + '/' + str(size) + '/'

		count = 1
		# while os.path.isdir(path):
		# 	count += 1
		# 	path += str(count) + '/'
		if not(os.path.isdir(path)):
			os.mkdir(path)

		for k in range(10):
			autoencoder.change_config('latent_dim', size)
			loss = autoencoder.train_model(list_metric=list_metric)
			# score_metric = autoencoder.test_model(list_metric=list_metric)
				
			pickle.dump(autoencoder, open(path + "autoencoder_" + str(size) + '_' + str(k) + ".pkl", "wb" ))
			pickle.dump(loss, open(path + "loss_" + str(size) + '_' + str(k) + ".pkl", "wb" ))







	# for metric in list_metric:

	# 	fig1 = plt.figure()
	# 	lines = []
	# 	for i, size in enumerate(size_list):
	# 		line, = plt.plot(loss[i][metric], label = str(size))
	# 	plt.legend()

	# plt.show()