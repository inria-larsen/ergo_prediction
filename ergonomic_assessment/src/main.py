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
	parser.add_argument('--file', '-f', help='Configuration file', default="config/config_AE.json")
	parser.add_argument('--config', '-c', help='Configuration type', default="DEFAULT")

	loss = []

	nbr_iterations = 10

	list_size = [2, 3, 5, 7, 10, 15]

	for size_latent in list_size:

		for k in range(nbr_iterations):
			autoencoder = AE.ModelAutoencoder(parser, local_path)
			list_metric = autoencoder.get_list_metric()
			metric = list_metric[0]

			# size_latent = autoencoder.get_config()['latent_dim']
			type_AE = autoencoder.get_config()['type_AE']

			path = local_path + "/save/" + type_AE + "/" + metric + '/' + str(size_latent) + '/'

			if not(os.path.isdir(path)):
				os.mkdir(path)
			if not(os.path.isdir(path + 'loss/')):
				os.mkdir(path + 'loss/')
			if os.path.exists(path + "autoencoder_" + str(size_latent) + '_' + str(k) + ".pkl"):
				continue

			loss = autoencoder.train_model(list_metric=list_metric)
			all_data_test = autoencoder.get_data_test()

		
			pickle.dump(autoencoder, open(path + "autoencoder_" + str(size_latent) + '_' + str(k) + ".pkl", "wb" ))
			pickle.dump(loss, open(path + 'loss/' + "loss_" + str(size_latent) + '_' + str(k) + ".pkl", "wb" ))

			del autoencoder

	plt.show()