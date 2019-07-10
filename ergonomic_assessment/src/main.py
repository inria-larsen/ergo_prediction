from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture
from Skeleton import Skeleton
from xsens_parser import mvnx_tree
import AE
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn
import tools
import random

import os
import argparse
import configparser

import visualization_tools as vtools
from copy import deepcopy

from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

import torchvision.transforms.functional as TF
import multiprocessing as mp

if __name__ == '__main__':
	local_path = os.path.dirname(os.path.abspath(__file__))
	# Get arguments
	parser=argparse.ArgumentParser()
	autoencoder = AE.ModelAutoencoder(parser, local_path)

	list_metric = autoencoder.get_list_metric()

	size_list = [1, 2, 5, 10, 20, 30, 45, 66]
	loss = []

	all_data_test = autoencoder.get_data_test()

	for metric in list_metric:
		test_data = autoencoder.prepare_data(metric, all_data_test)

		for i, size in enumerate(size_list):
			autoencoder.change_config('latent_variable_dim', size)
			loss = autoencoder.train_model(list_metric=list_metric)
			# score_metric = autoencoder.test_model(list_metric=list_metric)
			path = local_path + "/save/" + metric + "/"
			if not os.path.exists(path):
				os.mkdir(path)
			pickle.dump(autoencoder, open(path + "autoencoder_" + str(size) + ".pkl", "wb" ) )
			pickle.dump(loss, open(path + "loss_" + str(size) + ".pkl", "wb" ) )

	# for metric in list_metric:

	# 	fig1 = plt.figure()
	# 	lines = []
	# 	for i, size in enumerate(size_list):
	# 		line, = plt.plot(loss[i][metric], label = str(size))
	# 	plt.legend()

	# plt.show()