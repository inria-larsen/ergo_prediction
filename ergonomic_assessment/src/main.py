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
	autoencoder = AE.ModelAutoencoder(parser)

	list_metric = ['jointAngle']

	size_list = [1, 2, 5, 10, 20, 30, 45, 66]
	loss = [[]]

	for metric in list_metric:
		for i, size in enumerate(size_list):
			autoencoder.change_config('hidden_dim', size)
			loss[i] = autoencoder.train_model(list_metric=list_metric)
			loss.append([])
			path = local_path + "/save/" + metric + "/"
			if not os.path.exists(path):
				os.mkdir(path)
			pickle.dump(autoencoder, open(path + "autoencoder_" + str(size) + ".pkl", "wb" ) )
			pickle.dump(loss[i], open(path + "loss_" + str(size) + ".pkl", "wb" ) )

		del loss[-1]

	# skeleton = Skeleton('dhm66_ISB_Xsens.urdf')
	# data = autoencoder.get_data_test()

	for metric in list_metric:

		fig1 = plt.figure()
		lines = []
		for i, size in enumerate(size_list):
			line, = plt.plot(loss[i][metric], label = str(size))
		plt.legend()

	# skeleton.animate_skeleton([seq_data_test[0][0::40], decoded_joint[0::40]], color=color, save=True)
	plt.show()