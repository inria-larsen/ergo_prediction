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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 

# Define a function for the line plot with intervals
def lineplotCI(mean_data, low_CI, upper_CI, ax, color, label):
	# Create the plot object
	

	# Plot the data, set the linewidth, color and transparency of the
	# line, provide a label for the legend
	plt.plot(mean_data, lw = 1, color = color, alpha = 1, label = label)
	# Shade the confidence interval
	x_data = np.arange(0, len(mean_data))
	plt.fill_between(x_data, low_CI, upper_CI, color = color, alpha = 0.4)
	# Label the axes and provide a title
	# ax.set_title(title)
	# ax.set_xlabel(x_label)
	# ax.set_ylabel(y_label)

	# # Display legend
	# ax.legend(loc = 'best')

if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('--config', '-c', help='Type AE', default="AE")

	args=parser.parse_args()
	type_AE = args.config

	list_metric = ['jointAngle']

	size_list = [2, 3, 5, 7, 10]
	loss = [[]]
	autoencoder = []

	metric = 'jointAngle'

	# for metric in list_metric:
	for i, size in enumerate(size_list):
		path = "save/" + type_AE + "/" + metric + "/" + str(size) + '/loss/'
		# with open(path + "autoencoder_" + str(size) + ".pkl", 'rb') as input:
		# 	autoencoder.append(pickle.load(input))
		# list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path+ 'loss/', f))]
		list_files = [f for f in os.listdir(path)]

		for file in list_files:
			with open(path + file, 'rb') as input:
				data_loss = pickle.load(input)
				loss[i].append(data_loss)

		loss.append([])

	del loss[-1]


	# for metric in list_metric:

	lines = []
	_, ax = plt.subplots()
	color = ['r', 'b', 'g', 'y', 'c']
	for i, size in enumerate(size_list):

		df = pd.DataFrame(loss[i])
		CI_df = pd.DataFrame(columns = ['mean', 'low_CI', 'upper_CI'])
		CI_df['mean'] = df.median()
		CI_df['low_CI'] = df.quantile(0.25)
		CI_df['upper_CI'] = df.quantile(0.75)

		lineplotCI(mean_data = CI_df['mean']
			   , low_CI = CI_df['low_CI']
			   , upper_CI = CI_df['upper_CI']
			   , ax = ax
			   , color = color[i]
			   , label = str(size))

	plt.title('Loss function on joint angle reconstruction')
	plt.xlabel('Epoch')
	plt.ylabel('Degree')
	plt.legend()

# 	id_model = 2
# 	skeleton = Skeleton('dhm66_ISB_Xsens.urdf')
# 	path = "save/" + type_AE + "/" + metric + "/" + str(id_model) + '/'
# #	path = "save/AE/" + metric + "/" + str(id_model) + '/'
# 	print(path)
# 	with open(path + "autoencoder_" + str(id_model) + "_0.pkl", 'rb') as input:
# 		autoencoder = pickle.load(input)

# 	input_data = autoencoder.get_data_test()[200:500]
# 	data_output, encoded, score = autoencoder.test_model(input_data)


# 	color = ['b', 'r']

# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')


# 	skeleton.animate_skeleton([input_data, data_output], color=color, save=False)

	plt.show()