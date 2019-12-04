from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture
from Skeleton import Skeleton
from xsens_parser import mvnx_tree
import AE
from sklearn import preprocessing
import seaborn as sns

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
from pandas_ml import ConfusionMatrix

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
	list_metric = ['jointAngle']

	flag_ergo = True

	size_list = [2, 3, 5, 7, 10]
	# size_list = [10]
	loss = [[]]
	autoencoder = []

	# id_model = 2
	skeleton = Skeleton('dhm66_ISB_Xsens.urdf')

	all_score = []
	all_size = []
	type_data = []
	df_score = pd.DataFrame()

	for metric in list_metric:
		print(metric)
		for k, size in enumerate(size_list):
			print(size)
			path = "save/AE/" + metric + "/" + str(size) + '/'
			list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
			list_files.sort()
			for file in list_files[1:2]:
				print(file)
				with open(path + file, 'rb') as input:
					autoencoder = pickle.load(input)

				input_data = autoencoder.get_data_test()
				# data = np.deg2rad(data)
				data_output, encoded_data, score = autoencoder.test_model(input_data)
				score = autoencoder.evaluate_model(input_data, data_output, metric)
				# data_output, encoded_data, score = autoencoder.test_model()

				if flag_ergo == True:
					ergo_assessment = ErgoAssessment('config/rula_config.json')
					list_ergo_score = ergo_assessment.get_list_score_name()
					list_ergo_score.sort()

					pool = mp.Pool(mp.cpu_count())

					result_objects = [pool.apply_async(tools.compute_sequence_ergo, args=(data, i, list_ergo_score)) for i, data in enumerate(input_data)]
					input_ergo = [r.get() for r in result_objects]
					input_ergo = np.asarray(input_ergo)

					pool.close()
					pool.join()

					pool = mp.Pool(mp.cpu_count())

					result_objects = [pool.apply_async(tools.compute_sequence_ergo, args=(data, i, list_ergo_score)) for i, data in enumerate(data_output)]
					output_ergo = [r.get() for r in result_objects]
					output_ergo = np.asarray(output_ergo)

					pool.close()
					pool.join()

					level_ergo_input = []
					level_ergo_output = []
					
					for i in range(len(list_ergo_score)):
						if list_ergo_score[i] == 'RULA_SCORE':
							# for j in range(len(input_ergo)):
							# 	if input_ergo[j,i] <= 2:
							# 		level_ergo_input.append('Green')
							# 	elif 2 < input_ergo[j,i] <= 4:
							# 		level_ergo_input.append('Yellow')
							# 	elif 2 < input_ergo[j,i] <= 4:
							# 		level_ergo_input.append('Orange')
							# 	else:
							# 		level_ergo_input.append('Red')

							# 	if output_ergo[j,i] <= 2:
							# 		level_ergo_output.append('Green')
							# 	elif 2 < output_ergo[j,i] <= 4:
							# 		level_ergo_output.append('Yellow')
							# 	elif 4 < output_ergo[j,i] <= 6:
							# 		level_ergo_output.append('Orange')
							# 	else:
							# 		level_ergo_output.append('Red')

							length = int(len(input_ergo)/10)
							for j in range(10):
								data_input = input_ergo[j*length:j*length+length]
								data_output = output_ergo[j*length:j*length+length]
								all_score.append((abs(data_input[:,i] - data_output[:,i])).mean())
								
					all_size.append(size)
					if metric == 'jointAngle':
						type_data.append('Xsens Model')
					elif metric == 'posture':
						type_data.append('Reduced Model')

				else:
					all_score.append(score)



	# confusion_matrix = ConfusionMatrix(level_ergo_input, level_ergo_output)
	# # print(confusion_matrix(level_ergo_input, level_ergo_output))
	# confusion_matrix.plot()
	# fig.savefig('confusion_rula.png')

	df_score['Error (degree)'] = all_score
	df_score['Size'] = all_size
	df_score['Type_data'] = type_data

	# pickle.dump(df_score, open("save/RULA_error.pkl", "wb" ))

	fig = plt.figure()
	ax = sns.boxplot(x = df_score['Size'], y = df_score['Error (degree)'], hue = df_score['Type_data'])
	plt.title('Reconstruction error')
	plt.legend()
	# fig.savefig('error_reconstruction_H.png')
	plt.show()