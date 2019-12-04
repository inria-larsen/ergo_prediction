import AE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import pandas as pd

import os

from Skeleton import Skeleton
from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture
import tools

if __name__ == '__main__':
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

	path = "save/AE/" + metric + "/" + str(size_latent) + '/'
	list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	list_files.sort()
	file = list_files[0]

	with open(path + file, 'rb') as input:
		autoencoder = pickle.load(input)

	input_data = autoencoder.get_data_test()
	# data = np.deg2rad(data)
	data_output, encoded_data, score = autoencoder.test_model(input_data)
	score = autoencoder.evaluate_model(input_data, data_output, metric)
			# data_output, encoded_data, score = autoencoder.test_model()

			# ergo_assessment = ErgoAssessment('config/rula_config.json')

			# list_ergo_score = ergo_assessment.get_list_score_name()
			# list_ergo_score.sort()

			# all_score.append(score)
	
	Max = np.max(encoded_data, axis = 0)
	Min = np.min(encoded_data, axis = 0)
	Mean = np.mean(encoded_data, axis = 0)

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

	# ergo_grid = np.zeros((len(X), len(X)))

	# for i, data_x in enumerate(X):
	# 	for j, data_y in enumerate(X):

	# 		x = np.zeros((1,size_latent))
	# 		x[0, 0] = data_x
	# 		x[0, 1] = data_y
	# 		# x[0, 2] = 0.5
	# 		# print(x)
	# 		decoded_data = autoencoder.decode_data(x)
	# 		if metric == 'posture':
	# 			whole_body = reduce_posture.reduce2complete(decoded_data[0])
	# 			posture.update_posture(whole_body)
	# 		else:
	# 			posture.update_posture(decoded_data[0])

	# 		# ergo_grid[j,i] = tools.compute_sequence_ergo(decoded_data[0], 0, ergo_name, '')[0]
	# 		ergo_score = tools.compute_sequence_ergo(ergo_assessment, decoded_data[0], 0, ergo_name, path_src)[0]

	# 		if config_ergo == 'reba':
	# 			if ergo_score == 1:
	# 				ergo_score = 1
	# 			elif 1 < ergo_score < 5:
	# 				ergo_score = 2
	# 			elif 4 < ergo_score < 6:
	# 				ergo_score = 3
	# 			else:
	# 				ergo_score = 4

	# 		ergo_grid[j,i] = ergo_score

			# print(ergo_score)
			
	


	fig = plt.figure()
	# ax1 = Axes3D(fig)
	ax0 = fig.add_subplot(121)
	ax1 = fig.add_subplot(122, projection='3d')
	# plt.imshow(ergo_grid, extent = [X[0] , X[-1]-X[0], X[-1]-X[0] , X[0]])
	# plt.imshow(ergo_grid)


	# plt.scatter(encoded_data[:,0], encoded_data[:,1])

	cax = ax0.matshow(ergo_grid, cmap=plt.cm.Reds)

	# for i in range(len(X)):
	#     for j in range(len(X)):
	#         c = ergo_grid[j,i]
	#         ax.text(i, j, str(c), va='center', ha='center')
	fig.colorbar(cax, ax = ax0)

	labels = X[0::2].tolist()
	labels = ['{:.2f}'.format(name) for name in labels]

	ax0.set_xticklabels(['']+labels)
	ax0.set_yticklabels(['']+labels)

	ax0.set_title('Ergonomic score in latent space')

	def onclick(event):
		ax1.cla()
		plt.sca(ax1)

		x = np.zeros((1,size_latent))
		x[0, 0] = event.xdata/len(X)
		x[0, 1] = event.ydata/len(X)

		for i in range(size_latent):
			if x[0, i] < 0:
				x[0, i] = 0.0
			elif x[0, i] > 1.0:
				x[0, i] = 1.0

		# x[0, 0] = 0.5
		decoded_data = autoencoder.decode_data(x)
		if metric == 'posture':
			whole_body = reduce_posture.reduce2complete(decoded_data[0])
			posture.update_posture(whole_body)
		# elseax1:
			posture.update_posture(decoded_data[0])
		ergo_score = tools.compute_sequence_ergo(ergo_assessment, decoded_data[0], 0, ergo_name, '')
		posture.visualise_from_joints(ax1, [decoded_data[0]])

		plt.sca(ax0)
		ax0.scatter(x[0, 0]*len(X), x[0, 1]*len(X))

		plt.show()

	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	plt.sca(ax1)
	posture.visualise_from_joints(ax1, [np.zeros((66))])

	# pd.DataFrame(ergo_grid).to_csv(config_ergo + "_grid.csv", header=False, index=False)

	plt.show()
