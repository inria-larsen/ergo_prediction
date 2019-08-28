#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import os
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../ergonomic_assessment/src/')))

import AE
import tools
from Skeleton import Skeleton
from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture

class RealTimePlotModule():
	"""
	This module plots a bar chart with the probability distribution on the states.
	Usage
	python plot_probabilities.py
	Input port: /processing/NamePort:o
	"""
	def __init__(self):
		pg.mkQApp()

		self.port = yarp.BufferedPortBottle()
		self.port.open('/plot_latentspace')

		metric = 'jointAngle'
		ergo_name = ['RULA_SCORE']

		size_latent = 2
		dx = 0.1

		loss = [[]]
		autoencoder = []

		all_score = []
		all_size = []
		type_data = []
		path_src = "/home/amalaise/Documents/These/code/ergo_prediction/ergonomic_assessment/src/"
		path = path_src + "save/AE/" + metric + "/" + str(size_latent) + '/'
		list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		list_files.sort()
		file = list_files[0]

		with open(path + file, 'rb') as input:
			autoencoder = pickle.load(input)

		input_data = autoencoder.get_data_test()
		data_output, encoded_data, score = autoencoder.test_model(input_data)
		score = autoencoder.evaluate_model(input_data, data_output, metric)
		
		Max = np.max(encoded_data, axis = 0)
		Min = np.min(encoded_data, axis = 0)
		Mean = np.mean(encoded_data, axis = 0)

		# Compute ergo score
		ergo_assessment = ErgoAssessment(path_src + 'config/rula_config.json')
		list_ergo_score = ergo_assessment.get_list_score_name()
		list_ergo_score.sort()

		reduce_posture = HumanPosture(path_src + 'config/mapping_joints.json')
		posture = Skeleton('dhm66_ISB_Xsens.urdf')

		self.X = np.arange(0.0, 1.0 + dx, dx)

		self.ergo_grid = np.zeros((len(self.X), len(self.X)))

		for i, data_x in enumerate(self.X):
			for j, data_y in enumerate(self.X):

				x = np.zeros((1,size_latent))
				x[0, 0] = data_x
				x[0, 1] = data_y

				decoded_data = autoencoder.decode_data(x)
				if metric == 'posture':
					whole_body = reduce_posture.reduce2complete(decoded_data[0])
					posture.update_posture(whole_body)
				else:
					posture.update_posture(decoded_data[0])

				self.ergo_grid[j,i] = tools.compute_sequence_ergo(decoded_data[0], 0, ergo_name, path_src)[0]


		# plt.ion()
		# self.fig = plt.figure()
		# self.ax = self.fig.add_subplot(111)
		self.imv = pg.ImageView()
		self.imv.show()

		self.plot_latent_space()


	def plot_latent_space(self, x=0, y=0):
		
		
		self.imv.setImage(self.ergo_grid)

		pg.plot(x, y, pen=None, symbol='o') 

		
		# self.ax.cla()
		# plt.sca(self.ax)

		# cax = self.ax.matshow(self.ergo_grid, cmap=plt.cm.Reds)
		# self.fig.colorbar(cax, ax = self.ax)

		# labels = X[0::2].tolist()
		# labels = ['{:.2f}'.format(name) for name in labels]

		# self.ax.set_xticklabels(['']+labels)
		# self.ax.set_yticklabels(['']+labels)

		# self.ax.set_title('Ergonomic score in latent space')

		


	def update_plot(self):
		# b_in = self.port.read()
		# data = b_in.toString().split(' ')

		self.plot_latent_space()

		
		# print(data)



		

		# del data[0]

		# data = list(map(float, data))
		# data = np.asarray(data)

		# self.ax.scatter(data[0]*len(self.X), data[1]*len(self.X))

		# pg.plot(data[0]*len(self.X), data[1]*len(self.X), pen=None, symbol='o') 

		# self.fig.canvas.draw()


		QtGui.QApplication.processEvents()

		# b_in = self.port.read()
		# data = b_in.toString().split(' ')

		# if len(data) == 67:
		# 	del data[0]

		# data = list(map(float, data))
		# data = np.asarray(data)

		# self.ax.cla()
		# plt.sca(self.ax)
		# data = np.deg2rad(data)
		# self.skeleton.visualise_from_joints(data)

		# self.fig.canvas.draw()

		
        # plt.pause(0.001)

		# if(self.flag_init == 0):
		# 	for dim in range(dimension):
		# 		self.list_curve.append(self.plotData.plot(pen=(dim,dimension)))
		# 	self.flag_init = 1

		# value = list(map(float, data))

		# for dim in range(dimension):
		# 	if(len(self.buffer) <= dim):
		# 		self.buffer.append([])

		# 	self.buffer[dim].append(value[dim])
		# 	if(len(self.buffer[dim]) > self.size_window):
		# 		del self.buffer[dim][0]

		# for dim in range(dimension):
		# 	self.list_curve[dim].setData(self.buffer[dim])

		# QtGui.QApplication.processEvents()
		return

	def close(self):
		yarp.Network.disconnect(self.input_port, self.port.getName())
		self.port.close()


if __name__=="__main__":
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.configure(sys.argv)

	fig = RealTimePlotModule()

	while(True):
		try:
			fig.update_plot()
			i = 0
		except KeyboardInterrupt:
			fig.close()
			break

