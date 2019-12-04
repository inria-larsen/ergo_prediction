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
		self.app = pg.mkQApp()
		pg.setConfigOption('background', 'w')
		pg.setConfigOption('foreground', 'k')

		self.view = pg.PlotWidget()
		self.view.resize(800, 600)
		self.view.setWindowTitle('Ergonomic score in latent space')
		self.view.setAspectLocked(True)
		self.view.show()

		self.port = yarp.BufferedPortBottle()
		self.port.open('/plot_latentspace')

		metric = 'jointAngle'
		ergo_name = ['TABLE_REBA_C']

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

		self.X = np.arange(0.0, 1.0+dx, dx)

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

				ergo_score = tools.compute_sequence_ergo(decoded_data[0], 0, ergo_name, path_src)[0]
				if ergo_score == 1:
					ergo_score = 1
				elif 1 < ergo_score < 5:
					ergo_score = 2
				elif 4 < ergo_score < 6:
					ergo_score = 3
				else:
					ergo_score = 4

				self.ergo_grid[j,i] = ergo_score

		self.flag = 0

		self.plot_latent_space()


	def plot_latent_space(self, x=0, y=0):
		if self.flag:
			self.view.removeItem(self.scatter)
		else:
			self.flag = 1

		self.scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=10, color='r'), symbol='o', size=1)
		plot_traj = pg.PlotItem(pen=pg.mkPen(width=5, color='r'), size=1)

		img_np = np.rot90(np.rot90(np.rot90(self.ergo_grid)))
		
		img = pg.ImageItem(img_np)

		self.scatter.setData(x=[x], y=[y])
		plot_traj.setData(x, y)

		img.setZValue(-100)
		self.view.addItem(img)
		self.view.addItem(self.scatter)
		
	
	def update(self):
		b_in = self.port.read()
		data = b_in.toString().split(' ')

		del data[0]

		data = list(map(float, data))
		data = np.asarray(data)

		self.plot_latent_space(x=data[0]*len(self.X), y=len(self.X)-data[1]*len(self.X))

		QtGui.QApplication.processEvents()

		return

	def close(self):
		yarp.Network.disconnect(self.input_port, self.port.getName())
		self.port.close()
		sys.exit(self.app.exec_())


if __name__=="__main__":
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.configure(sys.argv)

	fig = RealTimePlotModule()

	while(True):
		try:
			fig.update()
			i = 0
		except KeyboardInterrupt:
			fig.close()
			break

