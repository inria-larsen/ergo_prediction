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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../ergonomic_assessment/src/')))

import tools
from Skeleton import Skeleton



class RealTimePlotModule():
	"""
	This module plots a bar chart with the probability distribution on the states.
	Usage
	python plot_probabilities.py
	Input port: /processing/NamePort:o
	"""
	def __init__(self, list_port):
		self.input_port = []
		self.port = []

		plt.ion()

		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection='3d')

		data = np.zeros((2,66))

		color = ['b', 'r']
		self.lines = []
		self.skeleton = []

		self.skeleton = Skeleton('dhm66_ISB_Xsens.urdf')

		for i, name_port in enumerate(list_port):	
			self.input_port.append(name_port)
			self.port.append(yarp.BufferedPortBottle())
			self.port[i].open('/plot_skeleton' + name_port)

			yarp.Network.connect(name_port, self.port[i].getName())

		self.lines = self.skeleton.visualise_from_joints(data, color_list=color, lines = [])
			# self.skeleton[i].visualise_from_joints(data, color=color[i])


	def update_plot(self):
		color = ['b', 'r']
		data_joint = []
		for i, port in enumerate(self.port):
			b_in = self.port[i].read()
			data = b_in.toString().split(' ')

			if len(data) == 67:
				del data[0]

			data = list(map(float, data))
			data = np.asarray(data)

			self.ax.cla()
			plt.sca(self.ax)
			data_joint.append(np.deg2rad(data))

			# self.lines[i] = self.skeleton[i].visualise_from_joints(data, color=color[i], lines=self.lines[i])
		self.skeleton.visualise_from_joints(data_joint, color_list=color, lines=self.lines)

		self.fig.canvas.draw()

		return

	def close(self):
		for i_port, port in zip(self.input_port, self.port):
			yarp.Network.disconnect(i_port, port.getName())
			port.close()


if __name__=="__main__":
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.configure(sys.argv)
	
	# name_port = rf.find("name_port").toString()

	list_port = ['/processing/xsens/JointAngles:o', '/AE/reconstruct:o']
	# list_port = ['/processing/xsens/JointAngles:o']

	fig = RealTimePlotModule(list_port)

	while(True):
		try:
			fig.update_plot()
			i = 0
		except KeyboardInterrupt:
			fig.close()
			break

