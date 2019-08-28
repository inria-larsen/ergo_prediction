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
	def __init__(self, name_port):
		# pg.mkQApp()

		self.skeleton = Skeleton('dhm66_ISB_Xsens.urdf')

		self.input_port = name_port
		self.port = yarp.BufferedPortBottle()
		self.port.open('/plot_skeleton' + name_port)

		plt.ion()

		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection='3d')


		yarp.Network.connect(name_port, self.port.getName())

		self.flag_init = 0


		# view = gl.GLViewWidget()
		# view.show()

		## create three grids, add each to the view
		# xgrid = gl.GLGridItem()
		# ygrid = gl.GLGridItem()
		# zgrid = gl.GLGridItem()
		# view.addItem(xgrid)
		# view.addItem(ygrid)
		# view.addItem(zgrid)

		# ## rotate x and y grids to face the correct direction
		# xgrid.rotate(90, 0, 1, 0)
		# ygrid.rotate(90, 1, 0, 0)

		# ## scale each grid differently
		# xgrid.scale(0.2, 0.1, 0.1)
		# ygrid.scale(0.2, 0.1, 0.1)
		# zgrid.scale(0.1, 0.2, 0.1)


	def update_plot(self):
		b_in = self.port.read()
		data = b_in.toString().split(' ')

		if len(data) == 67:
			del data[0]

		data = list(map(float, data))
		data = np.asarray(data)

		self.ax.cla()
		plt.sca(self.ax)
		data = np.deg2rad(data)
		self.skeleton.visualise_from_joints(data)

		self.fig.canvas.draw()

		
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
	
	name_port = rf.find("name_port").toString()

	fig = RealTimePlotModule(name_port)

	while(True):
		try:
			fig.update_plot()
			i = 0
		except KeyboardInterrupt:
			fig.close()
			break

