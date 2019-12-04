import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib
from matplotlib.figure import Figure

import random

import yarp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../ergonomic_assessment/src/')))

import tools
from Skeleton import Skeleton

matplotlib.use('QT5Agg')


class SkeletonWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        list_port = ['/processing/xsens/JointAngles:o', '/AE/reconstruct:o']

        self.input_port = []
        self.port = []

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        data = np.zeros((1,66))

        color = ['b', 'r']
        self.lines = []
        self.skeleton = []

        self.skeleton = Skeleton('dhm66_ISB_Xsens.urdf')

        for i, name_port in enumerate(list_port):   
            self.input_port.append(name_port)
            self.port.append(yarp.BufferedPortBottle())
            self.port[i].open('/plot_skeleton' + name_port)

            yarp.Network.connect(name_port, self.port[i].getName())

        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget

        self.fig = Figure()
        
        self.pltCanv = Canvas(self.fig)
        self.pltCanv.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.pltCanv.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.pltCanv.setFocus()
        self.pltCanv.updateGeometry()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.mouse_init()

        #=============================================
        # Main plot widget layout
        #=============================================
        self.layVMainMpl = QVBoxLayout()
        self.layVMainMpl.addWidget(self.pltCanv)
        self.setLayout(self.layVMainMpl)

        self.lines = []
        self.skeleton = []

        self.skeleton = Skeleton('dhm66_ISB_Xsens.urdf')

        self.timer = self.pltCanv.new_timer(
            100, [(self.update_canvas, (), {})])
        self.timer.start()

    def update_canvas(self):
        color = ['b', 'r']

        data_joint = []
        for i, port in enumerate(self.port):
            b_in = self.port[i].read()
            data = b_in.toString().split(' ')

            if len(data) == 67:
                del data[0]

            data = list(map(float, data))
            data = np.asarray(data)

            data_joint.append(np.deg2rad(data))
            
        self.ax.clear()
        self.lines = self.skeleton.visualise_from_joints(self.ax, data_joint, color_list=color, lines = [])
        self.pltCanv.figure.canvas.draw()