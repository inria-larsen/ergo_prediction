import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import yarp

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import andy_ergo

class MyApp(QtWidgets.QMainWindow, andy_ergo.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.__timer = QTimer()
        self.__timer.start(10)# every 10 ms

if __name__ == '__main__':
    yarp.Network.init()
    rf = yarp.ResourceFinder()
    rf.configure(sys.argv)

    app = QtWidgets.QApplication(sys.argv)
    form = MyApp()
    form.show()
    app.exec_()