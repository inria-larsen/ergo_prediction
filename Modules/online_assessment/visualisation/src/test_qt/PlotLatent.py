import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pandas as pd

import yarp
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../ergonomic_assessment/src/')))

import AE
import tools
from Skeleton import Skeleton
from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture

class  LatentSpaceWidget(pg.GraphicsWindow):
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    ptr1 = 0

    def __init__(self, parent=None, config_ergo = 'rula', **kargs):
        pg.GraphicsWindow.__init__(self, **kargs)
        self.setParent(parent)
        timer = pg.QtCore.QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(10) # number of seconds (every 1000) for next update

        self.view = pg.PlotWidget(self)
        self.view.resize(350, 350)
        self.view.setWindowTitle('Ergonomic score in latent space')
        self.view.setAspectLocked(True)
        self.view.show()

        self.port = yarp.BufferedPortBottle()
        self.port.open('/plot_latentspace/' + config_ergo)
        yarp.Network.connect('/AE/latent:o', self.port.getName())

        metric = 'jointAngle'

        path_src = "/home/amalaise/Documents/These/code/ergo_prediction/ergonomic_assessment/src/"

        self.config_ergo = config_ergo

        if self.config_ergo == 'rula':
            ergo_name = ['RULA_SCORE']
            ergo_assessment = ErgoAssessment(path_src + 'config/rula_config.json')
        else:
            ergo_name = ['TABLE_REBA_C']
            ergo_assessment = ErgoAssessment(path_src + 'config/reba_config.json')

        size_latent = 2
        dx = 0.02

        loss = [[]]
        autoencoder = []

        all_score = []
        all_size = []
        type_data = []
        
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
        
        list_ergo_score = ergo_assessment.get_list_score_name()
        list_ergo_score.sort()

        reduce_posture = HumanPosture(path_src + 'config/mapping_joints.json')
        posture = Skeleton('dhm66_ISB_Xsens.urdf')

        self.X = np.arange(0.0, 1.0+dx, dx)



        pd_ergo_grid = pd.read_csv(self.config_ergo + '_grid.csv', index_col=False)
        self.ergo_grid = np.asarray(pd_ergo_grid)

        ergo_color_grid = np.zeros((len(self.ergo_grid), len(self.ergo_grid), 3))

        print(np.shape(ergo_color_grid))

        for i in range(len(self.ergo_grid)):
            for j in range(len(self.ergo_grid)):
                if config_ergo == 'reba':
                    if self.ergo_grid[i,j] == 1:
                        ergo_color_grid[i,j] = [0,1,0]
                    elif self.ergo_grid[i,j] == 2:
                        ergo_color_grid[i,j] = [0.3,0.7,0]
                    elif self.ergo_grid[i,j] == 3:
                        ergo_color_grid[i,j] = [0.5,0.5,0]
                    elif self.ergo_grid[i,j] == 4:
                        ergo_color_grid[i,j] = [0.7,0.3,0]
                else:
                    if self.ergo_grid[i,j] == 2:
                        ergo_color_grid[i,j] = [0,1,0]
                    elif self.ergo_grid[i,j] == 3:
                        ergo_color_grid[i,j] = [0.3,0.7,0]
                    elif self.ergo_grid[i,j] == 4:
                        ergo_color_grid[i,j] = [0.5,0.5,0]
                    elif self.ergo_grid[i,j] == 5:
                        ergo_color_grid[i,j] = [0.7,0.3,0]

        self.ergo_grid = ergo_color_grid

        # self.ergo_grid = np.zeros((len(self.X), len(self.X)))

        # for i, data_x in enumerate(self.X):
        #     for j, data_y in enumerate(self.X):

        #         x = np.zeros((1,size_latent))
        #         x[0, 0] = data_x
        #         x[0, 1] = data_y

        #         decoded_data = autoencoder.decode_data(x)
        #         if metric == 'posture':
        #             whole_body = reduce_posture.reduce2complete(decoded_data[0])
        #             posture.update_posture(whole_body)
        #         else:
        #             posture.update_posture(decoded_data[0])

        #         ergo_score = tools.compute_sequence_ergo(ergo_assessment, decoded_data[0], 0, ergo_name, path_src)[0]

        #         if self.config_ergo == 'reba':
        #             if ergo_score == 1:
        #                 ergo_score = 1
        #             elif 1 < ergo_score < 5:
        #                 ergo_score = 2
        #             elif 4 < ergo_score < 6:
        #                 ergo_score = 3
        #             else:
        #                 ergo_score = 4

        #         self.ergo_grid[j,i] = ergo_score

        self.flag = 0

        self.plot_latent_space()


    def plot_latent_space(self, x=0, y=0):
        if self.flag:
            self.view.removeItem(self.scatter)
            self.view.removeItem(self.plot_traj)
        else:
            self.list_traj = [[], []]

        self.scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=10, color='b'), symbol='s', size=3)
      
        img_np = np.rot90(np.rot90(np.rot90(self.ergo_grid)))
        
        img = pg.ImageItem(img_np)

        self.scatter.setData(x=[x], y=[y])

        self.list_traj[0].append(x)
        self.list_traj[1].append(y)

        if len(self.list_traj[0]) > 10:
            del self.list_traj[0][0]
            del self.list_traj[1][0]

        self.plot_traj = pg.PlotCurveItem(x=self.list_traj[0],y=self.list_traj[1], pen=pg.mkPen(width=2, color='b'))

        img.setZValue(-100)
        self.view.addItem(img)
        self.view.addItem(self.scatter)
        self.view.addItem(self.plot_traj)
        self.view.getPlotItem().hideAxis('bottom')
        self.view.getPlotItem().hideAxis('left')

        if self.flag == 0:
            del self.list_traj[0][0]
            del self.list_traj[1][0]
            self.flag = 1

    
    def update(self):
        b_in = self.port.read()
        data = b_in.toString().split(' ')

        del data[0]

        data = list(map(float, data))
        data = np.asarray(data)

        self.plot_latent_space(x=data[0]*len(self.X), y=len(self.X)-data[1]*len(self.X))