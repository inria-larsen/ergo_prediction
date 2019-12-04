from Skeleton import Skeleton

import matplotlib.pyplot as plt

import numpy as np
import json
import pickle
import tools

import os
import argparse
import configparser

import visualization_tools as vtools



if __name__ == '__main__':
	# Get arguments
	parser=argparse.ArgumentParser()
	parser.add_argument('--file', '-f', help='Configuration file', default="config_AE.ini")
	parser.add_argument('--config', '-c', help='Configuration type', default="DEFAULT")
	args=parser.parse_args()
	config_file = args.file
	config_type = args.config

	local_path = os.path.abspath(os.path.dirname(__file__))

	# Parameters configuration
	config = configparser.ConfigParser()
	config.read('config/' + config_file)

	path = config[config_type]["path_data"]

	tracks = ['details']

	list_features, data_np, real_labels, timestamps, list_states = tools.load_data(path, tracks, 'jointAngle_')
	list_states = list_states[0]

	timestamp_all = []
	time_init = 0
	labels_all = []

	# Compute ergo score
	posture = Skeleton('dhm66_ISB_Xsens.urdf')
	# posture.update_posture(data_np[0][5000])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# posture.visualise_from_joints()

	posture.animate_skeleton([data_np[0][0::50]])
	# tools.animate_skeleton()

	plt.show()
