import AE
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

import os

from Skeleton import Skeleton
from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture
import tools

if __name__ == '__main__':
	metric = 'jointAngle'

	size_latent = 2
	loss = [[]]
	autoencoder = []

	all_score = []
	all_size = []
	type_data = []

	path = "save/AE/" + metric + "/" + str(size_latent) + '/'
	list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	list_files.sort()
	file = list_files[0]
	print(file)
	with open(path + file, 'rb') as input:
		autoencoder = pickle.load(input)

	input_data = autoencoder.get_data_test()
	data_output, encoded_data, score = autoencoder.test_model(input_data)

	reduce_posture = HumanPosture('config/mapping_joints.json')

	data_input_whole = np.zeros((len(input_data), 66))
	data_output_whole = np.zeros((len(input_data), 66))

	if metric == 'posture':
		for i, data in enumerate(data_output):
			data_input_whole[i] = reduce_posture.reduce2complete(input_data[i])
			data_output_whole[i] = reduce_posture.reduce2complete(data)

	else:
		data_input_whole = input_data
		data_output_whole = data_output

	posture = Skeleton('dhm66_ISB_Xsens.urdf')

	# plt.figure()
	# plt.plot(encoded_data)

	fig = plt.figure()
	 # plt.scatter(encoded_data[0,0], encoded_data[0,1])
	plt.xlim(xmin = 0.0, xmax=1.0)
	plt.ylim(ymin = 0.0, ymax=1.0)

	# for i, data in enumerate(encoded_data):
	plt.scatter(encoded_data[:,0], encoded_data[:,1], color = 'b')

	# def animate(i):
	# 	for num_seq, sequence in enumerate(joint_seq_list):
	# 			lines[num_seq][num_seg].set_data([self.position[id_ini*3],self.position[id_end*3]], 
	# 				[self.position[id_ini*3+1],self.position[id_end*3+1]])
	# 			lines[num_seq][num_seg].set_3d_properties([self.position[id_ini*3+2],self.position[id_end*3+2]])

	# anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=len(joint_seq_list[0]), interval=20)


#	fig = plt.figure()
#	ax = Axes3D(fig)

#	posture.animate_skeleton([data_input_whole, data_output_whole])

	plt.show()
