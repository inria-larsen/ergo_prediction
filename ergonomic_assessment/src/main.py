from ErgoScore import ErgoScore
from xsens_parser import mvnx_tree

import matplotlib.pyplot as plt
import numpy as np

import visualization_tools as vtools
from copy import deepcopy

if __name__ == '__main__':
	path = '/home/amalaise/Documents/These/code/AE_ProMPS/data/7x10_actions/XSens/xml/'
	mvnx_tree = mvnx_tree(path + 'standing-001.mvnx')

	position_data = mvnx_tree.get_data('position')
	joint_data = mvnx_tree.get_data('jointAngle')
	orientation = mvnx_tree.get_data('orientation')

	o_npose, p_npose = mvnx_tree.get_npose()

	# for t in range(len(position_data)):
		# abs_data = deepcopy(position_data[t])
		# o_data = orientation[t]

		# i = 0
		# q0 = o_data[i*4]
		# q1 = o_data[i*4 + 1]
		# q2 = o_data[i*4 + 2]
		# q3 = o_data[i*4 + 3]

		# R = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
		# 	[2*q1*q2 + 2*q0*q3, q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*q2*q3 - 2*q0*q1],
		# 	[2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0*q0 - q1*q1 - q2*q2 + q3*q3]])


		# for i in range(0, 23):
		# 	# position_data[t, i*3:i*3+3] = abs_data[i*3:i*3+3] - abs_data[0:3]
		# 	# print(np.shape(position_data[t,i*3:i*3+3]))
		# 	position_data[t, i*3:i*3+3] = position_data[t, i*3:i*3+3] - abs_data[0:3]
		# 	position_data[t, i*3:i*3+3] = R@position_data[t,i*3:i*3+3]


	ergo_score = ErgoScore()

	trunk_bend = []

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	count = 0

	for i in range(len(position_data)):
		ergo_score.update_posture(position_data[i])
		trunk_bend.append(ergo_score.compute_score_neck())

	


	pose = position_data[0]

	vtools.draw_pos(ax, pose)

	plt.figure()
	plt.plot(trunk_bend)

	# id_joint = mvnx_tree.get_id_joint()
	# plt.plot(joint_data[:,0:3])
	plt.show()




