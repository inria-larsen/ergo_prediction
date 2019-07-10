import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd
import seaborn as sns
import random 
import pickle
from sklearn.model_selection import train_test_split
from matplotlib import animation

from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture

list_joints = ['jL5S1',
			'jL4L3',
			'jL1T12',
			'jT9T8',
			'jT1C7',
			'jC1Head',
			'jRightT4Shoulder',
			'jRightShoulder',
			'jRightElbow',
			'jRightWrist',
			'jLeftT4Shoulder',
			'jLeftShoulder',
			'jLeftElbow',
			'jLeftWrist',
			'jRightHip',
			'jRightKnee',
			'jRightAnkle',
			'jRightBallFoot',
			'jLeftHip',
			'jLeftKnee',
			'jLeftAnkle',
			'jLeftBallFoot'
			]

list_segments = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',  
			'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',  
			'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',   	
			'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe',
			'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToe']


def sync_labels(timestamps, labels):
	labels_sync = []

	# print(np.shape(timestamps), np.shape(labels))

	T = round(len(timestamps)/len(labels))

	for i, label in enumerate(labels):
		for t in range(30):
			labels_sync.append(label)

	for i in range(len(labels_sync), len(timestamps)):
		labels_sync.append(label)

	return labels_sync

def load_data(path, tracks, name_feature):
	print('Loading data...')

	all_tracks = ['current_action']

	timestamps = []

	real_labels = []
	list_states = []

	df_all_data = []

	data_win2 = []

	with open(path + 'save_data_dump.pkl', 'rb') as input:
		data_win2 = pickle.load(input)
	with open(path + 'save_labels_dump.pkl', 'rb') as input:
		real_labels = pickle.load(input)
	with open(path + 'save_liststates_dump.pkl', 'rb') as input:
		list_states = pickle.load(input)
	with open(path + 'save_listfeatures_dump.pkl', 'rb') as input:
		list_features = pickle.load(input)

	list_reduce_features = []

	if(name_feature == 'jointAngle_'):
		for joint in list_joints:
			list_reduce_features.append(name_feature + joint.lower() + '_' + 'x')
			list_reduce_features.append(name_feature + joint.lower() + '_' + 'y')
			list_reduce_features.append(name_feature + joint.lower() + '_' + 'z')
	else:
		orientation_feature = []
		for segment in list_segments:
			list_reduce_features.append(name_feature + segment.lower() + '_' + 'x')
			list_reduce_features.append(name_feature + segment.lower() + '_' + 'y')
			list_reduce_features.append(name_feature + segment.lower() + '_' + 'z')

			orientation_feature.append('orientation_' + segment.lower() + '_' + 'q0')
			orientation_feature.append('orientation_' + segment.lower() + '_' + 'q1')
			orientation_feature.append('orientation_' + segment.lower() + '_' + 'q2')
			orientation_feature.append('orientation_' + segment.lower() + '_' + 'q3')



	# id_track_rm = 0
	# for num_track in range(len(all_tracks)):
	# 	if(not(all_tracks[num_track] in tracks)):
	# 		del real_labels[num_track - id_track_rm]
	# 		del list_states[num_track - id_track_rm]
	# 		id_track_rm += 1



	for data, num_data in zip(data_win2, range(len(data_win2))):
		df_all_data = pd.DataFrame(data, columns = list_features)
		df_reduce_data = df_all_data[list_reduce_features]
		data_win2[num_data] = df_reduce_data[list_reduce_features].values

		if name_feature == 'jointAngle_':
			data_win2[num_data] = np.deg2rad(data_win2[num_data])

		# if name_feature == 'position_':
		# 	orientation = df_all_data[orientation_feature].values

		# 	for t in range(len(data_win2[num_data])):
		# 		abs_data = np.copy(data[t])
		# 		o_data = orientation[t]

		# 		i = 0
		# 		q0 = o_data[i*4]
		# 		q1 = o_data[i*4 + 1]
		# 		q2 = o_data[i*4 + 2]
		# 		q3 = o_data[i*4 + 3]

		# 		R = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
		# 			[2*q1*q2 + 2*q0*q3, q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*q2*q3 - 2*q0*q1],
		# 			[2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0*q0 - q1*q1 - q2*q2 + q3*q3]])

		# 		for i in range(0, 23):
		# 			# data_win2[num_data][t, i*3:i*3+3] = abs_data[i*3:i*3+3] - abs_data[0:3]
		# 			data_win2[num_data][t, i*3:i*3+3] = R@data[t,i*3:i*3+3]

		timestamps.append(data[:,0])
		real_labels[0][num_data] = sync_labels(timestamps[num_data], real_labels[0][num_data])

	list_features = list_reduce_features

	dim_features = np.ones(len(list_features))

	print('Data Loaded')

	# return list_features, [data_win2[0]], [real_labels[0]], [timestamps[0]], list_states
	return list_features, data_win2, real_labels, timestamps, list_states

def simple_posture(mapping):
	posture = []

	for i in range(len(mapping)):
		for joint in mapping:
			print(joint)

	return posture

def mean_and_cov(all_data, labels, n_states, list_features):
	""" Compute the means and covariance matrix for each state 

	Return a vector of scalar and a vector of matrix corresponding to the mean 
	of the distribution and the covariance
	"""
	n_feature = len(list_features)
	df_data = pd.DataFrame(all_data, columns = list_features)

	df_labels = pd.DataFrame(labels)
	df_labels.columns = ['state']

	df_total = pd.concat([df_data, df_labels], axis=1)

	data = []
	for state, df in df_total.groupby('state'):
		data.append(df[list_features].values)

	sigma = np.zeros(((n_states,n_feature,n_feature)))
	mu = np.zeros((n_states, np.sum(n_feature)))

	for i in range(n_states):
		mu[i] = np.mean(data[i], axis=0)
		sigma[i] = np.cov(data[i].T)

	return mu, sigma


def split_data_base(data_set, labels, ratio):
	"""
	This function allows to split a database into three subset:
	- Reference subset
	- Validation subset
	- Test subset
	The number of elements in each subset correspond to the ratio in input such as ratio is a list of float or int such as sum(ratio) = 100
	"""

	nbr_sequences = len(data_set)

	base_ref = []
	labels_ref = []
	base_test = []
	labels_test = []
	base_val = []
	labels_val = []

	if(ratio[2] > 0):

		id_train, id_subset = train_test_split(np.arange(nbr_sequences), train_size=ratio[0]/100)
		id_test, id_val = train_test_split(id_subset, train_size=(ratio[2]*100/(100-ratio[0]))/100)

		for i in id_train:
			base_ref.append(data_set[i])
			labels_ref.append(labels[i])

		for i in id_test:
			base_test.append(data_set[i])
			labels_test.append(labels[i])

		for i in id_val:
			base_val.append(data_set[i])
			labels_val.append(labels[i])
		
		return base_ref, labels_ref, base_test, labels_test, base_val, labels_val, id_train, id_test, id_val

	else:
		id_train, id_test = train_test_split(np.arange(nbr_sequences), train_size=ratio[0]/100)

		for i in id_train:
			base_ref.append(data_set[i])
			labels_ref.append(labels[i])

		for i in id_test:
			base_test.append(data_set[i])
			labels_test.append(labels[i])
		
		return base_ref, labels_ref, base_test, labels_test, id_train, id_test

def quat2rot(q):
	R = np.matrix([	[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
			[2*q[0]*q[3] + 2*q[1]*q[2], q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
			[2*q[1]*q[3] - 2*q[0]*q[2], 2*q[0]*q[1] + 2*q[2]*q[3], q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])
	return R

def compute_rotation(theta, length):
	Rx = quat2rot(np.array([np.cos(theta[0]/2), np.sin(theta[0])/2, 0., 0.]))
	Ry = quat2rot(np.array([np.cos(theta[1]/2), 0., np.sin(theta[1])/2, 0.]))
	Rz = quat2rot(np.array([np.cos(theta[2]/2), 0., 0., np.sin(theta[2])/2]))
	R = Rx * Ry * Rz
	Hjoint = np.matrix([[R[0,0], R[0,1], R[0,2], length[0]], [R[1,0], R[1,1], R[1,2], length[1]], [R[2,0], R[2,1], R[2,2], length[2]], [0, 0, 0, 1]])

	return Hjoint

def animate_skeleton():
	fig = plt.figure()
	ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
	line, = ax.plot([], [], lw=2)

	# initialization function: plot the background of each frame
	def init():
		line.set_data([], [])
		return line,

	# animation function.  This is called sequentially
	def animate(i):
		print(i)
		x = np.linspace(0, 2, 1000)
		y = np.sin(2 * np.pi * (x - 0.01 * i))
		line.set_data(x, y)
		return line,

	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate, init_func=init,
								   frames=100, interval=20, blit=True)

	# save the animation as an mp4.  This requires ffmpeg or mencoder to be
	# installed.  The extra_args ensure that the x264 codec is used, so that
	# the video can be embedded in html5.  You may need to adjust this for
	# your system: for more information, see
	# http://matplotlib.sourceforge.net/api/animation_api.html
	# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

	# plt.show()
	return

def plot_loss_function(metric):
	return

def compute_sequence_ergo(data, num_frame, list_score):
	score_total = []
	human_posture = HumanPosture('config/mapping_joints.json')
	ergo_assessment = ErgoAssessment('config/rula_config.json')

	human_posture.update_posture(data)
	ergo_assessment.compute_ergo_scores(human_posture)
	for score in list_score:
		score_total.append(ergo_assessment[score])
	return score_total

def standardization(data):
	data = np.asarray(data)
	data_norm = np.copy(data)
	data_norm = data_norm.astype('float64')

	mean_data = np.mean(data, axis = 0)
	var_data = np.std(data, axis = 0)
	x_norm = np.copy(data)

	size_data, input_dim = np.shape(data)
	for i in range(input_dim):
		if var_data[i] == 0:
			var_data[i] = 0.00001
			
		data_norm[:,i] = (data[:,i] - mean_data[i])/var_data[i]
		# x_norm[:,i] = (x_joint[:,i] - mean_data[i])
	data_norm = data_norm.astype(np.float32)

	return data_norm, mean_data, var_data

def destandardization(data, mean_data, var_data):
	size_data, input_dim = np.shape(data)
	for i in range(input_dim):
		data[:,i] = np.asarray(data[:,i]*var_data[i] + mean_data[i])	

	return data

def normalization(data):
	data_norm = np.copy(data)
	data_norm = data_norm.astype('float64')
	min_data = np.min(data, axis = 0)
	max_data = np.max(data, axis = 0)

	size_data, input_dim = np.shape(data)
	for i in range(input_dim):
		data_norm[:,i] = (data[:,i] - min_data[i])/(max_data[i] - min_data[i])

	return data_norm, min_data, max_data

def denormalization(data, min_data, max_data):
	size_data, input_dim = np.shape(data)
	for i in range(input_dim):
		data[:,i] = data[:,i]*(max_data[i] - min_data[i])+min_data[i]

	return data





