import numpy as np
import tools
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
import os
import visualization_tools as vt

import tensorflow as tf
import pdb
import Data_driver
import matplotlib.pyplot as plt
import input_data
import matplotlib.cm as cm
import time
from matplotlib import animation
import AE
from dataTreatment import DataTreatment

list_segments = ['Pelvis',
			'L5',
			'L3',
			'T12',
			'T8',
			'Neck',
			'Head',
			'RightShoulder',
			'RightUpperArm',
			'RightForeArm',
			'RightHand',
			'LeftShoulder',
			'LeftUpperArm',
			'LeftForeArm',
			'LeftHand',
			'RightUpperLeg',
			'RightLowerLeg',
			'RightFoot',
			'RightToe',
			'LeftUpperLeg',
			'LeftLowerLeg',
			'LeftFoot',
			'LeftToe']

if __name__ == '__main__':
	# arguments

	local_path = os.path.abspath(os.path.dirname(__file__))

	path = os.path.join(local_path, '../../../../experiments/ANDY_DATASET/AndyData-lab-onePerson/')
	path_data_dump = os.path.join(local_path, '../../../score/')

	tracks = ['detailed_posture']

	all_tracks = ['general_posture', 'detailed_posture', 'details', 'current_action']


	path_data_root = path + '/xsens/allFeatures_csv/'

	list_participant = os.listdir(path_data_root)
	list_participant.sort()

	participant_label = []
	num_sequence = []
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 3
	nbr_subsets_iter = 10

	print('Loading data...')

	timestamps = []

	real_labels = []
	list_states = []

	df_all_data = []

	data_win, real_labels, list_states, list_features = tools.load_data_from_dump(path_data_dump)

	name_feature = 'position_'
	list_reduce_features = []

	for segment in list_segments:
		list_reduce_features.append(name_feature + segment + '_' + 'x')
		list_reduce_features.append(name_feature + segment + '_' + 'y')
		list_reduce_features.append(name_feature + segment + '_' + 'z')

	del list_reduce_features[0:3]

	id_track_rm = 0
	for num_track in range(len(all_tracks)):
		if(not(all_tracks[num_track] in tracks)):
			del real_labels[num_track - id_track_rm]
			del list_states[num_track - id_track_rm]
			id_track_rm += 1


	real_labels = real_labels[0]
	list_states = list_states[0]
	print(list_states)

	df_all_data = []

	for data, num_data in zip(data_win, range(len(data_win))):
		df_all_data.append(pd.DataFrame(data, columns = list_features))
		df_all_data[-1] = df_all_data[-1][list_reduce_features]
		data_win[num_data] = df_all_data[-1][list_reduce_features].values

	list_features = list_reduce_features

	dim_features = np.ones(len(list_features))

	print('Data Loaded')

	flag = 0
	score = []
	best_features = []

	data_states = [[]]

	# for state, num_state in zip(list_states, range(len(list_states))):
	# 	for i in range(len(data_win)):
	# 		for j in range((len(data_win[i]))):
	# 			if(real_labels[i][j] == state):
	# 				data_states[num_state].append(data_win[i][j])

	# 	if(num_state < len(list_states)-1):
	# 		data_states.append([])


	data_states = np.asarray(data_states)

	all_data = data_win[0]
	all_labels = real_labels[0]

	for i in range(1, len(data_win)):
		all_data = np.concatenate([all_data, data_win[i]])
		all_labels = np.concatenate([all_labels, real_labels[i]])


	my_data = input_data.read_data_sets('MNIST_data', one_hot=True) 
	# val_tmp2 = int(data_driver.data.shape[1] - data_driver.data.shape[1]/10 )
	# tall_tmp = data_driver.data.shape[1]/10

	base_ref, labels_ref, base_test, labels_test, base_val, labels_val, id_train, id_test, id_val = tools.split_data_base(all_data, all_labels, [60, 20, 20])



	#training Data
	my_data.train._images = None
	my_data.train._num_examples = len(base_ref)
	my_data.train._labels = np.zeros([my_data.train._num_examples,len(list_states)], np.float32)
	my_data.train._images = np.zeros([my_data.train._num_examples, len(list_features)],np.float32)
	my_data.train._index_in_epoch= 0
	my_data.train._epoch_completed= 0

	# choices = {'bent_fw': 0, 'bent_fw_strongly': 1,'kicking': 2,'lifting_box': 3,'standing': 4,'walking': 5,'window_open': 6}
	#choices = {'Setup_A_Seq_1': 0, 'Setup_A_Seq_2': 1,'Setup_A_Seq_3': 2,'Setup_A_Seq_4': 3,'Setup_A_Seq_5': 4,'Setup_A_Seq_6': 5}

	# for i in range(len(list_states)): #chaque type 
	for j in range(len(base_ref)): 
		id_label = list_states.index(labels_ref[j])
		my_data.train._labels[j][id_label] = 1
		my_data.train._images[j] = base_ref[j]

	#Validation Data
	my_data.validation._images = None
	my_data.validation._num_examples = len(base_val)
	my_data.validation._labels = np.zeros([my_data.validation._num_examples,len(list_states)], np.float32)
	my_data.validation._images = np.zeros([my_data.validation._num_examples, len(list_features)],np.float32)
	my_data.validation._index_in_epoch= 0
	my_data.validation._epoch_completed= 0
	for j in range(len(base_val)): 
		id_label = list_states.index(labels_val[j])
		my_data.validation._labels[j][id_label] = 1
		my_data.validation._images[j] = base_val[j]



	#Test Data
	my_data.test._images = None
	my_data.test._num_examples = len(base_test)
	my_data.test._labels = np.zeros([my_data.test._num_examples,len(list_states)], np.float32)
	my_data.test._images = np.zeros([my_data.test._num_examples, len(list_features)],np.float32)
	my_data.test._index_in_epoch= 0
	my_data.test._epoch_completed= 0
	for j in range(len(base_test)):
		id_label = list_states.index(all_labels[j])
		my_data.test._labels[j][id_label] = 1
		my_data.test._images[j] = base_test[j]


	mnist = my_data

	######################################################################

	flag_restore = True
	flag_plot_squeleton = True
	batch_size =1000
	training_epochs = 5000
	learning_rate = 0.001
	display_step = 1000
	examples_to_show = 10
	typeActivation = 'leaky' #sigmoid apprend mal
	typeInit = 'xavier_init' #random apprend mal
	typeOpti='losses.mean_squared_error'
	x_len = 22
	y_len = 3
	# Network Parameters
	num_hidden_1 = 500#256 # 1st layer num features
	n_z = 2
	num_input = len(list_features)
	# num_input = x_len*y_len # MNIST data input (img shape: 28*28)

	ae = AE.AE(num_input,num_hidden_1,n_z,learning_rate,typeInit = typeInit, typeActivation=typeActivation,typeOpti=typeOpti,restore=flag_restore, epoch=training_epochs)
	dt = DataTreatment

	if(flag_restore == False):
		for i in range(1, training_epochs+1):
			# Prepare Data
			# Get the next batch of MNIST data (only images are needed, not labels)
			batch_x, _ = mnist.train.next_batch(batch_size)

			# Run optimization op (backprop) and cost op (to get loss value)
			l = ae.partial_fit(batch_x, i)

			# Display logs per step
			if i % display_step == 0 or i == 1:
				print('Step %i: Minibatch Loss: %f' % (i, l))
		ae.save_session(training_epochs,n_z)


	n = examples_to_show*examples_to_show
	canvas_orig = np.empty((x_len * n, y_len * n))
	canvas_recon = np.empty((x_len * n, y_len * n))
	#for i in range(n):
	i=1
	# MNIST test set
	batch_x, _ = mnist.test.next_batch(n)
	# Encode and decode the digit image
	g = ae.sess.run(ae.decoder_op, feed_dict={ae.X: batch_x})

	# Display original images
	for j in range(n):
		# Draw the original digits
		canvas_orig[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len] = \
			batch_x[j].reshape([x_len, y_len])
	# Display reconstructed images
	for j in range(n):
		# Draw the reconstructed digits
		canvas_recon[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len] = \
			g[j].reshape([x_len, y_len])


	print(np.shape(g))


	#ploting squeleton
	if (flag_plot_squeleton==True):
		g2 = g
		DATA_PARAMS = {}
		DATA_PARAMS.update({"data_source": "MVNX", 'as_3D': True, 'data_types': ['position'],"unit_bounds": True})
		data_driver = Data_driver.Data_driver(DATA_PARAMS)
		data_driver.as_3D = True
		DATA_VISUALIZATION = {}
		DATA_VISUALIZATION.update({
			'transform_with_all_vtsfe': False,
			'dynamic_plot': False,
			'body_lines': True,
			'data_inf': [], 
			'show': True, 
			'nb_samples_per_mov': 1,
			'average_reconstruction': False,
			"data_driver" : data_driver,
			"reconstr_datasets" : g2, #(1, 1, 70, 70, 66)
			"reconstr_datasets_names" : ['testWithAE'], #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
			"x_samples" : batch_x, #(70,70,66)
			"sample_indices" : [0], #8
			"only_hard_joints" : False,
		   # "mov_types" : ['bent_fw'],
		   "displayed_movs" :['random posture'],
			#"data_inf" : data_inf,
			"plot_3D" : True, #add ori
			"time_step" : 1,
			'n_z': n_z,
			#"as_3D" : Trueki
			})
		dt.show_data(**DATA_VISUALIZATION)

