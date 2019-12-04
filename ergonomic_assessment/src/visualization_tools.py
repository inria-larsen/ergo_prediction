import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
# import cv2
from copy import deepcopy
import pandas as pd
import scipy.stats as st
from scipy.stats import kde


def draw_distribution(score, list_states, real_labels):
	
	clrs = np.zeros((len(list_states), 3))
	# sns.set(font_scale=1.5)
	id_pred = np.argmax(score)
	id_real = list_states.index(real_labels)

	labels = deepcopy(list_states)
	for i in range(len(labels)):
		labels[i] = labels[i].title()
		labels[i] = labels[i].replace("St", "Standing")
		labels[i] = labels[i].replace("Wa", "Walking")
		labels[i] = labels[i].replace("Kn", "Kneeling")
		labels[i] = labels[i].replace("Cr", "Crouching")

		labels[i] = labels[i].replace("Bf", "Bent forward")
		labels[i] = labels[i].replace("Bs", "Strongly bent")
		labels[i] = labels[i].replace("U", "Upright")
		labels[i] = labels[i].replace("Oh", "Hands above head")
		labels[i] = labels[i].replace("Os", "Elbows above shoulders")

		labels[i] = labels[i].replace("Id", "Idle")
		labels[i] = labels[i].replace("Re", "Reach")
		labels[i] = labels[i].replace("Rl", "Release")
		labels[i] = labels[i].replace("Fm", "Fine Manipulation")
		labels[i] = labels[i].replace("Sc", "Screw")
		labels[i] = labels[i].replace("Ca", "Carry")
		labels[i] = labels[i].replace("Pl", "Place")
		labels[i] = labels[i].replace("Pi", "Pick")	
		labels[i] = labels[i].replace("_", " ")

	for x in range(len(list_states)):
		if((id_pred == id_real) and (x == id_real)):
			
			clrs[x] = [0,1,0]
		# elif(x == id_real):
		# 	clrs[x] = [0,0,1]
		else:
			clrs[x] = [1,0,0]

		if((x == id_real)):
			labels[x] = labels[x].replace(" ", "\ ")
			labels[x] = '$\\bf{' + labels[x] + '}$'
			


	ax = sns.barplot(score, labels, palette=clrs)
	# ax.set_xlim(0, 1.0, 0.1)
	ax.set_xticks(np.arange(0, 1.05, 0.5))
	# ax.set_xticklabels(np.arange(0, 1.0), fontsize = 'x-large')
	plt.title('Action')

	ax.title.set_fontsize(50)
	plt.yticks(size = 40)
	plt.xticks(size = 30)
	# plt.ylabel('States')
	plt.xlabel('Probabilities', size=40)
	plt.subplots_adjust(left=0.7)
	
	return sns.barplot(score, labels, palette=clrs)

def video_distribution(score_samples, list_states, real_labels, fps, path, name_file):

	fig=plt.figure(figsize=(14,10))
	# plt.rcParams["figure.figsize"] = (5,10)
	
	ax = fig.add_subplot(1,1,1)
	n_frame = np.shape(real_labels)[0]

	ax = draw_distribution(score_samples[0], list_states, real_labels[0])

	def animate(i):
		plt.clf()
		ax = draw_distribution(score_samples[i], list_states, real_labels[i])

	anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n_frame, interval=fps)

	anim.save(path + name_file + '.mp4',writer=animation.FFMpegWriter(fps=8))
	# plt.show()
	return

def draw_pos(ax, pos_data):
	ax.set_xlim(-2,2)
	ax.set_ylim(-2,2)
	ax.set_zlim(-1,1)
	
	Xsens_bodies = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',  
			'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',  
			'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',   	
			'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
			'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe']

	Xsens_segments = [['Pelvis', 'L5'], ['L5', 'L3'], ['L3', 'T12'], ['T12', 'T8'], ['T8','Neck'], ['Neck', 'Head'], 
		['T8', 'Right Shoulder'], ['Right Shoulder', 'Right Upper Arm'], ['Right Upper Arm', 'Right Forearm'], ['Right Forearm', 'Right Hand'],
		['T8', 'Left Shoulder'],  ['Left Shoulder', 'Left Upper Arm'], ['Left Upper Arm', 'Left Forearm'], ['Left Forearm', 'Left Hand'],
		['Pelvis', 'Right Upper Leg'], ['Right Upper Leg', 'Right Lower Leg'], ['Right Lower Leg', 'Right Foot'], ['Right Foot', 'Right Toe'],
		['Pelvis', 'Left Upper Leg'], ['Left Upper Leg', 'Left Lower Leg'], ['Left Lower Leg', 'Left Foot'], ['Left Foot', 'Left Toe']]

	# Xsens_segments = [['L5', 'L3'], ['L3', 'T12'], ['T12', 'T8'], ['T8','Neck'],
	# 	['Neck', 'Head'], 
	# 	['T8', 'Right Shoulder'], ['Right Shoulder', 'Right Upper Arm'], ['Right Upper Arm', 'Right Forearm'], ['Right Forearm', 'Right Hand'],
	# 	['T8', 'Left Shoulder'],  ['Left Shoulder', 'Left Upper Arm'], ['Left Upper Arm', 'Left Forearm'], ['Left Forearm', 'Left Hand'],
	# 	['L5', 'Right Upper Leg'], ['Right Upper Leg', 'Right Lower Leg'], ['Right Lower Leg', 'Right Foot'], ['Right Foot', 'Right Toe'],
	# 	['L5', 'Left Upper Leg'], ['Left Upper Leg', 'Left Lower Leg'], ['Left Lower Leg', 'Left Foot'], ['Left Foot', 'Left Toe']]


	for seg in Xsens_segments:
		index_ini = Xsens_bodies.index(seg[0])
		x_ini = (pos_data[3*index_ini+0])
		y_ini = (pos_data[3*index_ini+1])
		z_ini = (pos_data[3*index_ini+2])

		index_fin = Xsens_bodies.index(seg[1])
		x_fin = (pos_data[3*index_fin+0])
		y_fin = (pos_data[3*index_fin+1])
		z_fin = (pos_data[3*index_fin+2])

		ax.plot([x_ini, x_fin],	[y_ini, y_fin], [z_ini, z_fin], 'm')


	return ax


def video_sequence(real_labels, predict_labels, video_input, video_output):
		cap = cv2.VideoCapture(video_input)

		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(video_output, fourcc, 24.0, (1280,720))
		# out = cv2.VideoWriter(path_destination + 'sequence_' + str(index) + '.avi',fourcc, 24.0, (800,600))

		count = 0
		flag = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			
			if(ret and flag < len(real_labels)):
				cv2.putText(frame,'Predicted Posture: ' + str(real_labels[flag])
					, (50,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
				cv2.putText(frame,'Predicted Action: ' + str(predict_labels[flag])
					, (50,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
				cv2.imshow('frame', frame)
				out.write(frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				count += 1
				if(count >= 60/24):
					flag += 1
					count = 0
			else:
				break

		cap.release()
		out.release()
		cv2.destroyAllWindows()
		return

def plot_ergo_score(ax, name_score, data, timestamps):
	ax.plot(timestamps, data, color='b')
	ax.set_title(name_score)
	ax.set_ylabel('score')
	# ax.set_xlabel('time (s)')

def plot_list_states(ax, labels, timestamps, list_states):
	color = cm.rainbow(np.linspace(0,1,len(list_states)))

	t_transition = [0]
	for t in range(1, len(labels)):
		if not(labels[t] == labels[t-1]):
			t_transition.append(t)
	t_transition.append(len(timestamps)-1)

	for t in range(1, len(t_transition)):
		id_states = list_states.index(labels[t_transition[t-1]])
		# ax.axvspan(timestamps[t_transition[t-1]], timestamps[t_transition[t]], ymin=0, ymax=1, alpha=0.5, color=color[id_states])

	legend = []
	for i in range(len(list_states)):
		legend.append(mpatches.Patch(color=color[i], label=list_states[i]))
	plt.legend(handles=legend)
	ax.set_xlabel('time (s)')
	ax.set_title('Current Action')
	# ax.yaxis.set_visible(False)
	plt.axis('off')



def plot_hist_score(list_states, labels, data):
	df_data = pd.DataFrame({'states': labels, 'score': data})
	df_group = df_data.groupby(['states'])
	ax = df_data.boxplot(grid=False, column='score', by='states')

def plot_ergo_latentspace(data, labels, color_label, color):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	df_data = pd.DataFrame({'data_1': data[0], 'data_2': data[1], 'labels': labels})
	df_group = df_data.groupby(['labels'])

	ax1.set_ylim(-1, 1)
	ax1.set_xlim(-1, 1)

	# for data_state in df_group:
	# # 	sns.jointplot(x=data_state[1]["data_1"], y=data_state[1]["data_2"], kind='kde', color="skyblue")
	# 	nbins = 20
	# 	values = np.vstack([data_state[1]["data_1"], data_state[1]["data_2"]])
	# 	k = kde.gaussian_kde(values)
	# 	xi, yi = np.mgrid[min(data_state[1]["data_1"]):max(data_state[1]["data_1"]):nbins*1j, min(data_state[1]["data_2"]):max(data_state[1]["data_2"]):nbins*1j]
	# 	zi = k(np.vstack([xi.flatten(), yi.flatten()]))

	# 	if data_state[1].iloc[0]['labels'] < 4:
	# 		color_map = plt.cm.Greens
	# 	elif data_state[1].iloc[0]['labels'] < 6:
	# 		color_map = plt.cm.Oranges
	# 	else:
	# 		color_map = plt.cm.Reds
	# 	# if data_state[1]['labels'] 

	# 	ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=color_map)
		# ax1.contour(xi, yi, zi.reshape(xi.shape) )

	plt.title('Ergonomic score in latent space')
	plt.savefig('map_postures.pdf')

	plt.figure()
	plt.scatter(data[0], data[1], c=color_label)
	legend = []
	for i in range(7):
		legend.append(mpatches.Patch(color=color[i], label=str(i+1)))

	plt.legend(handles=legend)

	plt.title('Ergonomic score in latent space')
	plt.savefig('map_postures.pdf')



def plot_state_latentspace(data, labels):
	fig = plt.figure(figsize=(8,8))
	ax = fig.gca()

	
	df_data = pd.DataFrame({'states': labels, 'dim_1': data[0], 'dim_2': data[1]})
	df_group = df_data.groupby(['states'])

	for data_state in df_group:
		x = data_state[1]['dim_1']
		y = data_state[1]['dim_2']

		if len(x) < 2:
			continue

		mean = np.array([0, 0])
		sigma_1 = 2
		sigma_2 = 1
		alpha = np.pi / 4


	# Define the borders
	# deltaX = (max(data[0]) - min(data[0]))/10
	# deltaY = (max(data[1]) - min(data[1]))/10

		# deltaX = 0.01
		# deltaY = 0.01

		# xmin = min(x) - deltaX
		# xmax = max(x) + deltaX

		# ymin = min(y) - deltaY
		# ymax = max(y) + deltaY

		# xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

		# positions = np.vstack([xx.ravel(), yy.ravel()])
		# values = np.vstack([x, y])
		# kernel = st.gaussian_kde(values)
		# f = np.reshape(kernel(positions).T, xx.shape)

		

		# ax.set_xlim(xmin, xmax)
		# ax.set_ylim(ymin, ymax)
		# cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
		# ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
		# cset = ax.contour(xx, yy, f, colors='k')
		# ax.clabel(cset, inline=1, fontsize=10)
		# ax.set_xlabel('X')
		# ax.set_ylabel('Y')
		# plt.title('2D Gaussian Kernel density estimation')






