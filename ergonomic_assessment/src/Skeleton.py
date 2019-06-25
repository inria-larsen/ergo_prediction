import numpy as np
import json
import torch
import math
import tools
from urdf_parser_py.urdf import URDF
from matplotlib import animation
import matplotlib.pyplot as plt


# Define segments (between origins of two consecutive links)
Xsens_segments = [['Pelvis', 'L5'], ['L5', 'L3'], ['L3', 'T12'], ['T12', 'T8'], ['T8','Neck'], ['Neck', 'Head'], 
		['T8', 'Right Shoulder'], ['Right Shoulder', 'Right Upper Arm'], ['Right Upper Arm', 'Right Forearm'], ['Right Forearm', 'Right Hand'],
		['T8', 'Left Shoulder'],  ['Left Shoulder', 'Left Upper Arm'], ['Left Upper Arm', 'Left Forearm'], ['Left Forearm', 'Left Hand'],
		['Pelvis', 'Right Upper Leg'], ['Right Upper Leg', 'Right Lower Leg'], ['Right Lower Leg', 'Right Foot'], ['Right Foot', 'Right Toe'],
		['Pelvis', 'Left Upper Leg'], ['Left Upper Leg', 'Left Lower Leg'], ['Left Lower Leg', 'Left Foot'], ['Left Foot', 'Left Toe']]

Xsens_bodies = [ 	'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',  
			'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',  
			'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',   	
			'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
			'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'
			]

# Xsens to URDF joint mapping
joint_mapping = {
			'jL5S1_X' : [0, 1],
			'jL5S1_Y' : [1, 1],
			'jL5S1_Z' : [2, -1],
			'jL4L3_X' : [3, 1],
			'jL4L3_Y' : [4, 1],
			'jL4L3_Z' : [5, -1],
			'jL1T12_X' : [6, 1],
			'jL1T12_Y' : [7, 1],
			'jL1T12_Z' : [8, -1],
			'jT9T8_X' : [9, 1],
			'jT9T8_Y' : [10,1],
			'jT9T8_Z' : [11, -1],
			'jT1C7_X' : [12, 1],
			'jT1C7_Y' : [13, 1],
			'jT1C7_Z' : [14, -1],
			'jC1Head_X' : [15, 1],
			'jC1Head_Y' : [16, 1],
			'jC1Head_Z' : [17, -1],

			'jRightC7Shoulder_X' : [18, -1],
			'jRightC7Shoulder_Y' : [19, 1],
			'jRightC7Shoulder_Z' : [20, 1],
			'jRightShoulder_X' : [21, -1],
			'jRightShoulder_Y' : [22, 1],
			'jRightShoulder_Z' : [23, 1],
			'jRightElbow_X' : [24, -1],
			'jRightElbow_Y' : [25, 1],
			'jRightElbow_Z' : [26, 1],
			'jRightWrist_X' : [27, -1],  
			'jRightWrist_Y' : [28, 1],
			'jRightWrist_Z' : [29, 1],

			'jLeftC7Shoulder_X' : [30, 1],
			'jLeftC7Shoulder_Y' : [31, -1],
			'jLeftC7Shoulder_Z' : [32, 1],
			'jLeftShoulder_X' : [33, 1],
			'jLeftShoulder_Y' : [34, -1],
			'jLeftShoulder_Z' : [35, 1],
			'jLeftElbow_X' : [36, 1],
			'jLeftElbow_Y' : [37, -1],
			'jLeftElbow_Z' : [38, 1],
			'jLeftWrist_X' : [39, 1],
			'jLeftWrist_Y' : [40, -1],
			'jLeftWrist_Z' : [41, 1],

			'jRightHip_X' : [42, -1],
			'jRightHip_Y' : [43, 1],
			'jRightHip_Z' : [44, 1],
			'jRightKnee_X' : [45, -1],
			'jRightKnee_Y' : [46, 1],
			'jRightKnee_Z' : [47, -1],
			'jRightAnkle_X' : [48, -1],
			'jRightAnkle_Y' : [49, 1],
			'jRightAnkle_Z' : [50, 1],
			'jRightBallFoot_X' : [51, -1],
			'jRightBallFoot_Y' : [52, 1],
			'jRightBallFoot_Z' : [53, -1],

			'jLeftHip_X' : [54, 1],
			'jLeftHip_Y' : [55, -1],
			'jLeftHip_Z' : [56, 1],
			'jLeftKnee_X' : [57, 1],
			'jLeftKnee_Y' : [58, -1],
			'jLeftKnee_Z' : [59, -1],
			'jLeftAnkle_X' : [60, 1],
			'jLeftAnkle_Y' : [61, -1],
			'jLeftAnkle_Z' : [62, 1],
			'jLeftBallFoot_X' : [63, 1],
			'jLeftBallFoot_Y' : [64, -1],
			'jLeftBallFoot_Z' : [65, -1],
			}


class Skeleton():
	"""
	This class is used to map the human whole body posture  of 66 Degrees of freedom into 
	a reduced skeleton.
	The output skeleton is defined by a json parameter file describing the joints used and
	their reference to the global skeleton.
	"""

	def __init__(self, urdf_file):
		self.urdf_model = URDF.from_xml_file(urdf_file)
		self.joint_angle = np.zeros(len(joint_mapping))
		self.position = np.zeros(len(Xsens_bodies)*3)

	def update_posture(self, input_joints):
		self.joint_angle = input_joints
		self.joint2pos()

	def getSegmentPose(self, linkname, q):
		root = self.urdf_model.get_root()
		bodies = self.urdf_model.get_chain(root, linkname)		
		H = np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
			
		while (linkname != root):
			joint = self.urdf_model.joint_map[self.urdf_model.parent_map[linkname][0]]
			[x, y, z] = [joint.origin.xyz[0], joint.origin.xyz[1], joint.origin.xyz[2]]
			[rx, ry, rz] = [joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2]]

			Rx = tools.quat2rot(np.array([np.cos(rx/2), np.sin(rx/2), 0., 0.]))
			Ry = tools.quat2rot(np.array([np.cos(ry/2), 0., np.sin(ry/2), 0.]))
			Rz = tools.quat2rot(np.array([np.cos(rz/2), 0., 0., np.sin(rz/2)]))
			R = Rx * Ry * Rz
			Hjoint = np.matrix([[R[0,0], R[0,1], R[0,2], x], [R[1,0], R[1,1], R[1,2], y], [R[2,0], R[2,1], R[2,2], z], [0, 0, 0, 1]])
	
			try:
				if joint.type == 'revolute':
					qi = q[joint.name]
					if joint.axis == [1., 0., 0.]:
						Hi = np.matrix([[1., 0., 0., 0.], [0., np.cos(qi), -np.sin(qi), 0.], [0., np.sin(qi), np.cos(qi), 0.], [0, 0, 0, 1]])
					elif joint.axis == [0., 1., 0.]: 
						Hi = np.matrix([[np.cos(qi), 0., np.sin(qi), 0.], [0., 1., 0., 0.], [-np.sin(qi), 0., np.cos(qi), 0.], [0, 0, 0, 1]])
					elif joint.axis == [0., 0., 1.]:
						Hi = np.matrix([[np.cos(qi), -np.sin(qi), 0., 0.], [np.sin(qi), np.cos(qi), 0., 0.], [0., 0., 1., 0.], [0, 0, 0, 1]])
					else:
						raise ValueError('Joint axis other than X, Y or Z not handled for now')
				else:
					Hi = np.matrix(np.identity(4))
				H = Hjoint * Hi * H
				linkname = joint.parent
			except:
				raise NameError('Joint ', joint.name, 'not defined in input joint vector')

		if 'root_RX' in q.keys():
			R = tools.quat2rot(np.array([np.cos(q['root_RZ']/2), 0., 0., np.sin(q['root_RZ']/2)])) * la.quat2rot(np.array([np.cos(q['root_RY']/2), 0., np.sin(q['root_RY']/2), 0.])) * la.quat2rot(np.array([np.cos(q['root_RX']/2), np.sin(q['root_RX']/2), 0., 0.]))
		elif 'root_qw' in q.keys():
			R = tools.quat2rot(np.array([q['root_qw'], q['root_qx'], q['root_qy'], q['root_qz']]))
		else:
			raise ValueError('Root rotation not defined')
		H_root = np.matrix([	[R[0,0], R[0,1], R[0,2], q['root_X']],
					[R[1,0], R[1,1], R[1,2], q['root_Y']], 
					[R[2,0], R[2,1], R[2,2], q['root_Z']], 
					[0., 0., 0., 1.]])
		H = H_root * H
		return H

	def joint2pos(self):
		"""
		Convert the joint angle data to cartesian position
		"""

		position_root = [0,0,0]
		quaternion_root = [1, 0, 0, 0]

		q = dict()
		for j in self.urdf_model.joints:
			if j.type != 'fixed':
				q[j.name] = joint_mapping[j.name][1] * self.joint_angle[joint_mapping[j.name][0]]*np.pi/180

		q['root_X'] = position_root[0]
		q['root_Y'] = position_root[1]
		q['root_Z'] = position_root[2]
		q['root_qw'] = quaternion_root[0]
		q['root_qx'] = quaternion_root[1]
		q['root_qy'] = quaternion_root[2]
		q['root_qz'] = quaternion_root[3]

		for num_seg, seg in enumerate(Xsens_segments):
			Hini = self.getSegmentPose(seg[0], q)
			id_ini = Xsens_bodies.index(seg[0])
			self.position[id_ini*3] = Hini[0,3]
			self.position[id_ini*3+1] = Hini[1,3]
			self.position[id_ini*3+2] = Hini[2,3]

			Hend = self.getSegmentPose(seg[1], q)
			id_end = Xsens_bodies.index(seg[1])
			self.position[id_end*3] = Hend[0,3]
			self.position[id_end*3+1] = Hend[1,3]
			self.position[id_end*3+2] = Hend[2,3]

	def visualise_from_joints(self, color='b'):
		"""
		Plot the skeleton based on joint angle data
		"""
		ax = plt.gca()

		lines = []

		ax.set_xlim(-2,2)
		ax.set_ylim(-2,2)
		ax.set_zlim(-1,1)

		for num_seg, seg in enumerate(Xsens_segments):
			id_ini = Xsens_bodies.index(seg[0])
			x_ini = self.position[id_ini*3]
			y_ini = self.position[id_ini*3+1]
			z_ini = self.position[id_ini*3+2]

			id_end = Xsens_bodies.index(seg[1])
			x_end = self.position[id_end*3]
			y_end = self.position[id_end*3+1]
			z_end = self.position[id_end*3+2]

			line, = ax.plot([x_ini, x_end],	[y_ini, y_end], [z_ini, z_end], 'm', color=color)
			lines.append(line)

		return lines

	def animate_skeleton(self, joint_seq):
		fig = plt.gcf()
		ax = plt.gca()

		lines = self.visualise_from_joints()

		def animate(i):
			self.update_posture(joint_seq[i])

			for num_seg, seg in enumerate(Xsens_segments):
				id_ini = Xsens_bodies.index(seg[0])
				id_end = Xsens_bodies.index(seg[1])

				line_length = np.linalg.norm(self.position[id_end*3:id_end*3+3]-self.position[id_ini*3:id_ini*3+3])

				lines[num_seg].set_data([self.position[id_ini*3],self.position[id_end*3]], 
					[self.position[id_ini*3+1],self.position[id_end*3+1]])
				lines[num_seg].set_3d_properties([self.position[id_ini*3+2],self.position[id_end*3+2]])

		anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=len(joint_seq), interval=20)

		plt.show()
		return fig

	def get_joint_mapping():
		return joint_mapping