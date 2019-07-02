import numpy as np
import json
import torch
import math
import tools

class HumanPosture():
	"""
	This class is used to map the human whole body posture  of 66 Degrees of freedom into 
	a reduced skeleton.
	The output skeleton is defined by a json parameter file describing the joints used and
	their reference to the global skeleton.
	"""
	def __init__(self, param_file):
		self.param_file = param_file
		self.load_param(param_file)
		self.joints_whole_body = np.zeros(self.input_dof)
		self.joint_reduce_body = np.zeros(self.reduced_dof)

	def load_param(self, param_file):
		with open(param_file, 'r') as f:
			param_mapping = json.load(f)

		input_param = param_mapping['INPUT_JOINTS']
		self.input_dof = input_param['nbr_dof']
		self.input_joints = input_param['input_joints']
		self.dimensions = input_param['dimensions']

		self.skeleton_param = param_mapping['SKELETON']

		self.mapping_joints = param_mapping['REDUCED_JOINTS']

		self.reduced_dof = 0
		self.reduced_joints = []

		self.list_all_joints = []
		for joint in self.mapping_joints:
			self.reduced_joints.append(joint)
			for dim in self.mapping_joints[joint]['mapping']:
				self.list_all_joints.append(joint + '_' + dim)
				self.reduced_dof += 1

	def update_posture(self, input_joints):
		self.joints_whole_body = np.deg2rad(input_joints)
		self.joint_reduce_body = np.zeros(self.reduced_dof)
		self.mapping_posture()

	def mapping_posture(self):
		num_dof = 0
		self.joint_reduce_body = np.zeros(self.reduced_dof)
		for joint, num_joint in zip(self.list_all_joints, range(self.reduced_dof)):
			name_joint, dim_joint = joint.split('_')

			for i_joint in self.mapping_joints[name_joint]['input_joints']:
			 	id_joint = self.get_id_input_joint(i_joint)
			 	dim = self.dimensions[dim_joint]
			 	self.joint_reduce_body[num_joint] += self.joints_whole_body[id_joint*3+dim]

	def set_posture(self, joints_dict):
		self.joints_whole_body = np.zeros(self.input_dof)
		count = 0
		for joint, num_joint in zip(self.input_joints, range(self.input_dof)):
			for i in range(len(joints_dict[joint]["value"])):
				self.joints_whole_body[count] = joints_dict[joint]["value"][i]
				count += 1

		self.mapping_posture()

	def get_id_input_joint(self, name_joint):
		id_joint = self.input_joints.index(name_joint)
		return id_joint

	def get_id_reduced_joint(self, name_joint):
		id_joint = self.reduced_joints.index(name_joint)
		return id_joint

	def get_joint_angle(self, name_joint):
		id_joint = self.list_all_joints.index(name_joint)
		return self.joint_reduce_body[id_joint]

	def get_point_pos(self, linkname):
		"""
		Return the cartesian position of the end point of a segment
		"""

		H = np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
		direction_joint = ['pitch', 'roll', 'yaw']

		link_skip = ['PelvisR', 'PelvisL', 'ShoulderR', 'ShoulderL']

		while linkname != 'root':
			segment = self.skeleton_param['list_segments'][linkname]
			joint = segment['points'][0]
			theta = np.zeros(3)

			length = segment['length']*np.asarray(segment['init_dir'])
			length = length*self.size

			Rx = tools.quat2rot(np.array([np.cos(theta[0]), np.sin(theta[0]), 0., 0.]))
			Ry = tools.quat2rot(np.array([np.cos(theta[1]), 0., np.sin(theta[1]), 0.]))
			Rz = tools.quat2rot(np.array([np.cos(theta[2]), 0., 0., np.sin(theta[2])]))
			R = Rx * Ry * Rz
			Hjoint = np.matrix([[R[0,0], R[0,1], R[0,2], length[0]], [R[1,0], R[1,1], R[1,2], length[1]], [R[2,0], R[2,1], R[2,2], length[2]], [0, 0, 0, 1]])

			if linkname in link_skip:
				linkname = segment['parent']
				
			else:
				jointAngle = np.zeros(3)
				for num_dir, direction in enumerate(direction_joint):
					jointAngle[num_dir] = np.deg2rad(self.get_joint_angle(joint + '_' + direction))

					qi = jointAngle[num_dir]

					if num_dir == 0:
						Hi = np.matrix([[1., 0., 0., 0.], [0., np.cos(qi), -np.sin(qi), 0.], [0., np.sin(qi), np.cos(qi), 0.], [0, 0, 0, 1]])
					elif num_dir == 1:
						Hi = np.matrix([[np.cos(qi), 0., np.sin(qi), 0.], [0., 1., 0., 0.], [-np.sin(qi), 0., np.cos(qi), 0.], [0, 0, 0, 1]])
					elif num_dir == 2:
						Hi = np.matrix([[np.cos(qi), -np.sin(qi), 0., 0.], [np.sin(qi), np.cos(qi), 0., 0.], [0., 0., 1., 0.], [0, 0, 0, 1]])
					
					H = Hi * H
					
			H = Hjoint * H
			linkname = segment['parent']

		pos = np.zeros(3)
		pos[0] = H[0, 3]
		pos[1] = H[1, 3]
		pos[2] = H[2, 3]

		return pos

	def joint2pos(self):
		"""
		Convert the joint angle data to cartesian position
		"""
		self.size = 1.75

		list_points = self.skeleton_param['list_points']
		list_segments = self.skeleton_param['list_segments']

		data_pos = np.zeros(3)

		for num_segment, segment in enumerate(list_segments):
			point = self.skeleton_param['list_segments'][segment]['points'][1]
			pos = self.getPointPos(segment)
			self.skeleton_param['list_points'][point]['position'] = pos

	def visualise_from_joints(self, ax):
		"""
		Plot the skeleton based on joint angle data
		"""
		ax.set_xlim(-2,2)
		ax.set_ylim(-2,2)
		ax.set_zlim(-1,1)

		self.joint2pos()
		skeleton_points = self.skeleton_param['list_points']
		skeleton_segments = self.skeleton_param['list_segments']

		for seg in skeleton_segments:
			point_ini = skeleton_segments[seg]['points'][0]
			x_ini = skeleton_points[point_ini]['position'][0]
			y_ini = skeleton_points[point_ini]['position'][1]
			z_ini = skeleton_points[point_ini]['position'][2]

			point_fin = skeleton_segments[seg]['points'][1]
			x_fin = skeleton_points[point_fin]['position'][0]
			y_fin = skeleton_points[point_fin]['position'][1]
			z_fin = skeleton_points[point_fin]['position'][2]

			ax.plot([x_ini, x_fin],	[y_ini, y_fin], [z_ini, z_fin], 'm')


		return ax

		