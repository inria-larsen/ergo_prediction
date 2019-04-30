import numpy as np
import json
import torch
import math

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
		self.joints_whole_body = input_joints
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
			 	self.joint_reduce_body[num_joint] += np.deg2rad(self.joints_whole_body[id_joint*3+dim])

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

	def get_skeleton(self):
		size = 1.75

		list_points = self.skeleton_param['list_points']
		self.skeleton = np.zeros((len(list_points) + 1, 3))

		for num_point, point in enumerate(list_points):
			print(self.skeleton_param['segments'][point])
			id_joint = self.get_id_input_joint(self.skeleton_param['segments'][point]['links'][0])
			length = size * self.skeleton_param['segments'][point]['length']
			print(length)
			print()
			# self.skeleton[num_point+1, 2] = 
			print(self.joints_whole_body[id_joint*3:id_joint*3+3])

		return


	def update_posture_tensor(self, input_joints):
		self.joints_whole_body = input_joints
		self.joint_reduce_body = torch.zeros([self.reduced_dof])
		self.mapping_posture_tensor()

	def mapping_posture_tensor(self):
		num_dof = 0
		self.joint_reduce_body = torch.zeros([self.reduced_dof])
		for joint, num_joint in zip(self.list_all_joints, range(self.reduced_dof)):
			name_joint, dim_joint = joint.split('_')

			for i_joint in self.mapping_joints[name_joint]['input_joints']:
			 	id_joint = self.get_id_input_joint(i_joint)
			 	dim = self.dimensions[dim_joint]
			 	self.joint_reduce_body[num_joint] += self.joints_whole_body[id_joint*3+dim]*(2*math.pi)/360		