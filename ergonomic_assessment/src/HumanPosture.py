import numpy as np
import json

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
		self.input_joints = input_param['xsens_joints']
		self.dimensions = input_param['dimensions']

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


	def get_id_input_joint(self, name_joint):
		id_joint = self.input_joints.index(name_joint)
		return id_joint

	def get_id_reduced_joint(self, name_joint):
		id_joint = self.reduced_joints.index(name_joint)
		return id_joint

	def get_joint_angle(self, name_joint):
		id_joint = self.list_all_joints.index(name_joint)
		return self.joint_reduce_body[id_joint]







		