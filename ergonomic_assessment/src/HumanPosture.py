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
		self.list_all_joints.sort()	
		self.reduced_joints.sort()

	def update_posture(self, input_joints, num_frame=0):
		if(type(input_joints) == np.ndarray):
			self.joints_whole_body = np.asarray(input_joints)
			self.joint_reduce_body = np.zeros(self.reduced_dof)

		elif(type(input_joints) == torch.Tensor):
			self.joints_whole_body = input_joints
			self.joint_reduce_body = torch.zeros(self.reduced_dof, requires_grad=True)

		self.mapping_posture()
		return self.joint_reduce_body

	def mapping_posture(self):
		num_dof = 0

		# self.joint_reduce_body = np.zeros(self.reduced_dof)
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

	def set_posture(self, joint_data):
		self.joint_reduce_body = joint_data

	def get_id_input_joint(self, name_joint):
		id_joint = self.input_joints.index(name_joint)
		return id_joint

	def get_id_reduced_joint(self, name_joint):
		id_joint = self.reduced_joints.index(name_joint)
		return id_joint

	def get_joint_angle(self, name_joint):
		id_joint = self.list_all_joints.index(name_joint)
		return self.joint_reduce_body[id_joint]

	def get_dim_input(self):
		return self.input_dof

	def get_dim_reduce(self):
		return self.reduced_dof

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


	def reduce2complete(self, data_reduce):
		# Joint 1 => 4
		id_joint = self.get_id_reduced_joint('jTorso')
		self.joints_whole_body[0] = data_reduce[id_joint*3+1]/4
		self.joints_whole_body[1] = data_reduce[id_joint*3+2]/4
		self.joints_whole_body[2] = data_reduce[id_joint*3]/4

		self.joints_whole_body[3] = data_reduce[id_joint*3+1]/4
		self.joints_whole_body[4] = data_reduce[id_joint*3+2]/4
		self.joints_whole_body[5] = data_reduce[id_joint*3]/4

		self.joints_whole_body[6] = data_reduce[id_joint*3+1]/4
		self.joints_whole_body[7] = data_reduce[id_joint*3+2]/4
		self.joints_whole_body[8] = data_reduce[id_joint*3]/4

		self.joints_whole_body[9] = data_reduce[id_joint*3+1]/4
		self.joints_whole_body[10] = data_reduce[id_joint*3+2]/4
		self.joints_whole_body[11] = data_reduce[id_joint*3]/4

		self.joints_whole_body[12:15] = 0.0

		id_joint = self.get_id_reduced_joint('jNeck')
		self.joints_whole_body[15] = data_reduce[id_joint*3+1]
		self.joints_whole_body[16] = data_reduce[id_joint*3+2]
		self.joints_whole_body[17] = data_reduce[id_joint*3]

		self.joints_whole_body[18:21] = 0.0

		id_joint = self.get_id_reduced_joint('jRShoulder')
		self.joints_whole_body[21] = data_reduce[id_joint*3+1]
		self.joints_whole_body[22] = data_reduce[id_joint*3+2]
		self.joints_whole_body[23] = data_reduce[id_joint*3]

		id_joint = self.get_id_reduced_joint('jRElbow')
		self.joints_whole_body[24] = data_reduce[id_joint*3+1]
		self.joints_whole_body[25] = data_reduce[id_joint*3+2]
		self.joints_whole_body[26] = data_reduce[id_joint*3]

		id_joint = self.get_id_reduced_joint('jRWrist')
		self.joints_whole_body[27] = data_reduce[id_joint*3+1]
		self.joints_whole_body[28] = data_reduce[id_joint*3+2]
		self.joints_whole_body[29] = data_reduce[id_joint*3]

		self.joints_whole_body[30:33] = 0.0

		id_joint = self.get_id_reduced_joint('jLShoulder')
		self.joints_whole_body[33] = data_reduce[id_joint*3+1]
		self.joints_whole_body[34] = data_reduce[id_joint*3+2]
		self.joints_whole_body[35] = data_reduce[id_joint*3]

		id_joint = self.get_id_reduced_joint('jLElbow')
		self.joints_whole_body[36] = data_reduce[id_joint*3+1]
		self.joints_whole_body[37] = data_reduce[id_joint*3+2]
		self.joints_whole_body[38] = data_reduce[id_joint*3]

		id_joint = self.get_id_reduced_joint('jLWrist')
		self.joints_whole_body[39] = data_reduce[id_joint*3+1]
		self.joints_whole_body[40] = data_reduce[id_joint*3+2]
		self.joints_whole_body[41] = data_reduce[id_joint*3]

		id_joint = self.get_id_reduced_joint('jRHip')
		self.joints_whole_body[42] = data_reduce[id_joint*3+1]
		self.joints_whole_body[43] = data_reduce[id_joint*3+2]
		self.joints_whole_body[44] = data_reduce[id_joint*3]

		id_joint = self.get_id_reduced_joint('jRKnee')
		self.joints_whole_body[45] = data_reduce[id_joint*3+1]
		self.joints_whole_body[46] = data_reduce[id_joint*3+2]
		self.joints_whole_body[47] = data_reduce[id_joint*3]

		self.joints_whole_body[48:51] = 0.0

		self.joints_whole_body[51:54] = 0.0

		id_joint = self.get_id_reduced_joint('jLHip')
		self.joints_whole_body[54] = data_reduce[id_joint*3+1]
		self.joints_whole_body[55] = data_reduce[id_joint*3+2]
		self.joints_whole_body[56] = data_reduce[id_joint*3]

		id_joint = self.get_id_reduced_joint('jLKnee')
		self.joints_whole_body[57] = data_reduce[id_joint*3+1]
		self.joints_whole_body[58] = data_reduce[id_joint*3+2]
		self.joints_whole_body[59] = data_reduce[id_joint*3]

		self.joints_whole_body[60:63] = 0.0

		self.joints_whole_body[63:66] = 0.0



		return self.joints_whole_body

		