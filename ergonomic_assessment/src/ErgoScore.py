import numpy as np
import pandas as pd

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


def id_segment(name_segment):
	return list_segments.index(name_segment)*3

def measure_angle(p1, p2, p3):
	num = (p1[0]-p2[0])*(p3[0]-p2[0]) + (p1[1]-p2[1])*(p3[1]-p2[1]) + (p1[2]-p2[2])*(p3[2]-p2[2])
	den = np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2])) * np.sqrt((p3[0]-p2[0])*(p3[0]-p2[0]) + (p3[1]-p2[1])*(p3[1]-p2[1]) + (p3[2]-p2[2])*(p3[2]-p2[2]))
	angle = 180 - np.arccos(num/den)*180/np.pi
	return angle


class ErgoScore():
	"""
	This class is used to compute ergonomic score based on the human posture (23 segments).

	RULA score:
	 - ergo_score_global <int>
	 - ergo_score_shoulder <int>
	 - ergo_score_elbow <int>
	 - ergo_score_trunk <int>
	 - ergo_score_neck <int>
	 - ergo_score_wrist <int>

	"""
	def __init__(self):
		self.ergo_score_global = 0
		self.ergo_score_shoulder = 0
		self.ergo_score_elbow = 0
		self.ergo_score_wrist = 0
		self.ergo_score_neck = 0
		self.ergo_score_trunk = 0


	def update_posture(self, segments):
		self.posture = segments
		return

	def compute_score_global(self):
		return self.ergo_score_global

	def compute_score_shoulder(self):
		return self.ergo_score_shoulder 

	def compute_score_elbow(self):
		return self.ergo_score_elbow 

	def compute_score_wrist(self):
		return self.ergo_score_wrist 

	def compute_score_neck(self):
		self.ergo_score_neck = 0
		return self.ergo_score_neck


	def compute_score_trunk(self):
		self.ergo_score_trunk = 0

		pelvis = self.posture[id_segment('Pelvis'):id_segment('Pelvis')+3]
		T8 = self.posture[id_segment('T8'):id_segment('T8')+3]
		origin = pelvis - np.asarray([0, 0, pelvis[2]])

		trunk_bend = measure_angle(T8, pelvis, origin)

		if(trunk_bend > 60):
			self.ergo_score_trunk += 4
		elif(trunk_bend > 20):
			self.ergo_score_trunk += 3
		elif(trunk_bend > 5):
			self.ergo_score_trunk += 2
		else:
			self.ergo_score_trunk += 1

		return self.ergo_score_trunk

	def get_posture(self):
		return self.posture


		