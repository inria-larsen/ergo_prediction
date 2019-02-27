import numpy as np
import pandas as pd
import json
import math

from HumanPosture import HumanPosture

class ErgoAssessment():
	"""
	This class is used to compute ergonomics score according to a configuration file 
	in parameter.
	Score are both local and global.
	"""
	def __init__(self, config_file):
		self.config_file = config_file
		self.list_ergo_score_name = []
		self.list_ergo_score = []
		self.list_ergo_value = []
		self.load_config_file()

	def load_config_file(self):
		with open(self.config_file, 'r') as f:
			config_ergo = json.load(f)

		for ergo_score in config_ergo['ERGO_SCORE']:
			if(config_ergo['ERGO_SCORE'][ergo_score]['type_score'] == 'jointAngle'):
				self.list_ergo_score_name.insert(0, ergo_score)
				self.list_ergo_score.insert(0, config_ergo['ERGO_SCORE'][ergo_score])
				self.list_ergo_value.insert(0, 0.0)
			else:
				self.list_ergo_score_name.append(ergo_score)
				self.list_ergo_score.append(config_ergo['ERGO_SCORE'][ergo_score])
				self.list_ergo_value.append(0.0)

		self.nbr_score = len(self.list_ergo_score)

		return

	def compute_ergo_score(self, posture):
		"""
		Compute all the ergonomic score related to the posture in parameter
		"""
		self.posture = posture
		for ergo_score, num_score in zip(self.list_ergo_score, range(self.nbr_score)):
			if(ergo_score["type_score"] == "jointAngle"):
				self.list_ergo_value[num_score] = self.compute_joint_score(ergo_score)
			elif(ergo_score["type_score"] == "table"):
				self.list_ergo_value[num_score] = self.compute_table_score(ergo_score)

		return self.list_ergo_value

	def compute_joint_score(self, local_score):
		"""
		Compute the local ergonomic score of the joint related to this score
		"""
		ergo_value = 0
		related_joint = local_score['related_joint']
		for joint, num_joint in zip(related_joint, range(len(related_joint))):
			joint_angle = self.posture.get_joint_angle(joint)
			for threshold, i in zip(local_score['threshold'][num_joint], range(len(local_score['threshold'][num_joint]))):
				if(joint_angle > np.deg2rad(threshold[0]) and joint_angle < np.deg2rad(threshold[1])):
					ergo_temp = local_score['related_value'][num_joint][i]
			ergo_value += ergo_temp

		return ergo_value

	def compute_table_score(self, table_score):
		df_table = pd.read_csv(table_score['related_table'], index_col = 0)
		# for related_score in table_score['related_score']:
		# 	print(related_score)
		print(df_table)
		return 0

	def get_score_value(self, name_score):
		id_score = self.list_ergo_score_name.index(name_score)
		return self.list_ergo_value[id_score]



