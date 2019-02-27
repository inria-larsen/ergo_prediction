import numpy as np
import pandas as pd
import json
import math

from HumanPosture import HumanPosture

class ErgoAssessment():
	"""
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

		config_human_body = config_ergo['HUMAN_SKELETON']['config_file']
		self.load_human_body_config(config_human_body)

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

	def load_human_body_config(self, config_file):
		self.human_body = HumanPosture(config_file)
		return

	def compute_ergo_score(self, posture):
		self.posture = posture
		for ergo_score, num_score in zip(self.list_ergo_score, range(self.nbr_score)):
			if(ergo_score["type_score"] == "jointAngle"):
				self.list_ergo_value[num_score] = self.compute_joint_score(ergo_score)
			elif(ergo_score["type_score"] == "table"):
				self.list_ergo_value[num_score] = self.compute_table_score(ergo_score)

		print(self.list_ergo_value)
		return 

	def compute_joint_score(self, local_score):
		related_joint = local_score['related_joint']
		joint_angle = self.posture.get_joint_angle(related_joint)
		
		for threshold, i in zip(local_score['threshold'], range(len(local_score['threshold']))):
			if(joint_angle > math.radians(threshold[0]) and joint_angle < math.radians(threshold[1])):
				ergo_value = local_score['related_value'][i]

		return ergo_value

	def compute_table_score(self, name_score):
		# df_table = pd.read_csv(name_score['related_table'], header = False)
		# print(df_table)
		return 0



