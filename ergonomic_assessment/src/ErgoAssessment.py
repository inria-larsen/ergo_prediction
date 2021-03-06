"""
Author: Adrien Malaisé
email: adrien.malaise@inria.fr
"""

import json
import math 
import numpy as np

from HumanPosture import HumanPosture

class ErgoAssessment: 
	"""
	This Class is used to compute automatic ergonomics assessment based on 
	ergonomic worksheet from industry.

	Example:
		>>> from ErgoAssessment import ErgoAssessment
		>>> ergo_assessment = ErgoAssessment("rula_config.json")
		>>> ergo_assessment.compute_ergo_scores(posture)
		>>> RULA_score = ergo_assessment['TABLE_RULA_C']

	Attributes:
		_list_ergo_score (dict): a dictionnary containing the ergonomics
			score to compute with:
			{
				"value (int/float)": the measure for the ergonomic score
				"info (dict)": contains all the parameter needed to compute the score
			}

		_config_file (string): name of the configuration file (.json) such as:
			rula_config.json
			reba_config.json
	"""

	def __init__(self, config_file):
		"""Constructor

		Args: 
			config_file (string): a configuration file in json with the 
				description of all the ergonomics score used by the module
		"""
		self._config_file = config_file
		self._list_ergo_score = dict()
		self._load_config_file()

	def _load_config_file(self):
		"""Initializes the _list_ergo_score according to the configuration file.
		All ergonomic score are initialized with the value 'NONE'.
		"""
		with open(self._config_file, 'r') as f:
			config_ergo = json.load(f)

		for ergo_score in config_ergo['ERGO_SCORE']:
			if(config_ergo['ERGO_SCORE'][ergo_score]['type_score'] == 'jointAngle'):
				for threshold_list in config_ergo['ERGO_SCORE'][ergo_score]['threshold']:
					for threshold in threshold_list:
						threshold[0] = math.radians(threshold[0])
						threshold[1] = math.radians(threshold[1])

			self._list_ergo_score[ergo_score] = {
				"value": "NONE",
				"info": config_ergo['ERGO_SCORE'][ergo_score]
				}

	def compute_ergo_scores(self, posture):
		"""Compute all the ergonomic score related to the posture in parameter

		Args: 
			posture (list): the joint angle values of the human skeleton
		"""
		self._initiliaze_score()
		self.posture = posture
		for ergo_score in self._list_ergo_score:
			self._compute_score(self._list_ergo_score[ergo_score])
		return 

	def _compute_score(self, ergo_score):
		"""Compute a specific ergonomic score send in parameter

		Args: 
			ergo_score (dict): One of the ergonomic score from _list_ergo_score
		"""
		if not(ergo_score['info']["related_score"] == "none"):
			for related_score in ergo_score['info']["related_score"]:
				if(self.__getitem__(related_score) == "NONE"):
					self._compute_score(self._list_ergo_score[related_score])

		if ergo_score['info']["type_score"] == "jointAngle":
			ergo_score['value'] = self._compute_joint_score(ergo_score)
		elif ergo_score['info']["type_score"] == "table":
			ergo_score['value'] = self._compute_table_score(ergo_score)
		elif ergo_score['info']["type_score"] == "max_value":
			ergo_score['value'] = self._compute_max_score(ergo_score)
		elif ergo_score['info']["type_score"] == "value":
			ergo_score['value'] = ergo_score['info']['value']

	def _compute_joint_score(self, local_score):
		"""Compute the local ergonomic score of the joint related to this score
		"""
		ergo_value = 0
		related_joint = local_score['info']['related_joint']
		for joint, num_joint in zip(related_joint, range(len(related_joint))):
			joint_angle = self.posture.get_joint_angle(joint)
			if joint_angle <= -np.pi/2:
				print(joint, joint_angle)
				joint_angle = -np.pi/2 + 0.1
			elif joint_angle >= np.pi/2:
				joint_angle = np.pi/2 - 0.1
			for threshold, i in zip(local_score['info']['threshold'][num_joint], 
				range(len(local_score['info']['threshold'][num_joint]))):
				if threshold[0] < joint_angle <= threshold[1]:
					ergo_temp = local_score['info']['related_value'][num_joint][i]
			try:
				ergo_value += ergo_temp
			except:
				raise NameError(joint, threshold, joint_angle)

		return ergo_value

	def _compute_table_score(self, table_score):
		""" Compute the ergonomic score related to a table
		"""
		related_score = table_score['info']['related_score']
		ergo_value = table_score['info']['table']
		for score in related_score:
			id_score = self.__getitem__(score)
			ergo_value = ergo_value[id_score-1]

		return ergo_value

	def _compute_max_score(self, local_score):
		"""Return the maximum ergonomic score in a list
		"""
		related_score = local_score['info']['related_score']
		ergo_value = []
		for score in related_score:
			ergo_value.append(self.__getitem__(score))
		ergo_value = max(ergo_value)
		return ergo_value

	def _initiliaze_score(self):
		"""Initialize all the score to the 'NONE' value
		"""
		for ergo_score in self._list_ergo_score:
			self._list_ergo_score[ergo_score]['value'] = "NONE"

	def get_list_score(self):
		return self._list_ergo_score

	def get_list_score_name(self):
		score = [score for score in self._list_ergo_score]
		return  score

	def get_ergo_score(self, name_score): 
		"""Return the dictionnary of the ergonomic score with the name in input.
		"""
		try:
			return self._list_ergo_score[name_score]
		except KeyError:
			raise KeyError(name_score + ' is not in _list_ergo_score')

	def __getitem__(self, name_score):
		"""Return the value of an ergonomic score with the name in input
		"""
		try:
			return self._list_ergo_score[name_score]['value']
		except KeyError:
			raise KeyError(name_score + ' is not in _list_ergo_score')

	def __setitem__(self, name_score, value):
		"""Set the value of an ergonomic score with the name in input
		"""
		try:
			self._list_ergo_score[name_score]['value'] = value
		except KeyError as err:
			raise KeyError(name_score + ' is not in _list_ergo_score')
