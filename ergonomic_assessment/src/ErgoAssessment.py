"""
Author: Adrien Malaisé
email: adrien.malaise@inria.fr
"""

import json
import math 

from HumanPosture import HumanPosture

class ErgoAssessment: 
	"""
	This Class is used to compute automatic ergonomics assessment based on 
	ergonomic worksheet from industry.

	Example:
		$ from ErgoAssessment import ErgoAssessment
		$ ergo_assessment = ErgoAssessment("rula_config.json")
		$ ergo_assessment.compute_ergo_scores(posture)
		$ RULA_score = ergo_assessment['TABLE_RULA_C']

	Attributes:
		_list_ergo_score: a list of dictionnary represented the ergonomics
			score to compute with:
			{
				"name": a string to represent the ergonomic score
				"value": a float or int depending the type of ergonomic score
				"info": a dictionnary of all the parameter needed to compute the score
			}
		_config_file: a string of the name of the configuration file (.json) such as:
			rula_config.json
			reba_config.json
	"""

	def __init__(self, config_file):
		"""
		input: a configuration file in json with the description of all the 
		ergonomics score used by the module
		"""
		self._config_file = config_file
		self._list_ergo_score = dict()
		self._load_config_file()

	def _load_config_file(self):
		"""
		This function initialize the _list_ergo_score attributes according to
		configuration file.
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
		"""
		Compute all the ergonomic score related to the posture in parameter
		input: a list or array <nbr_dof, 1> with all joint angle values
		"""
		self._initiliaze_score()
		self.posture = posture
		for ergo_score in self._list_ergo_score:
			self._compute_score(self._list_ergo_score[ergo_score])

	def _compute_score(self, ergo_score):
		"""
		Compute a specific ergonomic score send in parameter.
		input: ergo_score is a dictionnary of one of the ergonomic score from
		_list_ergo_score.
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
		"""
		Compute the local ergonomic score of the joint related to this score
		"""
		ergo_value = 0
		related_joint = local_score['info']['related_joint']
		for joint, num_joint in zip(related_joint, range(len(related_joint))):
			joint_angle = self.posture.get_joint_angle(joint)
			for threshold, i in zip(local_score['info']['threshold'][num_joint], 
				range(len(local_score['info']['threshold'][num_joint]))):
				if threshold[0] < joint_angle < threshold[1]:
					ergo_temp = local_score['info']['related_value'][num_joint][i]
			ergo_value += ergo_temp

		return ergo_value

	def _compute_table_score(self, table_score):
		""" 
		Compute the ergonomic score related to a table
		"""
		related_score = table_score['info']['related_score']
		ergo_value = table_score['info']['table']
		for score in related_score:
			id_score = self.__getitem__(score)
			ergo_value = ergo_value[id_score-1]

		return ergo_value

	def _compute_max_score(self, local_score):
		"""
		Return the maximum ergonomic score in a list
		"""
		related_score = local_score['info']['related_score']
		ergo_value = []
		for score in related_score:
			ergo_value.append(self.__getitem__(score))
		ergo_value = max(ergo_value)
		return ergo_value

	def _initiliaze_score(self):
		for ergo_score in self._list_ergo_score:
			self._list_ergo_score[ergo_score]['value'] = "NONE"

	def get_list_score(self):
		return self._list_ergo_score

	def get_ergo_score(self, name_score): 
		"""
		Return the dictionnary of the ergonomic score with the name in input.
		name_score: string
		"""
		try:
			return self._list_ergo_score[name_score]
		except KeyError:
			raise KeyError(name_score + ' is not in _list_ergo_score')

	def __getitem__(self, name_score):
		"""
		Return the value of an ergonomic score with the name in input
		"""
		try:
			return self._list_ergo_score[name_score]['value']
		except KeyError:
			raise KeyError(name_score + ' is not in _list_ergo_score')

	def __setitem__(self, name_score, value):
		"""
		Set the value of an ergonomic score with the name in input
		"""
		try:
			self._list_ergo_score[name_score]['value'] = value
		except KeyError as err:
			raise KeyError(name_score + ' is not in _list_ergo_score')
