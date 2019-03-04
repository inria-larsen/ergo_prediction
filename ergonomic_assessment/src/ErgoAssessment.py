# TODO missing module docstring
# TODO consider using a sensible color scheme (docstring more visible than comments)
# TODO consider limiting lines at 79 characters (PEP8)
import json
import math 

from HumanPosture import HumanPosture

class ErgoAssessment: 
	"""
	This class is used to compute ergonomics score according to a configuration file 
	in parameter.
	Score are both local and global.
	"""  # FIXME docstring should be completed with api/usage
	def __init__(self, config_file):
		"""
		input: a configuration file in json with the description of all the ergonomics score used 
		by the module
		"""
		self._config_file = config_file
		self._list_ergo_score = []
		self._load_config_file()

	def _load_config_file(self):  # FIXME missing docstring
		"""

		"""
		with open(self._config_file, 'r') as f:
			config_ergo = json.load(f)

		for ergo_score in config_ergo['ERGO_SCORE']:
			if(config_ergo['ERGO_SCORE'][ergo_score]['type_score'] == 'jointAngle'):
				for threshold_list in config_ergo['ERGO_SCORE'][ergo_score]['threshold']:
					for threshold in threshold_list:
						threshold[0] = math.radians(threshold[0])
						threshold[1] = math.radians(threshold[1])
	
			self._list_ergo_score.append({
				"name": ergo_score,
				"value": 0,
				"info": config_ergo['ERGO_SCORE'][ergo_score]
				})

		self.nbr_score = len(self._list_ergo_score)

	def compute_ergo_scores(self, posture):
		"""
		Compute all the ergonomic score related to the posture in parameter
		input: a list or array <nbr_dof, 1> with all joint angle values
		output: a list <nbr_score, 1> of all computed ergonomics score
		"""
		self._initiliaze_score()
		self.posture = posture
		for ergo_score in self._list_ergo_score:
			self._compute_score(ergo_score)

	def _compute_score(self, ergo_score):
		if(not(ergo_score['info']["related_score"] == "none")):
			for related_score in ergo_score['info']["related_score"]:
				if(self.__getitem__(related_score) == 0):
					self._compute_score(self.get_ergo_score(related_score))

		if ergo_score['info']["type_score"] == "jointAngle":
			ergo_score['value'] = self._compute_joint_score(ergo_score)
		elif ergo_score['info']["type_score"] == "table":
			ergo_score['value'] = self._compute_table_score(ergo_score)
		elif ergo_score['info']["type_score"] == "max_value":
			ergo_score['value'] = self._compute_max_score(ergo_score)

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

	def _compute_table_score(self, table_score):# FIXME missing docstring
		"""

		"""
		related_score = table_score['info']['related_score']
		ergo_value = table_score['info']['table']
		for score in related_score:
			id_score = self.__getitem__(score)
			ergo_value = ergo_value[id_score-1]

		return ergo_value

	def _compute_max_score(self, local_score):# FIXME missing docstring
		"""
		
		"""
		related_score = local_score['info']['related_score']
		ergo_value = []
		for score in related_score:
			ergo_value.append(self.__getitem__(score))
		ergo_value = max(ergo_value)
		return ergo_value

	def _initiliaze_score(self): # FIXME missing docstring
		for ergo_score in self._list_ergo_score:
			ergo_score['value'] = 0


	def get_ergo_score(self, name_score):  # FIXME missing docstring 
		"""
		"""
		for ergo_score in self._list_ergo_score:
			if(ergo_score['name'] == name_score):
				return ergo_score

	def __getitem__(self, name_score): # FIXME missing docstring
		"""
		"""
		for ergo_score in self._list_ergo_score:
			if(ergo_score['name'] == name_score):
				return ergo_score['value']

	def __setitem__(self, name_score, value):  # FIXME missing docstring
		"""
		"""
		for ergo_score in self._list_ergo_score:
			if(ergo_score['name'] == name_score):
				ergo_score['value'] = value
