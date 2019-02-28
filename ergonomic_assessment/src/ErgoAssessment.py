# TODO missing module docstring
# TODO consider using a sensible color scheme (docstring more visible than comments)
# TODO consider limiting lines at 79 characters (PEP8)
import numpy as np  # TODO why np§ consider replacing np.deg2rad by math.radians (fewer dependency)
import pandas as pd  # FIXME unneeded import
import json
import math  # FIXME unneeded import

from HumanPosture import HumanPosture

class ErgoAssessment():  # FIXME unneeded parentheses
	"""
	This class is used to compute ergonomics score according to a configuration file 
	in parameter.
	Score are both local and global.
	"""  # FIXME docstring should be completed with api/usage
	def __init__(self, config_file):
		self.config_file = config_file
		self.list_ergo_score_name = []  # TODO consider using a dictionary (Dict[str, Tuple[ErgoScore, float]]) rather than a tuple of (hand-synchronized) lists
		self.list_ergo_score = []
		self.list_ergo_value = []
		self.load_config_file()

	def load_config_file(self):  # TODO make private? FIXME missing docstring
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

		return  # FIXME unneeded

	def compute_ergo_score(self, posture):
		"""
		Compute all the ergonomic score related to the posture in parameter
		"""  # FIXME docstring should include information/type about argument
		self.posture = posture
		for ergo_score, num_score in zip(self.list_ergo_score, range(self.nbr_score)):
			if(ergo_score["type_score"] == "jointAngle"):
				self.list_ergo_value[num_score] = self.compute_joint_score(ergo_score)
			elif(ergo_score["type_score"] == "table"):
				self.list_ergo_value[num_score] = self.compute_table_score(ergo_score)

		return self.list_ergo_value

	def compute_joint_score(self, local_score):  # TODO make private: _compute_joint_score?
		"""
		Compute the local ergonomic score of the joint related to this score
		"""
		ergo_value = 0
		related_joint = local_score['related_joint']
		for joint, num_joint in zip(related_joint, range(len(related_joint))):
			joint_angle = self.posture.get_joint_angle(joint)
			for threshold, i in zip(local_score['threshold'][num_joint], range(len(local_score['threshold'][num_joint]))):
				if np.deg2rad(threshold[0]) < joint_angle < np.deg2rad(threshold[1]):  # TODO consider converting threshold values at load time (to avoid doing it for each frame)
					ergo_temp = local_score['related_value'][num_joint][i]
			ergo_value += ergo_temp

		return ergo_value

	def compute_table_score(self, table_score):  # TODO make private?
		# df_table = pd.read_csv(table_score['related_table'], index_col = 0)
		# print(df_table)
		# x_data = table_score['related_score']['x']
		# y_data = table_score['related_score']['y']

		# x_value = self.get_score_value(x_data)
		# y_value = self.get_score_value(y_data)

		return 0

	def get_score_value(self, name_score):  # FIXME missing docstring  # TODO consider renaming to __getitem__
		id_score = self.list_ergo_score_name.index(name_score)  # FIXME catch case of name_score not in list
		return self.list_ergo_value[id_score]
