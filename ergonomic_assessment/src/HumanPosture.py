
class HumanPosture():
	"""
	This class is used to compute ergonomic score based on the human posture (23 segments).
	A reduce skeleton is used to 

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
		return self.ergo_score_neck 

	def compute_score_trunk(self):
		return self.ergo_score_trunk 


		