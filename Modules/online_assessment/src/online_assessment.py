# Author: Adrien Malaisé
# email: adrien.malaise@inria.fr

import yarp
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ergonomic_assessment/src/')))

from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture

class ErgonomicAssessmentModule(yarp.RFModule):
	"""
	This module compute online the ergonomic scores of postures related
	to worksheets from industry
	"""
	def __init__(self):
		yarp.RFModule.__init__(self)
		self.handlerPort = yarp.Port()

	def configure(self, rf):
		self.handlerPort.open("/ErgonomicAssessmentModule")
		self.attach(self.handlerPort)

		self.port_posture = yarp.BufferedPortBottle()
		self.port_posture.open('/ergo_pred/posture:i')

		path_config_posture = rf.find('posture_config').toString()
		self.posture = HumanPosture(path_config_posture)
		
		path_config_ergo = rf.find('ergo_config').toString()
		self.ergo_assessment = ErgoAssessment(path_config_ergo)

		self.list_port_ergo = dict()

		for ergo_score in self.ergo_assessment.get_list_score():
			self.list_port_ergo[ergo_score] = yarp.BufferedPortBottle()
			self.list_port_ergo[ergo_score].open("/ergo_pred/" + ergo_score.lower() +':o')

		return True

	def close(self):
		self.port_posture.close()
		self.handlerPort.close()
		return True

	def updateModule(self):
		b_in = self.port_posture.read(True)

		data = b_in.toString().split(' ')
		value = list(map(float, data))

		self.posture.update_posture(value)
		self.ergo_assessment.compute_ergo_scores(self.posture)

		for ergo_score in self.list_port_ergo:
			b_out = self.list_port_ergo[ergo_score].prepare()
			b_out.clear()
			b_out.addDouble(self.ergo_assessment[ergo_score])
			self.list_port_ergo[ergo_score].write()

		return True
		
if __name__=="__main__":
	yarp.Network.init()

	rf = yarp.ResourceFinder()
	rf.setVerbose(True)
	rf.setDefaultContext("ergo_pred")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	mod_ergo = ErgonomicAssessmentModule()
	mod_ergo.configure(rf)

	mod_ergo.runModule(rf)

	mod_ergo.close()
	yarp.Network.fini()