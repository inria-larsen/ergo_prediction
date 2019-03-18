# Author: Adrien Malaisé
# email: adrien.malaise@inria.fr

import yarp
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ergonomic_assessment/src/')))

from ErgoAssessment import ErgoAssessment
from HumanPosture import HumanPosture

class ErgonomicAssessmentModule(yarp.RFModule):
	"""This module compute online the ergonomic scores of postures 
	related to worksheets from industry

	Example:
		>>> python online_assessment.py --from ergo_config.ini

	Attributes:
		in_port_posture (BufferedPortBottle): Input port. It is used to read
			the posture data.

		list_out_port_ergo (Dict{BufferedPortBottle}): List of output ports. 
			They send the ergonomic value of each measure included in the 
			configuration file.

	"""
	def __init__(self):
		yarp.RFModule.__init__(self)
		self.handlerPort = yarp.Port()

	def configure(self, rf):
		"""Configure the module according to the configuration file

		Args: 
			rf (string): name of the configuration file of the module. 
				This file must be include in the folder pointing to the yarp
				contexts. By default: rf = 'default.ini'
		"""

		self.handlerPort.open("/ErgonomicAssessmentModule")
		self.attach(self.handlerPort)

		self.in_port_posture = yarp.BufferedPortBottle()
		self.in_port_posture.open('/ergo_pred/posture:i')

		path_config_posture = rf.find('posture_config').toString()
		self._posture = HumanPosture(path_config_posture)
		
		path_config_ergo = rf.find('ergo_config').toString()
		self._ergo_assessment = ErgoAssessment(path_config_ergo)

		self.list_out_port_ergo = dict()

		for ergo_score in self._ergo_assessment.get_list_score():
			self.list_out_port_ergo[ergo_score] = yarp.BufferedPortBottle()
			self.list_out_port_ergo[ergo_score].open("/ergo_pred/" + ergo_score.lower() + ':o')

		return True

	def close(self):
		"""Close all the ports used by the module
		"""
		self.in_port_posture.close()
		for ergo_score in self.list_out_port_ergo:
			self.list_out_port_ergo[ergo_score].close()
		self.handlerPort.close()
		return True

	def updateModule(self):
		"""Read the data of posture and send the updated ergonomic scores
		"""
		b_in = self.in_port_posture.read(True)

		data = b_in.toString().split(' ')
		value = list(map(float, data))

		self._posture.update_posture(value)
		self._ergo_assessment.compute_ergo_scores(self._posture)

		for ergo_score in self.list_out_port_ergo:
			b_out = self.list_out_port_ergo[ergo_score].prepare()
			b_out.clear()
			b_out.addDouble(self._ergo_assessment[ergo_score])
			self.list_out_port_ergo[ergo_score].write()

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