# Author: Adrien Malaisé
# email: adrien.malaise@inria.fr

import yarp
import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ergonomic_assessment/src/')))

import AE

class OnlineAE(yarp.RFModule):
	
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

		self.handlerPort.open("/AE_Module")
		self.attach(self.handlerPort)

		self.in_port_posture = yarp.BufferedPortBottle()
		self.in_port_posture.open('/AE/posture:i')

		path_AE = rf.find('path_AE').toString()

		with open(path_AE, 'rb') as input:
			self._autoencoder = pickle.load(input)
		
		self.out_port_reconstruct = yarp.BufferedPortBottle()
		self.out_port_reconstruct.open("/AE/reconstruct:o")

		self.out_port_latent = yarp.BufferedPortBottle()
		self.out_port_latent.open("/AE/latent:o")

		return True

	def close(self):
		"""Close all the ports used by the module
		"""
		self.in_port_posture.close()
		self.out_port_reconstruct.close()
		self.handlerPort.close()
		return True

	def updateModule(self):
		"""Read the data of posture and send the updated ergonomic scores
		"""
		b_in = self.in_port_posture.read(True)

		input_data = b_in.toString().split(' ')
		input_value = list(map(float, input_data))

		del input_value[0]

		input_value = np.deg2rad(input_value)

		data_output, encoded_data, score = self._autoencoder.test_model(input_value)

		b_out = self.out_port_reconstruct.prepare()
		b_out.clear()
		b_out.addInt(len(data_output))
		for data in data_output[0]:
			b_out.addDouble(np.rad2deg(data.astype(np.float64)))
		self.out_port_reconstruct.write()

		b_out = self.out_port_latent.prepare()
		b_out.clear()

		b_out.addInt(len(encoded_data))
		for data in encoded_data[0]:
			b_out.addDouble(data.astype(np.float64))
		self.out_port_latent.write()

		return True

	def getPeriod(self):
		return 0.001
		
if __name__=="__main__":
	yarp.Network.init()

	rf = yarp.ResourceFinder()
	rf.setVerbose(True)
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)



	mod_AE = OnlineAE()
	mod_AE.configure(rf)


	mod_AE.runModule(rf)

	mod_AE.close()
	yarp.Network.fini()