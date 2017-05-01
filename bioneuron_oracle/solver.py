import nengo
import numpy as np
import neuron

class BioSolver(nengo.solvers.Solver):
	def __init__(self, decoders_bio=None):
		self.decoders_bio=decoders_bio
		self.solver=nengo.solvers.LstsqL2()
		super(BioSolver, self).__init__()

	def __call__(self,A,Y,rng=None,E=None):
		if self.decoders_bio is not None:
			return self.decoders_bio, dict()
		else:
			raise nengo.exceptions.ValidationError(
				"decoders not set in model definition",
				self.decoders_bio)
			return
