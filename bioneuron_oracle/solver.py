import numpy as np

from nengo.solvers import Solver
from nengo.exceptions import BuildError

__all__ = ['TrainedSolver', 'BioSolver']


class TrainedSolver(Solver):
    """Wraps weights going into bioneuron ensembles."""

    def __init__(self, weights_bio):
        weights_bio = np.asarray(weights_bio)
        if weights_bio.ndim != 3:
            raise BuildError("Shape mismatch: weight matrix for trained\
            				  connections must have ndim=3, got ndim=%s"
                             % weights_bio.ndim)
        self.weights_bio = weights_bio
        super(TrainedSolver, self).__init__()

    def __call__(self, A, Y, rng=None, E=None):
        # return self.weights_bio, {}
        return np.zeros((self.weights_bio.shape[0], self.weights_bio.shape[1])),\
        	{'weights_bio': self.weights_bio}


class BioSolver(Solver):
    """Wraps decoders coming from bioneuron ensembles."""

    def __init__(self, decoders_bio):
        decoders_bio = np.asarray(decoders_bio)
        if decoders_bio.ndim <= 1:
            decoders_bio = decoders_bio[:, None]
        self.decoders_bio = decoders_bio
        super(BioSolver, self).__init__()

    def __call__(self, A, Y, rng=None, E=None):
        return self.decoders_bio, {}
