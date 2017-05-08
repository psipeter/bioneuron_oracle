import numpy as np

from nengo.solvers import Solver

__all__ = ['BioSolver']


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
