from nengo.exceptions import ValidationError
from nengo.solvers import Solver


class BioSolver(Solver):

    def __init__(self, decoders_bio=None):
        self.decoders_bio = decoders_bio
        super(BioSolver, self).__init__()

    def __call__(self, A, Y, rng=None, E=None):
        if self.decoders_bio is not None:
            return self.decoders_bio, {}
        else:
            raise ValidationError(
                "decoders_bio not set in model definition",
                self.decoders_bio)
