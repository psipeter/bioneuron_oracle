import logging
import warnings

import nengo
from nengo import Connection as NengoConnection
from nengo import Simulator as NengoSimulator

from bioneuron_oracle.bio_connection import BioConnection
from bioneuron_oracle.bio_simulator import BioSimulator

__all__ = ['patch', 'unpatch']


def patch():
    """Monkeypatches bioneuron connections into Nengo"""
    logging.info("Monkeypatching Bio Connection / Simulator")

    if nengo.Connection is not BioConnection:
        nengo.Connection = BioConnection
    else:
        warnings.warn("BioConnection already patched", UserWarning)

    if nengo.Simulator is not BioSimulator:
        nengo.Simulator = BioSimulator
    else:
        warnings.warn("BioSimulator already patched", UserWarning)


def unpatch():
    logging.info("Unpatching Bio Connection / Simulator")

    if nengo.Connection is BioConnection:
        nengo.Connection = NengoConnection
    else:
        warnings.warn("BioConnection is not patched", UserWarning)

    if nengo.Simulator is BioSimulator:
        nengo.Simulator = NengoSimulator
    else:
        warnings.warn("BioSimulator is not patched", UserWarning)
