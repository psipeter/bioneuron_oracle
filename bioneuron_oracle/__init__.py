"""
Bioneuron Oracle

Interface NEURON with nengo for simulating biologically realistic
neuron models, using least-squares solvers and the oracle
to compute decoders for bioneurons
"""

from .bahl_neuron import *  # loads NEURON model
from .bio_connection import *
from .builder import *  # executes Builder.register methods
from .monkeypatch import *
from .signals import *
from .solver import *

patch()  # nengo.Connection <-> BioConnection, Simulator cleanup
