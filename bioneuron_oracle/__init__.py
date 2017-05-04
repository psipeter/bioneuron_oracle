"""
Bioneuron Oracle

Interface NEURON with nengo for simulating biologically realistic
neuron models, using least-squares solvers and the oracle
to compute decoders for bioneurons
"""

from .BioConnection import BioConnection
from .monkeypatch import patch_connections, unpatch_connections
from BahlNeuron import *
from builder import *
from signals import *
import pytest

patch_connections()
