import nengo
import bioneuron_oracle
from nengo import Connection as NengoConnection
from BioConnection import BioConnection


def patch_connections(connection=True):
    """Monkeypatches bioneuron connections into Nengo"""
    if connection:
        nengo.Connection = BioConnection


def unpatch_connections():
    nengo.Connection = NengoConnection
