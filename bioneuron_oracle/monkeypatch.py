import logging

import nengo
from nengo import Connection as NengoConnection

from bioneuron_oracle.bio_connection import BioConnection

__all__ = ['patch_connections', 'unpatch_connections']


def patch_connections():
    """Monkeypatches bioneuron connections into Nengo"""
    logging.info("Monkeypatching BioConnection")
    nengo.Connection = BioConnection


def unpatch_connections():
    logging.info("Unpatching BioConnection")
    nengo.Connection = NengoConnection
