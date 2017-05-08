import nengo

from bioneuron_oracle import patch, unpatch

from bioneuron_oracle.bio_connection import BioConnection
from bioneuron_oracle.bio_simulator import BioSimulator
from bioneuron_oracle.monkeypatch import NengoConnection
from bioneuron_oracle.monkeypatch import NengoSimulator


def test_monkepatching():
    assert BioConnection is not NengoConnection
    assert BioSimulator is not NengoSimulator

    assert nengo.Connection is BioConnection
    assert nengo.Simulator is BioSimulator

    unpatch()
    assert nengo.Connection is NengoConnection
    assert nengo.Simulator is NengoSimulator

    patch()
    assert nengo.Connection is BioConnection
    assert nengo.Simulator is BioSimulator
