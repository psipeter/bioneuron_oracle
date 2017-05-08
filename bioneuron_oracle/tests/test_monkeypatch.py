import nengo
from nengo.utils.testing import warns

from bioneuron_oracle import patch, unpatch, BioConnection, BioSimulator
from bioneuron_oracle.monkeypatch import NengoConnection
from bioneuron_oracle.monkeypatch import NengoSimulator


def test_monkeypatching():
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


def test_repeated_monkeypatching():
    unpatch()
    with warns(UserWarning):
        unpatch()

    assert nengo.Connection is NengoConnection
    assert nengo.Simulator is NengoSimulator

    patch()
    with warns(UserWarning):
        patch()

    assert nengo.Connection is BioConnection
    assert nengo.Simulator is BioSimulator
