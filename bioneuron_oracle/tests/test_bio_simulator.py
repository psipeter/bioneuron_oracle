import pytest

import nengo

from bioneuron_oracle import BahlNeuron


def test_simulator_cleanup(Simulator):
    with nengo.Network() as model:
        bio = nengo.Ensemble(10, 1, neuron_type=BahlNeuron())

    with Simulator(model) as sim:
        assert not any(bahl._clean for bahl in sim.data[bio.neurons])
    assert all(bahl._clean for bahl in sim.data[bio.neurons])

    with pytest.raises(RuntimeError):
        sim.data[bio.neurons][0].update()

    with pytest.raises(RuntimeError):
        sim.data[bio.neurons][0].cleanup()
