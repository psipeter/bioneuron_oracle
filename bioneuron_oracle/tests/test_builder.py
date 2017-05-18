import nengo

import numpy as np

from bioneuron_oracle import BahlNeuron, TrainedSolver, BioSolver


def test_repeat_neuron_type(Simulator):
    neuron_type = BahlNeuron()

    with nengo.Network() as model:
        bio1 = nengo.Ensemble(50, 1, neuron_type=neuron_type)
        bio2 = nengo.Ensemble(20, 1, neuron_type=neuron_type)
        assert bio1.n_neurons != bio2.n_neurons

    with Simulator(model) as sim:
        assert len(sim.data[bio1.neurons]) == bio1.n_neurons
        assert len(sim.data[bio2.neurons]) == bio2.n_neurons
        sim.step()  # sanity check

def test_repeat_build(Simulator):
    neuron_type = BahlNeuron()

    with nengo.Network() as model:
        lif1 = nengo.Ensemble(30, 1, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(40, 1, neuron_type=nengo.LIF())
        bio1 = nengo.Ensemble(50, 1, neuron_type=neuron_type)
        bio2 = nengo.Ensemble(20, 1, neuron_type=neuron_type)
        solver1 = TrainedSolver(weights_bio = np.zeros((50, 30, 1)))
        solver2 = BioSolver(decoders_bio = np.zeros((20, 1)))
        nengo.Connection(lif1, bio1, trained_weights=True, solver=solver1)
        nengo.Connection(bio2, lif2, solver=solver2)

    with Simulator(model) as sim:
        sim.step()
    with Simulator(model) as sim:
        sim.step()
    assert True