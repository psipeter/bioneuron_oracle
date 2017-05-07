import nengo

from bioneuron_oracle import BahlNeuron


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
