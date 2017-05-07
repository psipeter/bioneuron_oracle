import os

import numpy as np

import neuron

from nengo.neurons import NeuronType

# Load NEURON model (TODO: installation instructions)
neuron.h.load_file(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "NEURON_models/bahl.hoc")))

__all__ = ['BahlNeuron']


class BahlNeuron(NeuronType):
    """Compartmental neuron from Bahl et al 2012."""

    probeable = ('spikes', 'voltage')

    def __init__(self):
        super(BahlNeuron, self).__init__()

    def rates(self, x, gain, bias):
        return x

    def gain_bias(self, max_rates, intercepts):
        return np.ones(len(max_rates)), np.ones(len(max_rates))

    def step_math(self, dt, spiked, neurons, voltage, time):
        """
        Run NEURON forward one nengo timestep.
        Compare the current and previous spike arrays for this bioneuron.
        If they're different, the neuron has spiked.
        """
        neuron.run(time*1000)
        for i, nrn in enumerate(neurons):
            count = len(nrn.spikes) - nrn.num_spikes_last
            nrn.num_spikes_last = len(nrn.spikes)
            volt = np.asarray(nrn.v_record)[-1]  # first call neuron.init()
            spiked[i] = count / dt
            voltage[i] = volt
