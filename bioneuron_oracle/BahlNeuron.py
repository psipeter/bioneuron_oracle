import nengo
import numpy as np
import neuron
import os

# EDIT this to point to your local repository for Jupyter notebook
# neuron.h.load_file('/home/pduggins/bioneuron_oracle/NEURON_models/bahl.hoc')
# For the library
neuron.h.load_file(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "NEURON_models/bahl.hoc")))


class BahlNeuron(nengo.neurons.NeuronType):
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
            volt = np.array(nrn.v_record)[-1]  # first call neuron.init()
            nrn.num_spikes_last = len(nrn.spikes)
            spiked[i] = count / dt
            voltage[i] = volt


class Bahl(object):
    def __init__(self):
        super(Bahl, self).__init__()
        self.synapses = {}
        self.cell = neuron.h.Bahl()
        self.v_record = neuron.h.Vector()
        self.v_record.record(self.cell.soma(0.5)._ref_v)
        self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
        self.t_record = neuron.h.Vector()
        self.t_record.record(neuron.h._ref_t)
        self.spikes = neuron.h.Vector()
        self.ap_counter.record(neuron.h.ref(self.spikes))
        self.num_spikes_last = 0


class ExpSyn(object):
    """
    Conductance-based synapses.
    There are two types, excitatory and inhibitory,
    with different reversal potentials.
    If the synaptic weight is above zero, initialize an excitatory synapse,
    else initialize an inhibitory syanpse with abs(weight).
    """

    def __init__(self, sec, weight, tau, e_exc=0.0, e_inh=-80.0):
        self.tau = tau
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.ExpSyn(sec)
        self.syn.tau = 1000*self.tau  # no more 2x multiply
        self.weight = weight
        if self.weight >= 0.0:
            self.syn.e = self.e_exc
        else:
            self.syn.e = self.e_inh
        # time of spike arrival assigned in nengo step
        self.spike_in = neuron.h.NetCon(None, self.syn)
        self.spike_in.weight[0] = abs(self.weight)
