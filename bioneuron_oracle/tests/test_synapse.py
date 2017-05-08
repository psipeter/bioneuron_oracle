import numpy as np
from seaborn import set_palette, color_palette

import neuron

import nengo

from bioneuron_oracle import BahlNeuron


def test_synapse_g(Simulator, plt):
    """
    Run the model for t_transient so that
    bioneuron voltage settles to equilibrium
    hardcoded signal shutoff with first prespike
    at t=0.009 past t_transient
    """

    pre_neurons = 1
    bio_neurons = 1
    dt_nengo = 0.000025
    pre_seed = 3
    bio_seed = 10
    nengo_seeds = 33
    gain = [3.29141359]  # associated with bio_seed = 10
    t_transient = 0.2
    t_final = 0.25
    dim = 1

    plt.subplot(111)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('voltage')
    # plt.xlim((t_transient, t_final))

    """tiny tau"""
    tau_nengo_small = 0.001
    tau_neuron_small = 0.001
    tau_nengo_medium = 0.01
    tau_neuron_medium = 0.01
    tau_nengo_large = 0.1
    tau_neuron_large = 0.1

    with nengo.Network(seed=nengo_seeds) as model:
        stim = nengo.Node(
            lambda t: 0.75 * (t_transient < t < (t_transient + 0.009)))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                             seed=pre_seed, neuron_type=nengo.LIF())
        bio_small = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   seed=bio_seed, neuron_type=BahlNeuron())
        bio_medium = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                    seed=bio_seed, neuron_type=BahlNeuron())
        bio_large = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   seed=bio_seed, neuron_type=BahlNeuron())
        lif_small = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   neuron_type=nengo.LIF(), seed=bio_seed,
                                   gain=gain, bias=[0])
        lif_medium = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                    neuron_type=nengo.LIF(), seed=bio_seed,
                                    gain=gain, bias=[0])
        lif_large = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   neuron_type=nengo.LIF(), seed=bio_seed,
                                   gain=gain, bias=[0])

        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, bio_small, synapse=tau_neuron_small,
                         weights_bias_conn=False)
        nengo.Connection(pre, lif_small, synapse=tau_nengo_small)
        nengo.Connection(pre, bio_medium, synapse=tau_neuron_medium,
                         weights_bias_conn=False)
        nengo.Connection(pre, lif_medium, synapse=tau_nengo_medium)
        nengo.Connection(pre, bio_large, synapse=tau_neuron_large,
                         weights_bias_conn=False)
        nengo.Connection(pre, lif_large, synapse=tau_nengo_large)

        probe_pre_spikes = nengo.Probe(pre.neurons, 'spikes')

    with Simulator(model, dt=dt_nengo, seed=nengo_seeds) as sim:
        # NOTE: syn.get_loc() does not contain section information
        bioneuron_small = sim.data[bio_small.neurons][0]
        bioneuron_small.v_syn_record = neuron.h.Vector()
        bioneuron_small.g_syn_record = neuron.h.Vector()
        location = bioneuron_small.synapses[pre][0][0].syn.get_loc()
        bioneuron_small.v_syn_record.record(
            bioneuron_small.cell.apical(location)._ref_v)
        bioneuron_small.g_syn_record.record(
            bioneuron_small.synapses[pre][0][0].syn._ref_g)

        bioneuron_medium = sim.data[bio_medium.neurons][0]
        bioneuron_medium.v_syn_record = neuron.h.Vector()
        bioneuron_medium.g_syn_record = neuron.h.Vector()
        location = bioneuron_medium.synapses[pre][0][0].syn.get_loc()
        bioneuron_medium.v_syn_record.record(
            bioneuron_medium.cell.apical(location)._ref_v)
        bioneuron_medium.g_syn_record.record(
            bioneuron_medium.synapses[pre][0][0].syn._ref_g)

        bioneuron_large = sim.data[bio_large.neurons][0]
        bioneuron_large.v_syn_record = neuron.h.Vector()
        bioneuron_large.g_syn_record = neuron.h.Vector()
        location = bioneuron_large.synapses[pre][0][0].syn.get_loc()
        bioneuron_large.v_syn_record.record(
            bioneuron_large.cell.apical(location)._ref_v)
        bioneuron_large.g_syn_record.record(
            bioneuron_large.synapses[pre][0][0].syn._ref_g)

        neuron.init()
        sim.run(t_final)

    time_lif = sim.trange()
    time_bio = np.array(bioneuron_small.t_record)

    g_bio_small = np.array(bioneuron_small.g_syn_record)
    g_bio_medium = np.array(bioneuron_medium.g_syn_record)
    g_bio_large = np.array(bioneuron_large.g_syn_record)
    w_small = bioneuron_small.synapses[pre][0][0].weight
    w_medium = bioneuron_medium.synapses[pre][0][0].weight
    w_large = bioneuron_large.synapses[pre][0][0].weight

    plt.plot(time_bio / 1000, g_bio_small,
             label='g_ExpSyn, tau=%s' % tau_neuron_small)
    plt.plot(time_bio / 1000, g_bio_medium,
             label='g_ExpSyn, tau=%s' % tau_neuron_medium)
    plt.plot(time_bio / 1000, g_bio_large,
             label='g_ExpSyn, tau=%s' % tau_neuron_large)

    h_small = np.exp(-time_lif/tau_nengo_small)
    g_lif_manual_small = np.convolve(
        w_small * sim.data[probe_pre_spikes][:, 0] * dt_nengo,
        h_small, mode='same')
    # h_small = h_small/np.linalg.norm(h_small,1)
    h_medium = np.exp(-time_lif/tau_nengo_medium)
    g_lif_manual_medium = np.convolve(
        w_medium * sim.data[probe_pre_spikes][:, 0] * dt_nengo,
        h_medium, mode='same')
    # h_medium = h_medium/np.linalg.norm(h_medium,1)
    h_large = np.exp(-time_lif/tau_nengo_large)
    g_lif_manual_large = np.convolve(
        w_large * sim.data[probe_pre_spikes][:, 0] * dt_nengo,
        h_large, mode='same')
    # h_large = h_large/np.linalg.norm(h_large,1)

    plt.plot(time_lif, g_lif_manual_small,
             label='g_exp, tau=%s' % tau_nengo_small)
    plt.plot(time_lif, g_lif_manual_medium,
             label='g_exp, tau=%s' % tau_nengo_medium)
    plt.plot(time_lif, g_lif_manual_large,
             label='g_exp, tau=%s' % tau_nengo_large)
    plt.legend()

    assert True  # always passes, just for testing NEURON compatibility


def test_synapse_tau(Simulator, plt):
    """
    Run the model for t_transient so that
    bioneuron voltage settles to equilibrium
    hardcoded signal shutoff with first prespike
    at t=0.009 past t_transient
    """

    pre_neurons = 1
    bio_neurons = 1
    dt_nengo = 0.000025
    pre_seed = 3
    bio_seed = 10
    nengo_seeds = 33
    gain = [3.29141359]  # associated with bio_seed = 10
    t_transient = 0.2
    t_final = 0.5
    dim = 1

    # current_palette = color_palette(n_colors=20)
    set_palette(color_palette("Set1", 9))
    plt.subplot(111)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('voltage')
    plt.xlim((t_transient, t_final))

    """tiny tau"""
    tau_nengo_small = 0.001
    tau_neuron_small = 0.001
    tau_nengo_medium = 0.01
    tau_neuron_medium = 0.01
    tau_nengo_large = 0.1
    tau_neuron_large = 0.1

    with nengo.Network(seed=nengo_seeds) as model:
        stim = nengo.Node(
            lambda t: 0.75 * (t_transient < t < (t_transient + 0.009)))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                             seed=pre_seed, neuron_type=nengo.LIF())
        bio_small = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   seed=bio_seed, neuron_type=BahlNeuron())
        bio_medium = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                    seed=bio_seed, neuron_type=BahlNeuron())
        bio_large = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   seed=bio_seed, neuron_type=BahlNeuron())
        lif_small = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   neuron_type=nengo.LIF(), seed=bio_seed,
                                   gain=gain, bias=[0])
        lif_medium = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                    neuron_type=nengo.LIF(), seed=bio_seed,
                                    gain=gain, bias=[0])
        lif_large = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                   neuron_type=nengo.LIF(), seed=bio_seed,
                                   gain=gain, bias=[0])

        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, bio_small, synapse=tau_neuron_small,
                         weights_bias_conn=False)
        nengo.Connection(pre, lif_small, synapse=tau_nengo_small)
        nengo.Connection(pre, bio_medium, synapse=tau_neuron_medium,
                         weights_bias_conn=False)
        nengo.Connection(pre, lif_medium, synapse=tau_nengo_medium)
        nengo.Connection(pre, bio_large, synapse=tau_neuron_large,
                         weights_bias_conn=False)
        nengo.Connection(pre, lif_large, synapse=tau_nengo_large)

        probe_lif_small_voltage = nengo.Probe(lif_small.neurons, 'voltage')
        probe_lif_medium_voltage = nengo.Probe(lif_medium.neurons, 'voltage')
        probe_lif_large_voltage = nengo.Probe(lif_large.neurons, 'voltage')

    with Simulator(model, dt=dt_nengo, seed=nengo_seeds) as sim:
        # NOTE: syn.get_loc() does not contain section information
        bioneuron_small = sim.data[bio_small.neurons][0]
        bioneuron_small.v_syn_record = neuron.h.Vector()
        bioneuron_small.g_syn_record = neuron.h.Vector()
        location = bioneuron_small.synapses[pre][0][0].syn.get_loc()
        bioneuron_small.v_syn_record.record(
            bioneuron_small.cell.apical(location)._ref_v)
        bioneuron_small.g_syn_record.record(
            bioneuron_small.synapses[pre][0][0].syn._ref_g)

        bioneuron_medium = sim.data[bio_medium.neurons][0]
        bioneuron_medium.v_syn_record = neuron.h.Vector()
        bioneuron_medium.g_syn_record = neuron.h.Vector()
        location = bioneuron_medium.synapses[pre][0][0].syn.get_loc()
        bioneuron_medium.v_syn_record.record(
            bioneuron_medium.cell.apical(location)._ref_v)
        bioneuron_medium.g_syn_record.record(
            bioneuron_medium.synapses[pre][0][0].syn._ref_g)

        bioneuron_large = sim.data[bio_large.neurons][0]
        bioneuron_large.v_syn_record = neuron.h.Vector()
        bioneuron_large.g_syn_record = neuron.h.Vector()
        location = bioneuron_large.synapses[pre][0][0].syn.get_loc()
        bioneuron_large.v_syn_record.record(
            bioneuron_large.cell.apical(location)._ref_v)
        bioneuron_large.g_syn_record.record(
            bioneuron_large.synapses[pre][0][0].syn._ref_g)

        neuron.init()
        sim.run(t_final)

    time_lif = sim.trange()
    time_bio = np.array(bioneuron_small.t_record)
    # unscale LIF voltage: 50=delta V b/w rest, spike; -70=rest
    voltage_lif_small = 50 * sim.data[probe_lif_small_voltage] - 70
    voltage_lif_medium = 50 * sim.data[probe_lif_medium_voltage] - 70
    voltage_lif_large = 50 * sim.data[probe_lif_large_voltage] - 70
    voltage_bio_small = np.array(bioneuron_small.v_record)
    voltage_bio_medium = np.array(bioneuron_medium.v_record)
    voltage_bio_large = np.array(bioneuron_large.v_record)
    voltage_bio_syn_small = np.array(bioneuron_small.v_syn_record)
    voltage_bio_syn_medium = np.array(bioneuron_medium.v_syn_record)
    voltage_bio_syn_large = np.array(bioneuron_large.v_syn_record)

    plt.plot(time_lif, voltage_lif_small,
             label='lif w/ lowpass, tau=%s' % tau_nengo_small)
    plt.plot(time_bio / 1000, voltage_bio_small,
             label='bio @ soma, tau=%s' % tau_neuron_small)
    plt.plot(time_bio / 1000, voltage_bio_syn_small,
             label='bio @ synapse, tau=%s' % tau_neuron_small)
    plt.plot(time_lif, voltage_lif_medium,
             label='lif w/ lowpass, tau=%s' % tau_nengo_medium)
    plt.plot(time_bio / 1000, voltage_bio_medium,
             label='bio @ soma, tau=%s' % tau_neuron_medium)
    plt.plot(time_bio / 1000, voltage_bio_syn_medium,
             label='bio @ synapse, tau=%s' % tau_neuron_medium)
    plt.plot(time_lif, voltage_lif_large,
             label='lif w/ lowpass, tau=%s' % tau_nengo_large)
    plt.plot(time_bio / 1000, voltage_bio_large,
             label='bio @ soma, tau=%s' % tau_neuron_large)
    plt.plot(time_bio / 1000, voltage_bio_syn_large,
             label='bio @ synapse, tau=%s' % tau_neuron_large)
    plt.legend()

    assert True  # always passes, just for testing NEURON compatibility
