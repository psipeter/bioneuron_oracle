from functools32 import lru_cache

import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot

from bioneuron_oracle import BahlNeuron, prime_sinusoids, step_input

pre_neurons = 50
bio_neurons = 20
tau_nengo = 0.01
tau_neuron = 0.01
dt_nengo = 0.001
dt_neuron = 0.0001
pre_seed = 3
bio_seed = 6
t_final = 1.0
dim = 2
n_syn = 10
signal = 'prime_sinusoids'
decoders_bio = None

with nengo.Network() as model:
    """
    Simulate a feedforward network [stim]-[LIF]-[BIO]
    and compare to [stim]-[LIF]-[LIF].
    """

    if signal == 'prime_sinusoids':
        stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
    elif signal == 'step_input':
        stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))

    pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                         seed=pre_seed, neuron_type=nengo.LIF())
    bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                         seed=bio_seed, neuron_type=BahlNeuron())
    lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                         neuron_type=nengo.LIF(), seed=bio_seed)
    direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                            neuron_type=nengo.Direct())

    nengo.Connection(stim, pre, synapse=None)
    nengo.Connection(pre, bio, synapse=tau_neuron, weights_bias_conn=True)
    nengo.Connection(pre, lif, synapse=tau_nengo)
    nengo.Connection(stim, direct, synapse=tau_nengo)

    probe_stim = nengo.Probe(stim, synapse=None)
    probe_pre = nengo.Probe(pre, synapse=tau_nengo)
    probe_lif = nengo.Probe(lif, synapse=tau_nengo)
    probe_direct = nengo.Probe(direct, synapse=tau_nengo)
    probe_pre_spikes = nengo.Probe(pre.neurons, 'spikes')
    probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
    probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
    probe_lif_voltage = nengo.Probe(lif.neurons, 'voltage')


@lru_cache(maxsize=None)
def sim_feedforward():
    sim = nengo.Simulator(model, dt=dt_nengo)
    with sim:
        sim.run(t_final)
    return sim


def test_feedforward_connection(plt):
    """
    spike raster for PRE, BIO and comparison LIF ensembles
    test passes if:
        - a certain number of bioneurons fire at least once
        - this number should be within $cutoff$ %%
          of the number of active LIFs
    """
    sim = sim_feedforward()
    plt.subplot(3, 1, 1)
    rasterplot(sim.trange(), sim.data[probe_pre_spikes], use_eventplot=True)
    plt.ylabel('PRE')
    plt.title('spike raster')
    plt.yticks([])
    plt.subplot(3, 1, 2)
    rasterplot(sim.trange(), sim.data[probe_bio_spikes], use_eventplot=True)
    plt.ylabel('BIO')
    plt.yticks([])
    plt.subplot(3, 1, 3)
    rasterplot(sim.trange(), sim.data[probe_lif_spikes], use_eventplot=True)
    plt.xlabel('time (s)')
    plt.ylabel('LIF')
    plt.yticks([])
    cutoff = 0.5
    n_bio = len(np.nonzero(
        np.sum(sim.data[probe_bio_spikes], axis=0))[0])  # n_active
    n_lif = len(np.nonzero(
        np.sum(sim.data[probe_lif_spikes], axis=0))[0])  # n_active
    assert (1 - cutoff) * n_lif < n_bio
    assert n_bio < (1 + cutoff) * n_lif


def test_voltage(plt):
    """
    test passes if:
        - a neuron spends less than $cutoff_sat$ %% of time
          in the 'saturated' voltage regime -40<V<-20,
          which would indicate synaptic inputs are overstimulating
        - less that $cutoff_eq$ %% of time near equilibrium, -67<V<-63
          which would indicate that synaptic inputs are understimulating
          (or not being delivered)
    """
    sim = sim_feedforward()
    cutoff_sat = 0.3
    cutoff_eq = 0.5
    for bioneuron in sim.data[bio.neurons]:
        V = np.array(bioneuron.v_record)
        time = np.array(bioneuron.t_record)
        t_saturated = len(np.where((-40.0 < V) & (V < -20.0))[0])
        t_equilibrium = len(np.where((-70.0 < V) & (V < -69.0))[0])
        t_total = len(time)
        f_sat = 1. * t_saturated / t_total
        f_eq = 1. * t_equilibrium / t_total
        if (f_sat < cutoff_sat or f_eq < cutoff_eq):
            plt.subplot(111)
            plt.plot(time, V, label='saturated=%s, equilibrium=%s' %
                                    (f_sat, f_eq))
            plt.xlabel('time (ms)')
            plt.ylabel('voltage (mV)')
            plt.title('voltage')
        assert f_sat < cutoff_sat
        assert f_eq < cutoff_eq


def test_feedforward_decode(plt):
    """
    decoded output of bioensemble
    test passes if:
        - rmse_bio is within $cutoff$ %% of rmse_lif
    """
    sim = sim_feedforward()
    cutoff = 0.5
    plt.subplot(1, 1, 1)
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)
    rmse_bio = np.sqrt(np.average((
        sim.data[probe_direct] - xhat_bio)**2))
    rmse_lif = np.sqrt(np.average((
        sim.data[probe_direct] - sim.data[probe_lif])**2))
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_lif],
             label='lif, rmse=%.5f' % rmse_lif)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert (1 - cutoff) * rmse_lif < rmse_bio
    assert rmse_bio < (1 + cutoff) * rmse_lif
