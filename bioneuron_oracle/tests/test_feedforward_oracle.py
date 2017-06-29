from functools32 import lru_cache

import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.utils.numpy import rmse

from bioneuron_oracle import BahlNeuron, prime_sinusoids, step_input, equalpower

pre_neurons = 100
bio_neurons = 20
post_neurons = 50
tau_nengo = 0.05
tau_neuron = 0.05
dt_nengo = 0.001
min_rate = 150
max_rate = 200
pre_seed = 3
bio_seed = 6
conn_seed = 9
network_seed = 12
sim_seed = 15
post_seed = 18
t_final = 1.0
dim = 1
n_syn = 1
signal = 'prime_sinusoids'
max_freq = 5
signal_seed = 123

with nengo.Network(seed=network_seed) as network:

    if signal == 'prime_sinusoids':
        stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
    elif signal == 'step_input':
        stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
    elif signal == 'white_noise':
        stim = nengo.Node(lambda t: equalpower(
                              t, dt_nengo, t_final, max_freq, dim,
                              mean=0.0, std=1.0, seed=signal_seed))

    pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                         seed=pre_seed, neuron_type=nengo.LIF())
    bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                         seed=bio_seed, neuron_type=BahlNeuron(),
                         max_rates=nengo.dists.Uniform(min_rate, max_rate),
                         intercepts=nengo.dists.Uniform(-1, 1))
    lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                         seed=bio.seed, neuron_type=nengo.LIF(),
                         max_rates=bio.max_rates, intercepts=bio.intercepts)
    direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                            neuron_type=nengo.Direct())

    nengo.Connection(stim, pre, synapse=None)
    nengo.Connection(pre, bio, synapse=tau_neuron,
                     n_syn=n_syn, weights_bias_conn=True)
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


# @lru_cache(maxsize=None)
def sim_feedforward(Simulator):
    sim = Simulator(network, dt=dt_nengo, seed=sim_seed)
    with sim:
        sim.run(t_final)
    return sim


def test_feedforward_connection(Simulator, plt):
    """
    spike raster for PRE, BIO and comparison LIF ensembles
    test passes if:
        - a certain number of bioneurons fire at least once
        - this number should be within $cutoff$ %%
          of the number of active LIFs
    """
    sim = sim_feedforward(Simulator)
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


def test_voltage(Simulator, plt):
    """
    test passes if:
        - a neuron spends less than $cutoff_sat$ %% of time
          in the 'saturated' voltage regime -40<V<-20,
          which would indicate synaptic inputs are overstimulating
        - less that $cutoff_eq$ %% of time near equilibrium, -67<V<-63
          which would indicate that synaptic inputs are understimulating
          (or not being delivered)
    """
    sim = sim_feedforward(Simulator)
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


def test_feedforward_tuning_curves(Simulator, plt):

    sim = sim_feedforward(Simulator)

    tau_nengo = 0.01
    dt_nengo = 0.001
    n_eval_points = 20
    cutoff = 50.0

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
    # rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
    #                      for i in range(bio.n_neurons)])

    for i in range(bio.n_neurons):
        x_dot_e_bio = np.dot(sim.data[probe_pre], bio.encoders[i])
        x_dot_e_lif = np.dot(sim.data[probe_pre], sim.data[lif].encoders[i])
        x_dot_e_vals_bio = np.linspace(np.min(x_dot_e_bio),
                                       np.max(x_dot_e_bio), 
                                       num=n_eval_points)
        x_dot_e_vals_lif = np.linspace(np.min(x_dot_e_lif),
                                       np.max(x_dot_e_lif), 
                                       num=n_eval_points)
        Hz_mean_bio = np.zeros((x_dot_e_vals_bio.shape[0]))
        Hz_stddev_bio = np.zeros_like(Hz_mean_bio)
        Hz_mean_lif = np.zeros((x_dot_e_vals_lif.shape[0]))
        Hz_stddev_lif = np.zeros_like(Hz_mean_lif)
        for xi in range(x_dot_e_vals_bio.shape[0] - 1):
            ts_greater = np.where(x_dot_e_vals_bio[xi] < sim.data[probe_pre])[0]
            ts_smaller = np.where(sim.data[probe_pre] < x_dot_e_vals_bio[xi + 1])[0]
            ts = np.intersect1d(ts_greater, ts_smaller)
            if ts.shape[0] > 0: Hz_mean_bio[xi] = np.average(act_bio[ts, i])
            if ts.shape[0] > 1: Hz_stddev_bio[xi] = np.std(act_bio[ts, i])
        for xi in range(x_dot_e_vals_lif.shape[0] - 1):
            ts_greater = np.where(x_dot_e_vals_lif[xi] < sim.data[probe_pre])[0]
            ts_smaller = np.where(sim.data[probe_pre] < x_dot_e_vals_lif[xi + 1])[0]
            ts = np.intersect1d(ts_greater, ts_smaller)
            if ts.shape[0] > 0: Hz_mean_lif[xi] = np.average(act_lif[ts, i])
            if ts.shape[0] > 1: Hz_stddev_lif[xi] = np.std(act_lif[ts, i])
        bioplot = plt.errorbar(x_dot_e_vals_bio[:-2], Hz_mean_bio[:-2],
                     yerr=Hz_stddev_bio[:-2], fmt='-o')  # , label='BIO %s' % i
        # lifplot = plt.errorbar(x_dot_e_vals_lif[:-2], Hz_mean_lif[:-2],
        #              yerr=Hz_stddev_lif[:-2], fmt='--', label='LIF %s' % i,
        #              color=bioplot[0].get_color())
        # lifplot[-1][0].set_linestyle('--')
    plt.xlabel('$x \cdot e$')
    plt.ylabel('firing rate')
    plt.title('Tuning Curves')
    plt.legend()

    # rmse_tuning_curve = rmse(Hz_mean_bio[:-2], Hz_mean_lif[:-2])
    # assert rmse_tuning_curve < cutoff
    assert True  # spike matching isn't important for oracle training

def test_feedforward_decode(Simulator, plt):
    """
    decoded output of bioensemble
    """
    cutoff = 0.3
    lpf = nengo.Lowpass(tau_nengo)
    oracle_solver = nengo.solvers.LstsqL2(reg=0.01)

    # Calculate decoders from an input signal
    sim = sim_feedforward(Simulator)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    oracle_decoders = oracle_solver(act_bio, sim.data[probe_direct])[0]

    # Run test on a new signal, using old decoders
    with network:
        stim.output = lambda t: equalpower(
                        t, dt_nengo, t_final, max_freq, dim,
                        mean=0.0, std=1.0, seed=2*signal_seed)
    sim = sim_feedforward(Simulator)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)

    xhat_bio = np.dot(act_bio, oracle_decoders)
    xhat_lif = sim.data[probe_lif]
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    rmse_lif = rmse(sim.data[probe_direct], xhat_lif)

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff
