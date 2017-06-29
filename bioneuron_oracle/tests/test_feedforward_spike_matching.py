from functools32 import lru_cache

from seaborn import set_palette, color_palette, tsplot

import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.utils.numpy import rmse

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input, equalpower,
	                          TrainedSolver, OracleSolver, spike_match_train)

tau_nengo = 0.05
dt_nengo = 0.001

@lru_cache(maxsize=None)
def sim_feedforward(Simulator, signal='prime_sinusoids', t_final=1.0, train=True):
    """
    Simulate a feedforward network with bioneurons whose input weights
    have been trained by with a spike matching approach()
    """

    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 20
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
    dim = 1
    n_syn = 1
    max_freq = 5
    signal_seed = 2*123

    # Evolutionary Parameters
    evo_params = {
        'dt_nengo': 0.001,
        'tau_nengo': 0.05,
        'n_processes': 10,
        'popsize': 10,
        'generations' : 100,
        'w_0': 1e-3,
        'delta_w' :1e-4,
        'evo_seed' :9,
        'evo_t_final' :1.0,
        'evo_cutoff' :50.0,
        'sim_seed': 15,
    }

    try:
        w_bio_0 = np.load('weights/w_pre_to_bio.npz')['weights_bio']
        loaded = True
    except IOError:
        w_bio_0 = np.zeros((bio_neurons, pre_neurons, n_syn))
        loaded = False

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
        temp = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct()) 

        trained_solver = TrainedSolver(weights_bio = w_bio_0)
        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, lif, synapse=tau_nengo)
        pre_to_bio = nengo.Connection(pre, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver,
                         n_syn=n_syn)
        nengo.Connection(stim, direct, synapse=tau_nengo)
        conn_ideal_out = nengo.Connection(lif, temp, synapse=tau_nengo,
                         solver=nengo.solvers.LstsqL2())

        probe_pre = nengo.Probe(pre, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif, synapse=tau_nengo)
        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
        probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

    conn_before = {}
    for conn in network.connections:
        if hasattr(conn, 'trained_weights') and conn.trained_weights == True:
            conn_before[conn] = conn.solver.weights_bio

    if train:
        network = spike_match_train(network, method="1-N", params=evo_params, plots=True)
    
    with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_final)

    conn_after = {}
    for conn in network.connections:
        if hasattr(conn, 'trained_weights') and conn.trained_weights == True:
            conn_after[conn] = conn.solver.weights_bio

    for conn in conn_before.iterkeys():  # make sure training had some impact
        if loaded == False:
            assert conn_before[conn] is not conn_after[conn]

    np.savez('weights/w_pre_to_bio.npz',weights_bio=pre_to_bio.solver.weights_bio)

    return sim, pre, bio, lif, direct, conn_ideal_out, \
            probe_pre, probe_lif, probe_direct, \
            probe_bio_spikes, probe_lif_spikes


def test_feedforward_activities(Simulator, plt):
    """
    Plot a(t) for the trained network
    """

    sim, pre, bio, lif, direct, conn_ideal_out, \
            probe_pre, probe_lif, probe_direct, \
            probe_bio_spikes, probe_lif_spikes = \
        sim_feedforward(Simulator)

    cutoff = 50.0

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
    rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
                         for i in range(bio.n_neurons)])

    for i in range(bio.n_neurons):
        plt.subplot(1, 1, 1)
        bioplot = plt.plot(sim.trange(), act_bio[:,i],
                           label='BIO neuron %s, RMSE = %s'
                           % (i,rmses_act[i]))
        lifplot = plt.plot(sim.trange(), act_lif[:,i],
                           label='LIF neuron %s' % i,
                           color=bioplot[0].get_color(),
                           linestyle='--')
        plt.xlabel('time (s)')
        plt.ylabel('$a(t)$')
        # plt.title()
        plt.legend()

    for rmse_i in rmses_act:
        assert rmse_i < cutoff

def test_feedforward_tuning_curves(Simulator, plt):

    sim, pre, bio, lif, direct, conn_ideal_out, \
            probe_pre, probe_lif, probe_direct, \
            probe_bio_spikes, probe_lif_spikes = \
        sim_feedforward(Simulator)

    n_eval_points = 20
    cutoff = 50.0

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
    rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
                         for i in range(bio.n_neurons)])

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

        rmse_tuning_curve = rmse(Hz_mean_bio[:-2], Hz_mean_lif[:-2])
        bioplot = plt.errorbar(x_dot_e_vals_bio[:-2], Hz_mean_bio[:-2],
                     yerr=Hz_stddev_bio[:-2], fmt='-o', label='BIO %s, RMSE=%.5f' % (i, rmse_tuning_curve))
        lifplot = plt.errorbar(x_dot_e_vals_lif[:-2], Hz_mean_lif[:-2],
                     yerr=Hz_stddev_lif[:-2], fmt='--', label='LIF %s' % i,
                     color=bioplot[0].get_color())
        lifplot[-1][0].set_linestyle('--')
    plt.xlabel('$x \cdot e$')
    plt.ylabel('firing rate')
    plt.title('Tuning Curves')
    plt.legend()


    assert rmse_tuning_curve < cutoff


def test_feedforward_decode(Simulator, plt):
    """
    plot xhat_bio(t), xhat_ideal(t), and x(t) from direct
    """
    cutoff = 0.2

    signal = 'prime_sinusoids'
    sim, pre, bio, lif, direct, conn_ideal_out, \
            probe_pre, probe_lif, probe_direct, \
            probe_bio_spikes, probe_lif_spikes = \
        sim_feedforward(Simulator, signal, t_final=1.0, train=False)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio_old = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    act_lif_old = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
    oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
    oracle_decoders_bio_old = oracle_solver(act_bio_old, sim.data[probe_direct])[0]
    oracle_decoders_lif_old = oracle_solver(act_lif_old, sim.data[probe_direct])[0]

    signal = 'white_noise'
    sim, pre, bio, lif, direct, conn_ideal_out, \
            probe_pre, probe_lif, probe_direct, \
            probe_bio_spikes, probe_lif_spikes = \
        sim_feedforward(Simulator, signal, train=False)

    act_bio_new = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    act_lif_new = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
    oracle_decoders_bio_new = oracle_solver(act_bio_new, sim.data[probe_direct])[0]
    oracle_decoders_lif_new = oracle_solver(act_lif_new, sim.data[probe_direct])[0]
    static_decoders = sim.data[conn_ideal_out].weights.T

    xhat_bio_oracle_old = np.dot(act_bio_new, oracle_decoders_bio_old)
    xhat_bio_oracle_new = np.dot(act_bio_new, oracle_decoders_bio_new)
    xhat_bio_static = np.dot(act_bio_new, static_decoders)
    xhat_lif_oracle_old = np.dot(act_lif_new, oracle_decoders_lif_old)
    xhat_lif_oracle_new = np.dot(act_lif_new, oracle_decoders_lif_new)
    xhat_lif_static = np.dot(act_lif_new, static_decoders)

    rmse_bio_oracle_old = rmse(sim.data[probe_direct], xhat_bio_oracle_old)
    rmse_bio_oracle_new = rmse(sim.data[probe_direct], xhat_bio_oracle_new)
    rmse_bio_static = rmse(sim.data[probe_direct], xhat_bio_static)
    rmse_lif_oracle_old = rmse(sim.data[probe_direct], xhat_lif_oracle_old)
    rmse_lif_oracle_new = rmse(sim.data[probe_direct], xhat_lif_oracle_new)
    rmse_lif_static = rmse(sim.data[probe_direct], xhat_lif_static)

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio_oracle_old,
             label='bio, old oracle decoders, rmse=%.5f' % rmse_bio_oracle_old)
    plt.plot(sim.trange(), xhat_bio_oracle_new,
             label='bio, new oracle decoders, rmse=%.5f' % rmse_bio_oracle_new)
    plt.plot(sim.trange(), xhat_bio_static,
             label='bio, LIF static decoders, rmse=%.5f' % rmse_bio_static)
    plt.plot(sim.trange(), xhat_lif_oracle_old,
             label='lif, old oracle decoders, rmse=%.5f' % rmse_lif_oracle_old)
    plt.plot(sim.trange(), xhat_lif_oracle_new,
             label='lif, new oracle decoders, rmse=%.5f' % rmse_lif_oracle_new)
    plt.plot(sim.trange(), xhat_lif_static,
             label='lif, LIF static decoders, rmse=%.5f' % rmse_lif_static)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}

    assert rmse_bio_oracle_old < cutoff
    assert rmse_lif_static < cutoff
