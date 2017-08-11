from functools32 import lru_cache

import numpy as np

from scipy.stats import linregress

import neuron

import nengo
from nengo.utils.numpy import rmse
from nengo.utils.matplotlib import rasterplot

from seaborn import set_palette, color_palette, tsplot

from bioneuron_oracle import (BahlNeuron, TrainedSolver, OracleSolver, spike_train)
from bioneuron_oracle.builder import Bahl, ExpSyn


def test_tuning_curves(Simulator, plt):

    """
    Build a feedforward communication channel with the oracle method,
    then construct tuning curves and see how they match with the 'ideal'
    curves (those used for spike-matching). We expect them to have similar
    intercepts and signed-slopes (encoders), but different gains.
    """

    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 9
    tau = 0.01
    tau_readout = 0.01
    dt = 0.001
    min_rate = 150
    max_rate = 200
    radius = 1
    bio_radius = 1
    n_syn = 1

    pre_seed = 1
    bio_seed = 2
    conn_seed = 3
    network_seed = 4
    sim_seed = 5
    post_seed = 6
    inter_seed = 7

    max_freq = 5
    rms = 0.25

    dim = 1
    reg = 0.1
    t_final = 10.0
    cutoff = 0.1

    def sim(
        signal='sinusoids',
        t_final=1.0,
        freq=1,
        seeds=1,
        transform=1,
        plot=False):

        """
        Define the network
        """
        with nengo.Network(seed=network_seed) as network:

            if signal == 'sinusoids':
                stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq * t))
            elif signal == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=seeds))
            elif signal == 'step':
                stim = nengo.Node(lambda t:
                    np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
            elif signal == 'constant':
                stim = nengo.Node(lambda t: freq)

            pre = nengo.Ensemble(
                n_neurons=pre_neurons,
                dimensions=dim,
                seed=pre_seed,
                neuron_type=nengo.LIF(),
                radius=radius,
                label='pre')
            bio = nengo.Ensemble(
                n_neurons=bio_neurons,
                dimensions=dim,
                seed=bio_seed,
                neuron_type=BahlNeuron(),
                # neuron_type=nengo.LIF(),
                radius=bio_radius,
                max_rates=nengo.dists.Uniform(min_rate, max_rate),
                label='bio')
            lif = nengo.Ensemble(
                n_neurons=bio_neurons,
                dimensions=dim,
                seed=bio_seed,
                max_rates=nengo.dists.Uniform(min_rate, max_rate),
                radius=bio_radius,
                neuron_type=nengo.LIF(),
                label='lif')
            oracle = nengo.Node(size_in=dim)

            nengo.Connection(stim, pre, synapse=None)
            conn_bio = nengo.Connection(pre, bio,
                weights_bias_conn=True,
                seed=conn_seed,
                synapse=tau,
                transform=transform,
                n_syn=n_syn)

            conn_ideal = nengo.Connection(pre, lif,
                synapse=tau,
                transform=transform)
            conn_oracle = nengo.Connection(stim, oracle,
                synapse=tau,
                transform=transform)


            probe_stim = nengo.Probe(stim, synapse=None)
            probe_pre = nengo.Probe(pre, synapse=tau_readout)
            probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_oracle = nengo.Probe(oracle, synapse=tau_readout)


        """
        Run the simulation
        """
        with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_readout)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)

        """
        Plot tuning curves with the trained weights
        """
        if plot:
            n_eval_points = 20
            for i in range(bio.n_neurons):
                x_dot_e_bio = np.dot(
                    sim.data[probe_pre],
                    bio.encoders[i])
                x_dot_e_lif = np.dot(
                    sim.data[probe_pre],
                    sim.data[lif].encoders[i])
                x_dot_e_vals_bio = np.linspace(
                    np.min(x_dot_e_bio),
                    np.max(x_dot_e_bio), 
                    num=n_eval_points)
                x_dot_e_vals_lif = np.linspace(
                    np.min(x_dot_e_lif),
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
                # lifplot = plt.plot(x_dot_e_vals_lif[:-2], Hz_mean_lif[:-2],
                #     label='lif',
                #     ls='dotted')
                # plt.fill_between(x_dot_e_vals_lif[:-2],
                #     Hz_mean_lif[:-2]+Hz_stddev_lif[:-2],
                #     Hz_mean_lif[:-2]-Hz_stddev_lif[:-2],
                #     alpha=0.25,
                #     facecolor=lifplot[0].get_color())
                bioplot = plt.plot(x_dot_e_vals_bio[:-2], Hz_mean_bio[:-2],
                    # marker='.',
                    ls='solid')
                    # color=lifplot[0].get_color(),
                    # label='bio')
                plt.fill_between(x_dot_e_vals_bio[:-2],
                    Hz_mean_bio[:-2]+Hz_stddev_bio[:-2],
                    Hz_mean_bio[:-2]-Hz_stddev_bio[:-2],
                    alpha=0.5,
                    # facecolor=lifplot[0].get_color())
                    facecolor=bioplot[0].get_color())

            plt.ylim(ymin=0)
            plt.xlabel('$x \cdot e$')
            plt.ylabel('firing rate')
            plt.title('Tuning Curves')
            plt.legend()

        return


    """
    Run the test
    """
    sim(
        signal='white_noise',
        freq=1,
        seeds=1,
        transform=5,
        t_final=t_final,
        plot=True)