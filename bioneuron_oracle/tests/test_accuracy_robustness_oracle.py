from functools32 import lru_cache

from seaborn import set_palette, color_palette, tsplot

import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.utils.numpy import rmse

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input,
                              equalpower, OracleSolver)


def test_accuracy_vs_n_neurons(Simulator, plt):
    """
    Simulate a feedforward network with bioneurons whose input weights
    have been trained by with a spike matching approach(), varying
    the number of bio_neurons and measuring RMSE of xhat_bio
    """

    bio_neurons = np.array([2,3,4,5,6,7,8,9,10])
    n_avg = 10
    pre_seed = np.random.randint(0,9009,size=(bio_neurons.shape[0], n_avg))
    bio_seed = np.random.randint(0,9009,size=(bio_neurons.shape[0], n_avg))
    conn_seed = np.random.randint(0,9009,size=(bio_neurons.shape[0], n_avg))
    network_seed = 12
    sim_seed = 15
    post_seed = 18
    max_freq = 5
    signal_seed = 123

    def sim(Simulator, plt, bio_neurons, bio_seed, pre_seed, conn_seed):

        # Nengo Parameters
        pre_neurons = 100
        tau_nengo = 0.05
        tau_neuron = 0.05
        dt_nengo = 0.001
        min_rate = 150
        max_rate = 200
        t_final = 1.0
        dim = 1
        n_syn = 1
        signal = 'prime_sinusoids'

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
            lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=bio.dimensions,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())

            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, lif, synapse=tau_nengo)
            nengo.Connection(pre, bio, synapse=tau_neuron, seed=conn_seed,
                             weights_bias_conn=True, n_syn=n_syn)
            nengo.Connection(stim, direct, synapse=tau_nengo)

            probe_pre = nengo.Probe(pre, synapse=tau_nengo)
            probe_lif = nengo.Probe(lif, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
        
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)

        # compute oracle decoders
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
        oracle_decoders = oracle_solver(act_bio, sim.data[probe_direct])[0]

        # compute estimate on a new signal
        with network:
            stim.output = lambda t: equalpower(
                            t, dt_nengo, t_final, max_freq, dim,
                            mean=0.0, std=1.0, seed=2*signal_seed)
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        xhat_bio = np.dot(act_bio, oracle_decoders)
        xhat_lif = sim.data[probe_lif]
        rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
        rmse_lif = rmse(sim.data[probe_direct], xhat_lif)

        return rmse_bio, rmse_lif

    rmse_bio_mean = []
    rmse_bio_std = []
    rmse_lif_mean = []
    rmse_lif_std = []
    for i in range(bio_neurons.shape[0]):
        rmse_bio_i = []
        rmse_lif_i = []
        for j in range(bio_seed.shape[1]):
            rmse_bio, rmse_lif = sim(Simulator, plt,
                bio_neurons[i], bio_seed[i,j], pre_seed[i,j], conn_seed[i,j])
            rmse_bio_i.append(rmse_bio)
            rmse_lif_i.append(rmse_lif)
        rmse_bio_mean.append(np.mean(rmse_bio_i))
        rmse_bio_std.append(np.std(rmse_bio_i))
        rmse_lif_mean.append(np.mean(rmse_lif_i))
        rmse_lif_std.append(np.std(rmse_lif_i))
    rmse_bio_mean = np.array(rmse_bio_mean)
    rmse_bio_std = np.array(rmse_bio_std)
    rmse_lif_mean = np.array(rmse_lif_mean)
    rmse_lif_std = np.array(rmse_lif_std)

    plt.subplot(1, 1, 1)
    plt.errorbar(bio_neurons, rmse_bio_mean, yerr=rmse_bio_std, label='bio')
    plt.errorbar(bio_neurons, rmse_lif_mean, yerr=rmse_lif_std, label='lif')
    plt.plot(bio_neurons, 1.0 / bio_neurons, label='theory')
    plt.xlabel('n_neurons')
    plt.ylabel('RMSE ($\hat{x}(t)$, $x(t)$)')
    # plt.title('decode')
    plt.legend()  # prop={'size':8}


# def test_bio_switch(Simulator, plt):

#     cutoff = 0.3
#     transform = -0.5
#     dim = 1
#     bio_neurons = 20

#     def sim(w_train, decoders_bio=None, switched=False, plots=False):

#         if decoders_bio is None:
#             decoders_bio = np.zeros((bio_neurons, dim))

#         with nengo.Network() as network:

#             stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

#             pre1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                  seed=pre_seed, neuron_type=nengo.LIF())
#             bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=BahlNeuron())
#             lif1 = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
#                                  seed=bio.seed, neuron_type=nengo.LIF(),
#                                  max_rates=bio.max_rates, intercepts=bio.intercepts)
#             direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                     neuron_type=nengo.Direct())
#             pre2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=nengo.LIF())
#             bio2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                   seed=post_seed, neuron_type=BahlNeuron())
#             lif2 = nengo.Ensemble(n_neurons=bio2.n_neurons, dimensions=bio2.dimensions,
#                                  seed=bio2.seed, neuron_type=nengo.LIF(),
#                                  max_rates=bio2.max_rates, intercepts=bio2.intercepts)
#             direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                         neuron_type=nengo.Direct())

#             oracle_solver = OracleSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

#             nengo.Connection(stim, pre1, synapse=None)
#             nengo.Connection(pre1, bio, synapse=tau_neuron,
#                              weights_bias_conn=True)
#             nengo.Connection(pre1, lif1, synapse=tau_nengo)
#             nengo.Connection(pre1, direct, synapse=tau_nengo)
#             nengo.Connection(direct, pre2, synapse=None)
#             nengo.Connection(pre2, bio2, synapse=tau_nengo, transform=w_train)
#             nengo.Connection(lif1, lif2, synapse=tau_nengo, transform=transform)
#             nengo.Connection(bio, bio2, synapse=tau_nengo,
#                              solver=oracle_solver, transform=transform)
#             nengo.Connection(direct, direct2,
#                              synapse=tau_nengo, transform=transform)
#             # TODO: test output to node, direct, etc

#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
#             probe_lif_spikes = nengo.Probe(lif1.neurons, 'spikes')
#             probe_lif2_spikes = nengo.Probe(lif2.neurons, 'spikes')
#             probe_direct = nengo.Probe(direct, synapse=tau_nengo)
#             probe_direct2 = nengo.Probe(direct2, synapse=None)

#         # switch the bio1 population for one with new tuning curves (new seed)
#         if switched == True:
#             bio.seed = 5643

#         with Simulator(network, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
#         act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
#         act_lif2 = lpf.filt(sim.data[probe_lif2_spikes], dt=dt_nengo)
#         assert np.sum(act_bio) > 0.0
#         assert np.sum(act_bio2) > 0.0
#         solver = nengo.solvers.LstsqL2(reg=0.1)
#         decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
#         decoders_bio_new2, info2 = solver(act_bio2, sim.data[probe_direct2])
#         decoders_lif_new, info = solver(act_lif, sim.data[probe_direct])
#         decoders_lif_new2, info2 = solver(act_lif2, sim.data[probe_direct2])
#         xhat_bio = np.dot(act_bio, decoders_bio_new)
#         xhat_bio2 = np.dot(act_bio2, decoders_bio_new2)
#         xhat_lif = np.dot(act_lif, decoders_lif_new)
#         xhat_lif2 = np.dot(act_lif2, decoders_lif_new2)
#         rmse_bio = rmse(xhat_bio, sim.data[probe_direct])
#         rmse_bio2 = rmse(xhat_bio2, sim.data[probe_direct2])
#         rmse_lif = rmse(xhat_lif, sim.data[probe_direct])
#         rmse_lif2 = rmse(xhat_lif2, sim.data[probe_direct2])

#         # plt.plot(sim.trange(), sim.data[probe_bio],
#         #          label='BIO (probe)')
#         # plt.plot(sim.trange(), xhat_bio,
#         #          label='BIO, rmse=%.5f' % rmse_bio)
#         # plt.plot(sim.trange(), xhat_lif,
#         #          label='lif, rmse=%.5f' % rmse_lif)
#         plt.plot(sim.trange(), xhat_bio2,
#                  label='bio 2, %s, rmse=%.5f' % (plots, rmse_bio2))
#         # plt.plot(sim.trange(), xhat_lif2,
#         #          label='lif 2, rmse=%.5f' % rmse_lif2)
#         # plt.plot(sim.trange(), sim.data[probe_direct],
#         #          label='oracle')
#         # plt.plot(sim.trange(), sim.data[probe_direct2],
#         #          label='oracle 2')
#         plt.legend()  # prop={'size':8}


#         if plots == 'new_spikes_new_decoders':
#             plt.plot(sim.trange(), sim.data[probe_direct2],
#                      label='oracle 2')            

#         return decoders_bio_new, rmse_bio, rmse_bio2

#     plt.subplot(1, 1, 1)
#     plt.xlabel('time (s)')
#     plt.ylabel('$\hat{x}(t)$')
#     plt.title('decode')

#     decoders_bio, rmse_bio, rmse_bio2 = sim(
#         w_train=1.0, decoders_bio=None,
#         plots='untrained')
#     decoders_bio_old, rmse_bio, rmse_bio2 = sim(
#         w_train=0.0, decoders_bio=decoders_bio,
#         plots='old_spikes_old_decoders')
#     decoders_bio_new, rmse_bio, rmse_bio2 = sim(
#         w_train=0.0, decoders_bio=decoders_bio_old,
#         plots='new_spikes_old_decoders', switched=True)
#     decoders_bio_extra, rmse_bio, rmse_bio2 = sim(
#         w_train=0.0, decoders_bio=decoders_bio_new,
#         plots='new_spikes_new_decoders', switched=True) #todo: plotting

#     assert rmse_bio < cutoff
#     assert rmse_bio2 < cutoff