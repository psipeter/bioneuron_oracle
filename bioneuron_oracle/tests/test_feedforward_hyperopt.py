from functools32 import lru_cache

# import multiprocessing as mp
from pathos import multiprocessing as mp

import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.utils.numpy import rmse

from bioneuron_oracle import BahlNeuron, prime_sinusoids, step_input
from bioneuron_oracle import TrainedSolver


# @lru_cache(maxsize=None)
def test_feedforward_1_N_ES(Simulator, plt):
    """
    Use a 1-N Evolutionary Strategy to train the weights into each bioneuron
    Fitness is the RMSE of the smoothed spikes for BIO vs ideal LIF
    """

    # Nengo Parameters
    pre_neurons = 10
    bio_neurons = 2
    tau_nengo = 0.05
    tau_neuron = 0.05
    dt_nengo = 0.001
    min_rate = 150
    max_rate = 200
    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    t_final = 1.0
    dim = 1
    n_syn = 1
    cutoff = 10.0

    # Evolutionary Parameters
    n_processes = 8
    popsize = 100
    generations = 50
    delta_w = 2e-3
    evo_seed = 9
    rng = np.random.RandomState(seed=evo_seed)
    w_0 = rng.uniform(-1e-1, 1e-1, size=(popsize, bio_neurons, pre_neurons, n_syn))

    def evaluate(weights_bio, plots=False):
        with nengo.Network() as model:
            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 neuron_type=nengo.LIF(), seed=bio_seed,
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate))
            trained_solver = TrainedSolver(weights_bio = weights_bio)
            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, lif, synapse=tau_nengo)
            nengo.Connection(pre, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver=trained_solver)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
        with Simulator(model, dt=dt_nengo) as sim:
            sim.run(t_final)
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
        rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
                             for i in range(bio_neurons)])
        if plots:
            for i in range(1):
                plt.subplot(1, 1, 1)
                plt.plot(sim.trange(), act_bio[:,i], label='BIO')
                plt.plot(sim.trange(), act_lif[:,i], label='LIF')
                plt.xlabel('time (s)')
                plt.ylabel('$\hat{x}(t)$')
                plt.title('RMSE = %s' % rmses_act[i])
                plt.legend()  # prop={'size':8}
        return rmses_act
        # output.put(rmses_act)

    w_new = w_0
    fit_vs_gen = np.zeros((generations, bio_neurons))
    pool = mp.ProcessingPool(nodes=n_processes)
    # output = mp.Queue()

    for g in range(generations):
        fitnesses = np.zeros((popsize, bio_neurons))
        w_new = [w_new[p] for p in range(popsize)]
        fitnesses = np.array(pool.map(evaluate, w_new))
        # for p in range(popsize):
        #     fitnesses[p] = evaluate(w_new[p])
        # print 'fitnesses', fitnesses
        w_old = np.array(w_new)
        w_best = np.zeros((bio_neurons, pre_neurons, n_syn))
        fitnesses_best = np.zeros((bio_neurons))
        for i in range(bio_neurons):
            min_fitness_idx = np.argmin(fitnesses[:,i])
            fitnesses_best[i] = fitnesses[min_fitness_idx,i]
            w_best[i] = w_old[min_fitness_idx,i]
        fit_vs_gen[g] = fitnesses_best
        # print 'best fitnesses', fitnesses_best
        w_new = np.zeros_like(w_old)
        for p in range(popsize):
            w_new[p] = w_best + rng.uniform(-delta_w, delta_w,
                size=(bio_neurons, pre_neurons, n_syn))

    weights_bio = w_best
    f = evaluate(w_best, plots=True)
    print fit_vs_gen
    # plt.subplot(1, 1, 1)
    # plt.plot(range(generations), fit_vs_gen)
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness (RMSE_act')
    # plt.legend()  # prop={'size':8}
    for f in fitnesses_best:
        assert f < cutoff





# @lru_cache(maxsize=None)
# def sim_feedforward(Simulator):
#     sim = Simulator(model, dt=dt_nengo)
#     with sim:
#         sim.run(t_final)
#     return sim


# def test_feedforward_connection(Simulator, plt):
#     """
#     spike raster for PRE, BIO and comparison LIF ensembles
#     test passes if:
#         - a certain number of bioneurons fire at least once
#         - this number should be within $cutoff$ %%
#           of the number of active LIFs
#     """
#     sim = sim_feedforward(Simulator)
#     plt.subplot(3, 1, 1)
#     rasterplot(sim.trange(), sim.data[probe_pre_spikes], use_eventplot=True)
#     plt.ylabel('PRE')
#     plt.title('spike raster')
#     plt.yticks([])
#     plt.subplot(3, 1, 2)
#     rasterplot(sim.trange(), sim.data[probe_bio_spikes], use_eventplot=True)
#     plt.ylabel('BIO')
#     plt.yticks([])
#     plt.subplot(3, 1, 3)
#     rasterplot(sim.trange(), sim.data[probe_lif_spikes], use_eventplot=True)
#     plt.xlabel('time (s)')
#     plt.ylabel('LIF')
#     plt.yticks([])
#     cutoff = 0.5
#     n_bio = len(np.nonzero(
#         np.sum(sim.data[probe_bio_spikes], axis=0))[0])  # n_active
#     n_lif = len(np.nonzero(
#         np.sum(sim.data[probe_lif_spikes], axis=0))[0])  # n_active
#     assert (1 - cutoff) * n_lif < n_bio
#     assert n_bio < (1 + cutoff) * n_lif


# def test_voltage(Simulator, plt):
#     """
#     test passes if:
#         - a neuron spends less than $cutoff_sat$ %% of time
#           in the 'saturated' voltage regime -40<V<-20,
#           which would indicate synaptic inputs are overstimulating
#         - less that $cutoff_eq$ %% of time near equilibrium, -67<V<-63
#           which would indicate that synaptic inputs are understimulating
#           (or not being delivered)
#     """
#     sim = sim_feedforward(Simulator)
#     cutoff_sat = 0.3
#     cutoff_eq = 0.5
#     for bioneuron in sim.data[bio.neurons]:
#         V = np.array(bioneuron.v_record)
#         time = np.array(bioneuron.t_record)
#         t_saturated = len(np.where((-40.0 < V) & (V < -20.0))[0])
#         t_equilibrium = len(np.where((-70.0 < V) & (V < -69.0))[0])
#         t_total = len(time)
#         f_sat = 1. * t_saturated / t_total
#         f_eq = 1. * t_equilibrium / t_total
#         if (f_sat < cutoff_sat or f_eq < cutoff_eq):
#             plt.subplot(111)
#             plt.plot(time, V, label='saturated=%s, equilibrium=%s' %
#                                     (f_sat, f_eq))
#             plt.xlabel('time (ms)')
#             plt.ylabel('voltage (mV)')
#             plt.title('voltage')
#         assert f_sat < cutoff_sat
#         assert f_eq < cutoff_eq


# def test_feedforward_decode(Simulator, plt):
#     """
#     decoded output of bioensemble
#     test passes if:
#         - rmse_bio is within $cutoff$ %% of rmse_lif
#     """
#     sim = sim_feedforward(Simulator)
#     cutoff = 0.5
#     plt.subplot(1, 1, 1)
#     lpf = nengo.Lowpass(tau_nengo)
#     solver = nengo.solvers.LstsqL2(reg=0.01)
#     act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#     decoders_bio, info = solver(act_bio, sim.data[probe_direct])
#     xhat_bio = np.dot(act_bio, decoders_bio)
#     rmse_bio = np.sqrt(np.average((
#         sim.data[probe_direct] - xhat_bio)**2))
#     rmse_lif = np.sqrt(np.average((
#         sim.data[probe_direct] - sim.data[probe_lif])**2))
#     plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
#     plt.plot(sim.trange(), sim.data[probe_lif],
#              label='lif, rmse=%.5f' % rmse_lif)
#     plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
#     plt.xlabel('time (s)')
#     plt.ylabel('$\hat{x}(t)$')
#     plt.title('decode')
#     plt.legend()  # prop={'size':8}
#     assert (1 - cutoff) * rmse_lif < rmse_bio
#     assert rmse_bio < (1 + cutoff) * rmse_lif
