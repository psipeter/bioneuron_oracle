# import multiprocessing as mp
from pathos import multiprocessing as mp

import numpy as np

import copy

import nengo
from nengo.utils.numpy import rmse
from nengo.base import ObjView

from bioneuron_oracle import (BahlNeuron, prime_sinusoids,
                              equalpower, TrainedSolver)

# __all__ = ['spike_match_train']

def deref_objview(o):  # handle slicing
    return o.obj if isinstance(o, ObjView) else o

def spike_match_train(network, method="1-N", params=None, plots=False):
    if params == None:
        dt_nengo = 0.001
        tau_nengo = 0.01
        n_processes = 10
        popsize = 10
        generations = 2
        delta_w = 2e-3
        evo_seed = 9
        evo_t_final = 1.0
        evo_signal = 'prime_sinusoids'
        evo_max_freq = 5.0
        evo_signal_seed = 234
        evo_cutoff = 50.0
    else:
        dt_nengo = params['dt_nengo']
        tau_nengo = params['tau_nengo']
        n_processes = params['n_processes']
        popsize = params['popsize']
        generations = params['generations']
        delta_w = params['delta_w']
        evo_seed = params['evo_seed']
        evo_t_final = params['evo_t_final']
        evo_signal = params['evo_signal']
        evo_max_freq = params['evo_max_freq']
        evo_signal_seed = params['evo_signal_seed']
        evo_cutoff = params['evo_cutoff']

    # Add a spike probe and a ideal lif ensemble for each bioensemble
    bio_ens = {}
    bio_probes = {}
    ideal_ens = {}
    ideal_probes = {}
    for ens in network.ensembles:
        if isinstance(ens.neuron_type, BahlNeuron):
            with network:
                bio_ens[ens] = ens
                bio_probes[ens] = nengo.Probe(ens.neurons, 'spikes')
                ideal_ens[ens] = nengo.Ensemble(
                             n_neurons=ens.n_neurons, dimensions=ens.dimensions,
                             seed=ens.seed, neuron_type=nengo.LIF(),
                             max_rates=ens.max_rates, intercepts=ens.intercepts)
                ideal_probes[ens] = nengo.Probe(ideal_ens[ens].neurons, 'spikes')

    # TODO: implement loading previously trained weights
    # problem: conn can't be used as a dictionary key becuause it changes
    # every time the network is rebuilt, even with identical seeds. So I can't
    # build the test network, run this function to train, build it again,
    # and check if the new conns are in the old conn list (without explicit labels)
    # try:
    #     saved_weights = np.load('weights/%s.npz' % name)['all_weights']
    #     print 'loading saved weights...'
    #     print saved_weights
    #     for conn in network.connections:
    #         if (hasattr(conn, 'trained_weights')
    #                 and conn.trained_weights == True):
    #             print conn
    #             print saved_weights[conn]
    #             conn.solver.weights_bio = saved_weights[conn]
    #     return network
    # except IOError:
    #     saved_weights = {}
    #     print 'training weights...'
    pool = mp.ProcessingPool(nodes=n_processes)
    rng = np.random.RandomState(seed=evo_seed)

    # Evaluate the fitness of a network using the specified weights,
    # where fitness = sum of rmse(act_bio, act_ideal) for each bio ensemble
    def evaluate(inputs):
        w_bio_p = inputs[0]  # new weights
        network = inputs[1]  # original network
        ens = inputs[2]  # ens to be trained
        bio_probes = inputs[3]  # dict of probes of bioensembles
        ideal_probes = inputs[4]  # dict of probes of ideal ensembles
        w_original = {}  # original weights

        # replace all bioensembles with corresponding LIF ensembles
        # except for the one that is being trained
        ens_changed = {}
        for ens_test in network.ensembles:
            if (isinstance(ens_test.neuron_type, BahlNeuron)
                    and ens_test is not ens):
                ens_changed[ens_test] = True
                with network:
                    ens_test.neuron_type = nengo.LIF()

        for conn in network.connections:
            conn_post = deref_objview(conn.post)
            # only select the ens to be trained
            if conn_post == ens and conn in w_bio_p:  # and conn not in saved_weights
                # save original weights
                w_original[conn] = conn.solver.weights_bio
                # update weights for this fitness evaluation
                conn.solver.weights_bio = w_bio_p[conn]

        with nengo.Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
            sim.run(evo_t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[bio_probes[ens]], dt=dt_nengo)
        act_lif = lpf.filt(sim.data[ideal_probes[ens]], dt=dt_nengo)

        # ensemble-by-ensemble training
        # rmse_act = rmse(act_bio, act_lif)
        # neuron-by-neuron training
        rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
                             for i in range(act_bio.shape[1])])

        # reset network back to original weights and neuron types
        for ens_test in network.ensembles:
            if ens_test in ens_changed:
                with network:
                    ens_test.neuron_type = BahlNeuron()  # TODO: untested
        for conn in network.connections:
            if conn_post == ens and conn in w_bio_p:
                conn.solver.weights_bio = w_original[conn]

        # return rmse_act  # ensemble-by-ensemble training
        return rmses_act  # neuron-by-neuron training

    # all_weights = {}
    for ens in bio_ens.iterkeys():  # For each bioensemble in the network:
        # Figure out how many weight matrices need to be trained,
        # then initialize a random weight matrix of the propper shape for each
        w_pop = [dict() for p in range(popsize)]
        for conn in network.connections:
            conn_pre = deref_objview(conn.pre)
            conn_post = deref_objview(conn.post)
            if (hasattr(conn, 'trained_weights')
                    and conn.trained_weights == True
                    and conn_post == ens):
                    # and conn not in saved_weights):
                n_bio = conn_post.n_neurons
                conn.solver = TrainedSolver(
                    weights_bio = np.zeros((conn_post.n_neurons, conn_pre.n_neurons, conn.n_syn)))
                for p in range(popsize):
                    w_pop[p][conn] = rng.uniform(-1e-1, 1e-1,
                        size=conn.solver.weights_bio.shape)

        # Train the weights using some evolutionary strategy
        # fit_vs_gen = np.zeros((generations))  # ensemble-by-ensemble training
        fit_vs_gen = np.zeros((generations, n_bio))  # neuron-by-neuron training
        for g in range(generations):
            inputs = [[w_pop[p], network, ens, bio_probes, ideal_probes]
                      for p in range(popsize)]
            fitnesses = np.array([evaluate(inputs[0]), evaluate(inputs[1])])  # debugging
            # fitnesses = np.array(pool.map(evaluate, inputs))

            # Find the evo individual with the lowest fitness
            # ensemble-by-ensemble training
            # fit_vs_gen[g] = np.min(fitnesses)
            # w_best = copy.copy(w_pop[np.argmin(fitnesses)])
            # neuron-by-neuron training
            w_best = dict()
            for conn in w_pop[0].iterkeys():
                w_best[conn] = np.zeros_like(w_pop[0][conn])
                for nrn in range(fit_vs_gen.shape[1]):
                    fit_vs_gen[g, nrn] = fitnesses[np.argmin(fitnesses[:,nrn]),nrn]
                    w_best[conn][nrn] = w_pop[np.argmin(fitnesses[:,nrn])][conn][nrn]

            # Copy and mutate the best weight matrices for each new evo individual
            w_pop_new=[]
            for p in range(popsize):
                w_pop_new.append(dict())
                for conn in w_pop[p].iterkeys():
                    w_mutate = rng.uniform(-delta_w, delta_w, size=w_best[conn].shape)
                    w_pop_new[p][conn] = w_best[conn] + w_mutate
            w_pop = copy.copy(w_pop_new)

            # update the connections' solvers in the network
            for conn in network.connections:
                if conn in w_pop[0]:
                    conn.solver.weights_bio = w_best[conn]

        # for conn in network.connections:
        #     if conn in w_pop[0]:
        #         all_weights[conn] = conn.solver.weights_bio

        if plots:
            import matplotlib.pyplot as plt
            figure, ax1 = plt.subplots(1,1)
            ax1.plot(np.arange(0,generations), fit_vs_gen)
            ax1.set(xlabel='Generation', ylabel='Fitness (RMSE_act)')
            ax1.legend()
            figure.savefig('plots/%s_fitness_vs_generation.png' % ens)

    # TODO: save all connection weights
    # if name is None: name = str(network)  # untested
    # np.savez('weights/%s.npz' % name, all_weights = all_weights)

    return network




















# def feedforward(Simulator, plt, pre_neurons, bio_neurons,
#                             tau_nengo, tau_neuron, dt_nengo,
#                             min_rate, max_rate, signal,
#                             pre_seed, bio_seed, conn_seed,
#                             t_final, dim, n_syn, evo_cutoff,
#                             n_processes, popsize, generations,
#                             delta_w, evo_seed, 
#                             max_freq=5, signal_seed=123, plots=False):
#     """
#     Use a 1-N Evolutionary Strategy to train the weights into each bioneuron
#     Fitness is the RMSE of the smoothed spikes for BIO vs ideal LIF
#     """

#     rng = np.random.RandomState(seed=evo_seed)
#     w_0 = rng.uniform(-1e-1, 1e-1, size=(popsize, bio_neurons, pre_neurons, n_syn))

#     def evaluate(weights_bio):
#         with nengo.Network() as model:

#             if signal == 'prime_sinusoids':
#                 stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
#             elif signal == 'step_input':
#                 stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
#             elif signal == 'white_noise':
#                 stim = nengo.Node(lambda t: equalpower(
#                                       t, dt_nengo, t_final, max_freq, dim,
#                                       mean=0.0, std=1.0, seed=signal_seed))

#             pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                  seed=pre_seed, neuron_type=nengo.LIF())
#             bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=BahlNeuron())
#             lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  neuron_type=nengo.LIF(), seed=bio_seed,
#                                  max_rates=nengo.dists.Uniform(min_rate, max_rate))

#             trained_solver = TrainedSolver(weights_bio = weights_bio)
#             nengo.Connection(stim, pre, synapse=None)
#             nengo.Connection(pre, lif, synapse=tau_nengo)
#             nengo.Connection(pre, bio,
#                              seed=conn_seed,
#                              synapse=tau_neuron,
#                              trained_weights=True,
#                              solver=trained_solver)

#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

#         with Simulator(model, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
#         rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
#                              for i in range(bio_neurons)])
#         return rmses_act

#     # Evolutionary section
#     w_new = w_0
#     fit_vs_gen = np.zeros((generations, bio_neurons))
#     pool = mp.ProcessingPool(nodes=n_processes)
#     for g in range(generations):
#         fitnesses = np.zeros((popsize, bio_neurons))
#         w_new = [w_new[p] for p in range(popsize)]
#         fitnesses = np.array(pool.map(evaluate, w_new))
#         # for p in range(popsize):
#         #     fitnesses[p] = evaluate(w_new[p])
#         w_old = np.array(w_new)
#         w_best = np.zeros((bio_neurons, pre_neurons, n_syn))
#         for i in range(bio_neurons):
#             fit_vs_gen[g, i] = fitnesses[np.argmin(fitnesses[:,i]),i]
#             w_best[i] = w_old[np.argmin(fitnesses[:,i]),i]
#         w_new = np.zeros_like(w_old)
#         for p in range(popsize):
#             w_new[p] = w_best + rng.uniform(-delta_w, delta_w,
#                 size=(bio_neurons, pre_neurons, n_syn))

#     # Plot fitness vs generation
#     weights_bio = w_best
#     if plots:
#         import matplotlib.pyplot as plt
#         figure, ax1 = plt.subplots(1,1)
#         ax1.plot(np.arange(0,generations), fit_vs_gen)
#         ax1.set(xlabel='Generation', ylabel='Fitness (RMSE_act)')
#         ax1.legend()
#         figure.savefig('plots/fitness_vs_generation.png')
#     # for f in fit_vs_gen[-1]:
#     #     assert f < evo_cutoff
#     np.savez('weights/weights_bio_feedforward.npz',
#              weights_bio=weights_bio)



# def transform_in(Simulator, plt, pre_neurons, bio_neurons,
#                             tau_nengo, tau_neuron, dt_nengo,
#                             min_rate, max_rate, signal,
#                             pre_seed, bio_seed, conn_seed,
#                             t_final, dim, n_syn, evo_cutoff,
#                             n_processes, popsize, generations,
#                             delta_w, evo_seed, transform=1.0,
#                             max_freq=5, signal_seed=123, plots=False):
#     """
#     Use a 1-N Evolutionary Strategy to train the weights into each bioneuron
#     Fitness is the RMSE of the smoothed spikes for BIO vs ideal LIF
#     """

#     rng = np.random.RandomState(seed=evo_seed)
#     w_0 = rng.uniform(-1e-1, 1e-1, size=(popsize, bio_neurons, pre_neurons, n_syn))

#     def evaluate(weights_bio):
#         with nengo.Network() as model:

#             if signal == 'prime_sinusoids':
#                 stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
#             elif signal == 'step_input':
#                 stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
#             elif signal == 'white_noise':
#                 stim = nengo.Node(lambda t: equalpower(
#                                       t, dt_nengo, t_final, max_freq, dim,
#                                       mean=0.0, std=1.0, seed=signal_seed))

#             pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                  seed=pre_seed, neuron_type=nengo.LIF())
#             bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=BahlNeuron())
#             lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  neuron_type=nengo.LIF(), seed=bio_seed,
#                                  max_rates=nengo.dists.Uniform(min_rate, max_rate))

#             trained_solver = TrainedSolver(weights_bio = weights_bio)
#             nengo.Connection(stim, pre, synapse=None)
#             nengo.Connection(pre, lif, synapse=tau_nengo, transform=transform)
#             nengo.Connection(pre, bio,
#                              seed=conn_seed,
#                              synapse=tau_neuron,
#                              trained_weights=True,
#                              transform=transform,
#                              solver=trained_solver)

#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

#         with Simulator(model, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
#         rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
#                              for i in range(bio_neurons)])
#         return rmses_act

#     # Evolutionary section
#     w_new = w_0
#     fit_vs_gen = np.zeros((generations, bio_neurons))
#     pool = mp.ProcessingPool(nodes=n_processes)
#     for g in range(generations):
#         fitnesses = np.zeros((popsize, bio_neurons))
#         w_new = [w_new[p] for p in range(popsize)]
#         fitnesses = np.array(pool.map(evaluate, w_new))
#         # for p in range(popsize):
#         #     fitnesses[p] = evaluate(w_new[p])
#         w_old = np.array(w_new)
#         w_best = np.zeros((bio_neurons, pre_neurons, n_syn))
#         for i in range(bio_neurons):
#             fit_vs_gen[g, i] = fitnesses[np.argmin(fitnesses[:,i]),i]
#             w_best[i] = w_old[np.argmin(fitnesses[:,i]),i]
#         w_new = np.zeros_like(w_old)
#         for p in range(popsize):
#             w_new[p] = w_best + rng.uniform(-delta_w, delta_w,
#                 size=(bio_neurons, pre_neurons, n_syn))

#     # Plot fitness vs generation
#     weights_bio = w_best
#     if plots:
#         import matplotlib.pyplot as plt
#         figure, ax1 = plt.subplots(1,1)
#         ax1.plot(np.arange(0,generations), fit_vs_gen)
#         ax1.set(xlabel='Generation', ylabel='Fitness (RMSE_act)')
#         ax1.legend()
#         figure.savefig('plots/fitness_vs_generation.png')
#     # for f in fit_vs_gen[-1]:
#     #     assert f < evo_cutoff
#     np.savez('weights/weights_bio_transform_in.npz',
#              weights_bio=weights_bio)


# def two_inputs_two_dims(Simulator, plt, pre_neurons, bio_neurons,
#                             tau_nengo, tau_neuron, dt_nengo,
#                             min_rate, max_rate, signal,
#                             pre_seed, bio_seed, conn_seed,
#                             t_final, dim, n_syn, evo_cutoff,
#                             n_processes, popsize, generations,
#                             delta_w, evo_seed, transform=1.0,
#                             max_freq=5, signal_seed=123, plots=False):
#     """
#     Use a 1-N Evolutionary Strategy to train the weights into each bioneuron
#     Fitness is the RMSE of the smoothed spikes for BIO vs ideal LIF
#     """

#     rng = np.random.RandomState(seed=evo_seed)
#     w_0_1 = rng.uniform(-1e-1, 1e-1, size=(popsize, bio_neurons, pre_neurons, n_syn))
#     w_0_2 = rng.uniform(-1e-1, 1e-1, size=(popsize, bio_neurons, pre_neurons, n_syn))

#     def evaluate(weights_bio1, weights_bio2):
#         with nengo.Network() as model:

#             if signal == 'prime_sinusoids':
#                 stim1 = nengo.Node(
#                     lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])
#                 stim2 = nengo.Node(
#                     lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim])
#             elif signal == 'step_input':
#                 stim1 = nengo.Node(
#                     lambda t: step_input(t, dim, t_final, dt_nengo)[0:dim/2])
#                 stim2 = nengo.Node(
#                     lambda t: step_input(t, dim, t_final, dt_nengo)[dim/2:dim])
#             elif signal == 'white_noise':
#                 stim1 = nengo.Node(lambda t: equalpower(
#                                       t, dt_nengo, t_final, max_freq, dim,
#                                       mean=0.0, std=1.0, seed=signal_seed
#                                       )[0:dim/2])
#                 stim2 = nengo.Node(lambda t: equalpower(
#                                       t, dt_nengo, t_final, max_freq, dim,
#                                       mean=0.0, std=1.0, seed=signal_seed
#                                       )[dim/2:dim])

#             pre1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
#                                   seed=pre_seed, neuron_type=nengo.LIF())
#             pre2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
#                                   seed=2*pre_seed, neuron_type=nengo.LIF())
#             bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=BahlNeuron())
#             lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  neuron_type=nengo.LIF(), seed=bio_seed,
#                                  max_rates=nengo.dists.Uniform(min_rate, max_rate))

#             trained_solver1 = TrainedSolver(weights_bio = weights_bio1)
#             trained_solver2 = TrainedSolver(weights_bio = weights_bio2)
#             nengo.Connection(stim1, pre1, synapse=None)
#             nengo.Connection(stim2, pre2, synapse=None)
#             nengo.Connection(pre1, bio[0],
#                              seed=conn_seed,
#                              synapse=tau_neuron,
#                              trained_weights=True,
#                              transform=transform,
#                              solver=trained_solver1)
#             nengo.Connection(pre2, bio[1],
#                              seed=conn_seed,
#                              synapse=tau_neuron,
#                              trained_weights=True,
#                              transform=transform,
#                              solver=trained_solver2)
#             nengo.Connection(pre1, lif[0], synapse=tau_nengo)
#             nengo.Connection(pre2, lif[1], synapse=tau_nengo)
#             nengo.Connection(stim1, direct[0], synapse=tau_nengo)
#             nengo.Connection(stim2, direct[1], synapse=tau_nengo)

#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

#         with Simulator(model, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
#         rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
#                              for i in range(bio_neurons)])
#         return rmses_act

#     # Evolutionary section
#     w_new_1 = w_0_1
#     w_new_2 = w_0_2
#     fit_vs_gen = np.zeros((generations, bio_neurons))
#     pool = mp.ProcessingPool(nodes=n_processes)
#     for g in range(generations):
#         fitnesses = np.zeros((popsize, bio_neurons))
#         w_new_1 = [w_new_1[p] for p in range(popsize)]
#         w_new_2 = [w_new_2[p] for p in range(popsize)]
#         fitnesses = np.array(pool.map(evaluate, w_new_1, w_new_2))
#         w_old_1 = np.array(w_new_1)
#         w_old_2 = np.array(w_new_2)
#         w_best_1 = np.zeros((bio_neurons, pre_neurons, n_syn))
#         w_best_2 = np.zeros((bio_neurons, pre_neurons, n_syn))
#         for i in range(bio_neurons):
#             fit_vs_gen[g, i] = fitnesses[np.argmin(fitnesses[:,i]),i]
#             w_best_1[i] = w_old_1[np.argmin(fitnesses[:,i]),i]
#             w_best_2[i] = w_old_2[np.argmin(fitnesses[:,i]),i]
#         w_new_1 = np.zeros_like(w_old_1)
#         w_new_2 = np.zeros_like(w_old_2)
#         for p in range(popsize):
#             w_new_1[p] = w_best_1 + rng.uniform(-delta_w, delta_w,
#                 size=(bio_neurons, pre_neurons, n_syn))
#             w_new_2[p] = w_best_2 + rng.uniform(-delta_w, delta_w,
#                 size=(bio_neurons, pre_neurons, n_syn))

#     # Plot fitness vs generation
#     weights_bio_1 = w_best_1
#     weights_bio_2 = w_best_2
#     if plots:
#         import matplotlib.pyplot as plt
#         figure, ax1 = plt.subplots(1,1)
#         ax1.plot(np.arange(0,generations), fit_vs_gen)
#         ax1.set(xlabel='Generation', ylabel='Fitness (RMSE_act)')
#         ax1.legend()
#         figure.savefig('plots/fitness_vs_generation.png')
#     # TODO: repeat training until fitness goal satisfied
#     # for f in fit_vs_gen[-1]:
#     #     assert f < evo_cutoff
#     np.savez('weights/weights_bio_two_inputs_two_dims.npz',
#              weights_bio_1=weights_bio_1, weights_bio2=weights_bio2)