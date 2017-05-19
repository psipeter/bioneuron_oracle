from functools32 import lru_cache

import numpy as np

import nengo
from nengo.utils.numpy import rmse

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input,
                              equalpower, BioSolver, TrainedSolver)
from bioneuron_oracle.tests.spike_match_train import spike_match_train


def test_new_LIF_old_decoders_spike_matching(Simulator, plt):

    """
    Simulate a feedforward network with bioneurons whose input weights
    have been trained by with a spike matching approach(),
    then change the pre_seed, run again, and compare the decoded RMSE
    """

    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 20
    tau_nengo = 0.01
    tau_neuron = 0.01
    dt_nengo = 0.001
    min_rate = 150
    max_rate = 200
    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    t_final = 1.0
    dim = 1
    n_syn = 1
    signal = 'prime_sinusoids'

    # Evolutionary Parameters
    evo_params = {
        'dt_nengo': 0.001,
        'tau_nengo': 0.01,
        'n_processes': 10,
        'popsize': 20,
        'generations' : 40,
        'delta_w' :2e-3,
        'evo_seed' :9,
        'evo_t_final' :1.0,
        'evo_signal': 'prime_sinusoids',
        'evo_max_freq': 5.0,
        'evo_signal_seed': 234,
        'evo_cutoff' :50.0,
    }

    with nengo.Network() as network:

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

        trained_solver = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, pre.n_neurons, n_syn)))
        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, lif, synapse=tau_nengo)
        nengo.Connection(pre, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver,
                         n_syn=n_syn)
        nengo.Connection(stim, direct, synapse=tau_nengo)
        conn_ideal_out = nengo.Connection(lif, temp, synapse=tau_nengo,
                         solver=nengo.solvers.LstsqL2())

        probe_pre = nengo.Probe(pre, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif)
        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
        probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

    network = spike_match_train(network, method="1-N",
                                params=evo_params, plots=True)

    with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
        sim.run(t_final)

    # Generate decoders and a basic decoding for comparison
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    decoders_bio_old, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio_old = np.dot(act_bio, decoders_bio_old)
    rmse_bio_old = rmse(sim.data[probe_direct], xhat_bio_old)

    pre_seed = 9
    cutoff_mixed = 0.3
    cutoff_compare = 0.3
    pre.seed = pre_seed    
    with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
        sim.run(t_final)

    # Generate a new decoding using the old decoders and new activities
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio_mixed = np.dot(act_bio, decoders_bio_old)
    rmse_bio_mixed = rmse(sim.data[probe_direct], xhat_bio_mixed)
    # Generate a new decoding using the new decoders and new activities
    decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio_new = np.dot(act_bio, decoders_bio_new)
    rmse_bio_new = rmse(sim.data[probe_direct], xhat_bio_new)
    # Calculate the RMSE of the mixed and new decoding
    rmse_bio_compare = rmse(xhat_bio_mixed, xhat_bio_new)

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio_old,
             label='old, rmse=%.5f' % rmse_bio_old)
    plt.plot(sim.trange(), xhat_bio_mixed,
             label='mixed, rmse=%.5f' % rmse_bio_mixed)
    plt.plot(sim.trange(), xhat_bio_new,
             label='new, rmse=%.5f' % rmse_bio_new)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}

    assert rmse_bio_mixed < cutoff_mixed
    assert rmse_bio_compare < cutoff_compare    


def test_new_signal_old_decoders_spike_matching(Simulator, plt):

    """
    Simulate a feedforward network with bioneurons whose input weights
    have been trained by with a spike matching approach(),
    then change the pre_seed, run again, and compare the decoded RMSE
    """

    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 20
    tau_nengo = 0.01
    tau_neuron = 0.01
    dt_nengo = 0.001
    min_rate = 150
    max_rate = 200
    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    t_final = 1.0
    dim = 1
    n_syn = 1
    signal = 'prime_sinusoids'

    # Evolutionary Parameters
    evo_params = {
        'dt_nengo': 0.001,
        'tau_nengo': 0.01,
        'n_processes': 10,
        'popsize': 10,
        'generations' : 2,
        'delta_w' :2e-3,
        'evo_seed' :9,
        'evo_t_final' :1.0,
        'evo_signal': 'prime_sinusoids',
        'evo_max_freq': 5.0,
        'evo_signal_seed': 234,
        'evo_cutoff' :50.0,
    }

    with nengo.Network() as network:

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

        trained_solver = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, pre.n_neurons, n_syn)))
        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, lif, synapse=tau_nengo)
        nengo.Connection(pre, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver,
                         n_syn=n_syn)
        nengo.Connection(stim, direct, synapse=tau_nengo)
        conn_ideal_out = nengo.Connection(lif, temp, synapse=tau_nengo,
                         solver=nengo.solvers.LstsqL2())

        probe_pre = nengo.Probe(pre, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif)
        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
        probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

    network = spike_match_train(network, method="1-N",
                                params=evo_params, plots=True)

    with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
        sim.run(t_final)

    # Generate decoders and a basic decoding for comparison
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    decoders_bio_old, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio_old = np.dot(act_bio, decoders_bio_old)
    rmse_bio_old = rmse(sim.data[probe_direct], xhat_bio_old)

    signal = 'equalpower'
    max_freq = 5.0
    signal_seed = 123
    stim.output = lambda t: equalpower(t, dt_nengo, t_final, max_freq, dim,
                         mean=0.0, std=1.0, seed=signal_seed)
    cutoff_mixed = 0.3
    cutoff_compare = 0.3
    with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
        sim.run(t_final)

    # Generate a new decoding using the old decoders and new activities
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio_mixed = np.dot(act_bio, decoders_bio_old)
    rmse_bio_mixed = rmse(sim.data[probe_direct], xhat_bio_mixed)
    # Generate a new decoding using the new decoders and new activities
    decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio_new = np.dot(act_bio, decoders_bio_new)
    rmse_bio_new = rmse(sim.data[probe_direct], xhat_bio_new)
    # Calculate the RMSE of the mixed and new decoding
    rmse_bio_compare = rmse(xhat_bio_mixed, xhat_bio_new)

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio_old,
             label='old, rmse=%.5f' % rmse_bio_old)
    plt.plot(sim.trange(), xhat_bio_mixed,
             label='mixed, rmse=%.5f' % rmse_bio_mixed)
    plt.plot(sim.trange(), xhat_bio_new,
             label='new, rmse=%.5f' % rmse_bio_new)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}

    assert rmse_bio_mixed < cutoff_mixed
    assert rmse_bio_compare < cutoff_compare    










# @lru_cache(maxsize=None)
# def sim_feedforward_spike_matching(Simulator, plt, pre_neurons, bio_neurons,
#     tau_nengo, tau_neuron, dt_nengo, min_rate, max_rate, pre_seed, bio_seed,
#     conn_seed, t_final, dim, n_syn, signal, max_freq=5, signal_seed=234):

#     # Evolutionary Parameters
#     n_processes = 10
#     popsize = 10
#     generations = 20
#     delta_w = 2e-3
#     evo_seed = 9
#     evo_t_final = 1.0
#     evo_signal = 'white_noise'
#     if evo_signal == 'white_noise':
#         evo_max_freq = 5.0
#         evo_signal_seed = 234
#     evo_cutoff = 10.0

#     try:  # use saved weights
#         weights_bio = np.load('weights/weights_bio_feedforward.npz')
#         weights_bio = weights_bio['weights_bio']
#     except:
#         feedforward(
#             Simulator, plt, pre_neurons, bio_neurons, tau_nengo, tau_neuron,
#             dt_nengo, min_rate, max_rate, evo_signal, pre_seed, bio_seed,
#             conn_seed, evo_t_final, dim, n_syn, evo_cutoff, n_processes,
#             popsize, generations, delta_w, evo_seed,
#             plots=True, max_freq=evo_max_freq, signal_seed=evo_signal_seed)
#         weights_bio = np.load('weights/weights_bio_feedforward.npz')
#         weights_bio = weights_bio['weights_bio']

#     with nengo.Network() as model:
#         """
#         Simulate a feedforward network [stim]-[LIF]-[BIO]
#         and compare to [stim]-[LIF]-[LIF].
#         """

#         if signal == 'prime_sinusoids':
#             stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
#         elif signal == 'step_input':
#             stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
#         elif signal == 'white_noise':
#             stim = nengo.Node(lambda t: equalpower(
#                                   t, dt_nengo, t_final, max_freq, dim,
#                                   mean=0.0, std=1.0, seed=signal_seed))

#         pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                              seed=pre_seed, neuron_type=nengo.LIF())
#         bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                              seed=bio_seed, neuron_type=BahlNeuron())
#         lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                              neuron_type=nengo.LIF(), seed=bio_seed,
#                              max_rates=nengo.dists.Uniform(min_rate, max_rate))
#         direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                 neuron_type=nengo.Direct())
#         temp = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                 neuron_type=nengo.Direct())

#         trained_solver = TrainedSolver(weights_bio = weights_bio)
#         nengo.Connection(stim, pre, synapse=None)
#         nengo.Connection(pre, lif, synapse=tau_nengo)
#         nengo.Connection(pre, bio,
#                          seed=conn_seed,
#                          synapse=tau_neuron,
#                          trained_weights=True,
#                          solver=trained_solver)
#         nengo.Connection(stim, direct, synapse=tau_nengo)

#         probe_pre = nengo.Probe(pre, synapse=tau_nengo)
#         probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#         probe_lif = nengo.Probe(lif)
#         probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
#         probe_direct = nengo.Probe(direct, synapse=tau_nengo)

#     with Simulator(model, dt=dt_nengo) as sim:
#         sim.run(t_final)

#     # Generate decoders and a basic decoding for comparison
#     lpf = nengo.Lowpass(tau_nengo)
#     solver = nengo.solvers.LstsqL2(reg=0.01)
#     act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#     decoders_bio_old, info = solver(act_bio, sim.data[probe_direct])
#     xhat_bio_old = np.dot(act_bio, decoders_bio_old)
#     rmse_bio_old = rmse(sim.data[probe_direct], xhat_bio_old)
#     # rmse_lif = rmse(sim.data[probe_direct], sim.data[probe_lif])

#     return decoders_bio_old, xhat_bio_old, rmse_bio_old, weights_bio


# def test_new_LIF_old_decoders(Simulator, plt):
#     """
#     Change the LIF input (seed) but decode with original decoders
#     This tests the generalizability of decoders for novel bio activities

#     test passes if:
#         - RMSE (xhat_bio_mixed, xhat_bio_new) < $cutoff$
#         - rmse_mixed < cutoff
#     """
#     pre_neurons = 100
#     bio_neurons = 20
#     tau_nengo = 0.01
#     tau_neuron = 0.01
#     dt_nengo = 0.001
#     min_rate = 150
#     max_rate = 200
#     pre_seed = 3
#     bio_seed = 6
#     conn_seed = 9
#     t_final = 1.0
#     dim = 2
#     n_syn = 1
#     # signal = 'prime_sinusoids'
#     signal = 'prime_sinusoids'
#     # if signal == 'white_noise':
#     max_freq = 5.0
#     signal_seed = 234

#     # Generate weights with old pre_seed
#     decoders_bio_old, xhat_bio_old, rmse_bio_old, weights_bio = \
#         sim_feedforward_spike_matching(Simulator, plt, pre_neurons, bio_neurons,
#             tau_nengo, tau_neuron, dt_nengo, min_rate, max_rate, pre_seed, bio_seed,
#             conn_seed, 0.001, dim, n_syn, signal, max_freq=5, signal_seed=234)

#     pre_seed = 9
#     cutoff_mixed = 0.3
#     cutoff_compare = 0.3
#     decoders_bio_old, xhat_bio_old, rmse_bio_old, weights_bio = \
#         sim_feedforward_spike_matching(Simulator, plt, pre_neurons, bio_neurons,
#             tau_nengo, tau_neuron, dt_nengo, min_rate, max_rate, pre_seed, bio_seed,
#             conn_seed, t_final, dim, n_syn, signal, max_freq=5, signal_seed=234)

#     with nengo.Network() as model:
#         """
#         Simulate a feedforward network [stim]-[LIF]-[BIO]
#         and compare to [stim]-[LIF]-[LIF].
#         """

#         if signal == 'prime_sinusoids':
#             stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
#         elif signal == 'step_input':
#             stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
#         elif signal == 'white_noise':
#             stim = nengo.Node(lambda t: equalpower(
#                                   t, dt_nengo, t_final, max_freq, dim,
#                                   mean=0.0, std=1.0, seed=signal_seed))

#         pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                              seed=pre_seed, neuron_type=nengo.LIF())
#         bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                              seed=bio_seed, neuron_type=BahlNeuron())
#         lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                              neuron_type=nengo.LIF(), seed=bio_seed,
#                              max_rates=nengo.dists.Uniform(min_rate, max_rate))
#         direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                 neuron_type=nengo.Direct())

#         trained_solver = TrainedSolver(weights_bio = weights_bio)
#         nengo.Connection(stim, pre, synapse=None)
#         nengo.Connection(pre, lif, synapse=tau_nengo)
#         nengo.Connection(pre, bio,
#                          seed=conn_seed,
#                          synapse=tau_neuron,
#                          trained_weights=True,
#                          solver=trained_solver)
#         nengo.Connection(stim, direct, synapse=tau_nengo)

#         probe_pre = nengo.Probe(pre, synapse=tau_nengo)
#         probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#         probe_lif = nengo.Probe(lif)
#         probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
#         probe_direct = nengo.Probe(direct, synapse=tau_nengo)

#     with Simulator(model, dt=dt_nengo) as sim:
#         sim.run(t_final)

#     # Generate a new decoding using the old decoders and new activities
#     lpf = nengo.Lowpass(tau_nengo)
#     solver = nengo.solvers.LstsqL2(reg=0.01)
#     act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#     xhat_bio_mixed = np.dot(act_bio, decoders_bio_old)
#     rmse_bio_mixed = rmse(sim.data[probe_direct], xhat_bio_mixed)
#     # Generate a new decoding using the new decoders and new activities
#     decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
#     xhat_bio_new = np.dot(act_bio, decoders_bio_new)
#     rmse_bio_new = rmse(sim.data[probe_direct], xhat_bio_new)
#     # Calculate the RMSE of the mixed and new decoding
#     rmse_bio_compare = rmse(xhat_bio_mixed, xhat_bio_new)

#     plt.subplot(1, 1, 1)
#     plt.plot(sim.trange(), xhat_bio_old,
#              label='old, rmse=%.5f' % rmse_bio_old)
#     plt.plot(sim.trange(), xhat_bio_mixed,
#              label='mixed, rmse=%.5f' % rmse_bio_mixed)
#     plt.plot(sim.trange(), xhat_bio_new,
#              label='new, rmse=%.5f' % rmse_bio_new)
#     plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
#     plt.xlabel('time (s)')
#     plt.ylabel('$\hat{x}(t)$')
#     plt.title('decode')
#     plt.legend()  # prop={'size':8}

#     assert rmse_bio_mixed < cutoff_mixed
#     assert rmse_bio_compare < cutoff_compare


# def test_new_signal_old_decoders(Simulator, plt):
#     """
#     Change the signal type (step_input) but decode with original
#     decoders. This tests the generalizability of decoders for novel
#     activities and targets.

#     test passes if:
#         - RMSE (xhat_bio_mixed, xhat_bio_new) < $cutoff$
#         - rmse_mixed < cutoff
#     """
#     pre_neurons = 100
#     bio_neurons = 20
#     tau_nengo = 0.01
#     tau_neuron = 0.01
#     dt_nengo = 0.001
#     min_rate = 150
#     max_rate = 200
#     pre_seed = 3
#     bio_seed = 6
#     conn_seed = 9
#     t_final = 1.0
#     dim = 2
#     n_syn = 1
#     signal = 'white_noise'
#     if signal == 'white_noise':
#         max_freq = 5.0
#         signal_seed = 123

#     cutoff_mixed = 0.3
#     cutoff_compare = 0.3
#     decoders_bio_old, xhat_bio_old, rmse_bio_old, weights_bio = \
#         sim_feedforward_spike_matching(Simulator, plt, pre_neurons, bio_neurons,
#             tau_nengo, tau_neuron, dt_nengo, min_rate, max_rate, pre_seed, bio_seed,
#             conn_seed, t_final, dim, n_syn, signal, max_freq=5, signal_seed=234)

#     with nengo.Network() as model:
#         """
#         Simulate a feedforward network [stim]-[LIF]-[BIO]
#         and compare to [stim]-[LIF]-[LIF].
#         """

#         if signal == 'prime_sinusoids':
#             stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
#         elif signal == 'step_input':
#             stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
#         elif signal == 'white_noise':
#             stim = nengo.Node(lambda t: equalpower(
#                                   t, dt_nengo, t_final, max_freq, dim,
#                                   mean=0.0, std=1.0, seed=signal_seed))

#         pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                              seed=pre_seed, neuron_type=nengo.LIF())
#         bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                              seed=bio_seed, neuron_type=BahlNeuron())
#         lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                              neuron_type=nengo.LIF(), seed=bio_seed,
#                              max_rates=nengo.dists.Uniform(min_rate, max_rate))
#         direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                 neuron_type=nengo.Direct())

#         trained_solver = TrainedSolver(weights_bio = weights_bio)
#         nengo.Connection(stim, pre, synapse=None)
#         nengo.Connection(pre, lif, synapse=tau_nengo)
#         nengo.Connection(pre, bio,
#                          seed=conn_seed,
#                          synapse=tau_neuron,
#                          trained_weights=True,
#                          solver=trained_solver)
#         nengo.Connection(stim, direct, synapse=tau_nengo)

#         probe_pre = nengo.Probe(pre, synapse=tau_nengo)
#         probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#         probe_lif = nengo.Probe(lif)
#         probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
#         probe_direct = nengo.Probe(direct, synapse=tau_nengo)

#     with Simulator(model, dt=dt_nengo) as sim:
#         sim.run(t_final)

#     # Generate a new decoding using the old decoders and new activities
#     lpf = nengo.Lowpass(tau_nengo)
#     solver = nengo.solvers.LstsqL2(reg=0.01)
#     act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#     xhat_bio_mixed = np.dot(act_bio, decoders_bio_old)
#     rmse_bio_mixed = rmse(sim.data[probe_direct], xhat_bio_mixed)
#     # Generate a new decoding using the new decoders and new activities
#     decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
#     xhat_bio_new = np.dot(act_bio, decoders_bio_new)
#     rmse_bio_new = rmse(sim.data[probe_direct], xhat_bio_new)
#     # Calculate the RMSE of the mixed and new decoding
#     rmse_bio_compare = rmse(xhat_bio_mixed, xhat_bio_new)

#     plt.subplot(1, 1, 1)
#     plt.plot(sim.trange(), xhat_bio_old,
#              label='old, rmse=%.5f' % rmse_bio_old)
#     plt.plot(sim.trange(), xhat_bio_mixed,
#              label='mixed, rmse=%.5f' % rmse_bio_mixed)
#     plt.plot(sim.trange(), xhat_bio_new,
#              label='new, rmse=%.5f' % rmse_bio_new)
#     plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
#     plt.xlabel('time (s)')
#     plt.ylabel('$\hat{x}(t)$')
#     plt.title('decode')
#     plt.legend()  # prop={'size':8}

#     assert rmse_bio_mixed < cutoff_mixed
#     assert rmse_bio_compare < cutoff_compare


# def test_biosolver_spike_matching(Simulator, plt):
#     """
#     Simulate a network [stim]-[LIF]-[BIO]-[Probe]
#                              -[Direct]
#     decoders_bio: decoders out of [BIO] that are trained
#                   by the oracle method (iterative)
#     """

#     pre_neurons = 100
#     bio_neurons = 20
#     tau_nengo = 0.01
#     tau_neuron = 0.01
#     dt_nengo = 0.001
#     min_rate = 150
#     max_rate = 200
#     pre_seed = 3
#     bio_seed = 6
#     conn_seed = 9
#     t_final = 1.0
#     dim = 2
#     n_syn = 1
#     signal = 'white_noise'
#     if signal == 'white_noise':
#         max_freq = 5.0
#         signal_seed = 123

#     cutoff = 0.3
#     cutoff_old_vs_new_decoding = 0.3

#     decoders_bio_old, xhat_bio_old, rmse_bio_old, weights_bio = \
#         sim_feedforward_spike_matching(Simulator, plt, pre_neurons, bio_neurons,
#             tau_nengo, tau_neuron, dt_nengo, min_rate, max_rate, pre_seed, bio_seed,
#             conn_seed, t_final, dim, n_syn, signal, max_freq=5, signal_seed=234)

#     def sim(w_train, weights_bio, decoders_bio=None, plots=False):

#         if decoders_bio is None:
#             decoders_bio = np.zeros((bio_neurons, dim))

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
#             direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                     neuron_type=nengo.Direct())

#             trained_solver = TrainedSolver(weights_bio = weights_bio)
#             bio_solver = BioSolver(decoders_bio = decoders_bio)
#             nengo.Connection(stim, pre, synapse=None)
#             nengo.Connection(pre, lif, synapse=tau_nengo)
#             nengo.Connection(pre, bio,
#                              seed=conn_seed,
#                              synapse=tau_neuron,
#                              trained_weights=True,
#                              solver=trained_solver)
#             nengo.Connection(stim, direct, synapse=tau_nengo)

#             probe_pre = nengo.Probe(pre, synapse=tau_nengo)
#             probe_bio = nengo.Probe(bio, synapse=tau_neuron, solver=bio_solver)
#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_lif = nengo.Probe(lif)
#             probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
#             probe_direct = nengo.Probe(direct, synapse=tau_nengo)

#         with Simulator(model, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         assert np.sum(act_bio) > 0.0
#         solver = nengo.solvers.LstsqL2(reg=0.1)
#         decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
#         xhat_bio = np.dot(act_bio, decoders_bio_new)
#         rmse_bio = rmse(xhat_bio, sim.data[probe_direct])
#         rmse_old_vs_new_decoding = rmse(xhat_bio, sim.data[probe_bio])

#         if plots:
#             plt.subplot(1, 1, 1)
#             plt.plot(sim.trange(), sim.data[probe_bio],
#                      label='[STIM]-[LIF]-[BIO]-[probe]')
#             plt.plot(sim.trange(), xhat_bio,
#                      label='[STIM]-[LIF]-[BIO], rmse=%.5f' % rmse_bio)
#             plt.plot(sim.trange(), sim.data[probe_direct],
#                      label='[STIM]-[LIF]-[LIF_EQ]-[Direct]')
#             plt.xlabel('time (s)')
#             plt.ylabel('$\hat{x}(t)$')
#             plt.title('decode')
#             plt.legend()  # prop={'size':8}

#         return decoders_bio_new, rmse_bio, rmse_old_vs_new_decoding

#     decoders_bio_new, _, _ = sim(w_train=1.0,
#                                  weights_bio=weights_bio,
#                                  decoders_bio=None)
#     _, rmse_bio, rmse_old_vs_new_decoding = sim(w_train=0.0,
#                                                 weights_bio=weights_bio, 
#                                                 decoders_bio=decoders_bio_new,
#                                                 plots=True)

#     assert rmse_bio < cutoff
#     assert rmse_old_vs_new_decoding < cutoff_old_vs_new_decoding
