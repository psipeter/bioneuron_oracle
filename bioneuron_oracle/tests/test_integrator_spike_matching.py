import numpy as np

import nengo
from nengo.utils.numpy import rmse

from nengolib.signal import s

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input,
                              equalpower, OracleSolver, TrainedSolver,
                              spike_match_train)

# Nengo Parameters
pre_neurons = 100
bio_neurons = 100
tau_nengo = 0.1
tau_neuron = 0.1
dt_nengo = 0.001
min_rate = 150
max_rate = 200
pre_seed = 3
bio_seed = 6
conn_seed = 9
network_seed = 12
sim_seed = 15
post_seed = 18
dim = 1
n_syn = 1
jl_dims = 0
# signal = 'white_noise'
max_freq = 5
signal_seed = 123

# Evolutionary Parameters
evo_params = {
    'dt_nengo': 0.001,
    'tau_nengo': 0.05,
    'n_processes': 10,
    'popsize': 10,
    'generations' :100,
    'w_0': 1e-3,
    'delta_w' :1e-4,
    'evo_seed' :9,
    'evo_t_final' :1.0,
    'evo_signal': 'prime_sinusoids',
    'evo_max_freq': 5.0,
    'evo_signal_seed': 234,
    'evo_cutoff' :50.0,
    'sim_seed': 15,
}


def test_integrator(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-[BIO]
                                   -[LIF]-[LIF]
                             -[Direct]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

    dim = 1
    transform = -0.5
    cutoff = 0.3
    bio_neurons = 100
    evo_params['generations'] = 2 
    signal_train = 'white_noise'
    signal_test = 'white_noise'
    t_test = 1.0
    t_train = 1.0
    train = False
    plots = True

    try:
        w_bio1_0 = np.load('weights/w_integrator_pre_to_bio.npz')['weights_bio']
        w_bio2_0 = np.load('weights/w_integrator_bio_to_bio.npz')['weights_bio']
    except IOError:
        w_bio1_0 = np.zeros((bio_neurons, pre_neurons, n_syn))
        w_bio2_0 = np.zeros((bio_neurons, bio_neurons, n_syn))

    with nengo.Network(seed=network_seed) as network:

        if signal_train == 'prime_sinusoids':
            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_train))
        elif signal_train == 'step_input':
            stim = nengo.Node(lambda t: step_input(t, dim, t_train, dt_nengo))
        elif signal_train == 'white_noise':
            stim = nengo.Node(lambda t: equalpower(
                                  t, dt_nengo, t_train, max_freq, dim,
                                  mean=0.0, std=1.0, seed=signal_seed))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                             seed=pre_seed, neuron_type=nengo.LIF())
        # pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
        #                      seed=pre_seed, neuron_type=nengo.LIF(),
        #                      max_rates=nengo.dists.Uniform(50, 100),
        #                      intercepts=nengo.dists.Uniform(-1.0, 1.0))
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                             seed=bio.seed, neuron_type=nengo.LIF(),
                             max_rates=bio.max_rates, intercepts=bio.intercepts)
        integral = nengo.Node(size_in=dim)

        trained_solver1 = TrainedSolver(weights_bio = w_bio1_0)
        trained_solver2 = TrainedSolver(weights_bio = w_bio2_0)

        # if jl_dims > 0:
        #     # TODO: magnitude should scale with n_neurons (maybe 1./n^2)?
        #     jl_decoders = np.random.RandomState(seed=333).randn(
        #         bio_neurons, jl_dims) * 1e-4
        #     oracle_solver.decoders_bio = np.hstack(
        #         (oracle_solver.decoders_bio, jl_decoders))

        nengo.Connection(stim, pre, synapse=None)
        # connect feedforward to non-jl_dims of bio
        pre_to_bio = nengo.Connection(pre, bio,  # [:dim]
                                 seed=conn_seed,
                                 synapse=tau_neuron,
                                 trained_weights=True,
                                 solver = trained_solver1,
                                 n_syn=n_syn,
                                 transform=tau_neuron)
        # nengo.Connection(lif, bio[:dim], weights_bias_conn=True,
        #                  synapse=tau_neuron, transform=tau_neuron)
        nengo.Connection(pre, lif, synapse=tau_nengo, transform=tau_nengo)
        # nengo.Connection(bio, bio, synapse=tau_neuron, solver=oracle_solver)
        bio_to_bio = nengo.Connection(bio, bio,
                                 seed=2*conn_seed,
                                 synapse=tau_neuron,
                                 trained_weights=True,
                                 solver = trained_solver2,
                                 n_syn=n_syn)
        nengo.Connection(lif, lif, synapse=tau_nengo)

        nengo.Connection(stim, integral,
                         synapse=nengo.LinearFilter([1.], [1., 0]))
        # nengo.Connection(integral, pre, synapse=None)
        # nengo.Connection(pre, bio[:dim],  # oracle connection
        #                  synapse=tau_neuron, transform=w_train)

        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
        probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
        probe_integral = nengo.Probe(integral, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif, synapse=tau_nengo)

    # train the weights using spike-matching
    if train:
        network = spike_match_train(network, method="1-N", params=evo_params, plots=True)
        np.savez('weights/w_integrator_pre_to_bio.npz',
               weights_bio=pre_to_bio.solver.weights_bio)
        np.savez('weights/w_integrator_bio_to_bio.npz',
               weights_bio=bio_to_bio.solver.weights_bio)

    # set stim to be different than during spike match training
    # with network:
        # if signal_test == 'prime_sinusoids':
        #     stim.output = lambda t: prime_sinusoids(t, dim, t_test)
        # elif signal_test == 'step_input':
        #     stim.output = lambda t: step_input(t, dim, t_test, dt_nengo)
        # elif signal_test == 'white_noise':
        #     stim.output = lambda t: equalpower(
        #                           t, dt_nengo, t_test, max_freq, dim,
        #                           mean=0.0, std=1.0, seed=2*signal_seed)
    with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_train)

    # compute oracle decoders
    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
    oracle_decoders = oracle_solver(act_bio, sim.data[probe_integral])[0]

    # compute estimate on a new signal
    with network:
        if signal_test == 'prime_sinusoids':
            stim.output = lambda t: prime_sinusoids(t, dim, t_test)
        elif signal_test == 'step_input':
            stim.output = lambda t: step_input(t, dim, t_test, dt_nengo)
        elif signal_test == 'white_noise':
            stim.output = lambda t: equalpower(
                                  t, dt_nengo, t_test, max_freq, dim,
                                  mean=0.0, std=1.0, seed=2*signal_seed)
    with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_test)

    #compute estimates and rmses
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio = np.dot(act_bio, oracle_decoders)
    xhat_lif = sim.data[probe_lif]
    rmse_bio = rmse(sim.data[probe_integral], xhat_bio)
    rmse_lif = rmse(sim.data[probe_integral], xhat_lif)

    if plots:
        plt.subplot(1, 1, 1)
        # plt.plot(sim.trange(), sim.data[probe_bio], label='bio probe')
        plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
        plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
        plt.plot(sim.trange(), sim.data[probe_integral], label='oracle')
        plt.xlabel('time (s)')
        plt.ylabel('$\hat{x}(t)$')
        plt.title('decode')
        plt.legend()  # prop={'size':8}
        assert rmse_bio < cutoff
