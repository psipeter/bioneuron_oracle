from functools32 import lru_cache

import numpy as np

import nengo
from nengo.utils.numpy import rmse

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input,
                              equalpower, OracleSolver, TrainedSolver,
                              spike_match_train)

# Nengo Parameters
pre_neurons = 100
bio_neurons = 50
post_neurons = 50
tau_nengo = 0.01
tau_neuron = 0.01
dt_nengo = 0.001
min_rate = 150
max_rate = 200
pre_seed = 3
bio_seed = 6
post_seed = 12
conn_seed = 9
t_final = 1.0
dim = 2
n_syn = 1
signal = 'prime_sinusoids'

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
}

def test_transform_in(Simulator, plt):
    """
    Simulate a feedforward transformation into a bioensemble
    """

    dim = 1
    transform = -0.5
    cutoff = 0.3
    bio_neurons = 50
    evo_params['generations'] = 10

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
        nengo.Connection(pre, lif, synapse=tau_nengo, transform=transform)
        nengo.Connection(pre, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         transform=transform,
                         trained_weights=True,
                         solver = trained_solver,
                         n_syn=n_syn)
        nengo.Connection(stim, direct, synapse=tau_nengo, transform=transform)
        conn_ideal_out = nengo.Connection(lif, temp, synapse=tau_nengo,
                         solver=nengo.solvers.LstsqL2())

        probe_pre = nengo.Probe(pre, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif)
        probe_direct = nengo.Probe(direct, synapse=None)
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
    act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    decoders_lif, info = solver(act_lif, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)
    xhat_lif = np.dot(act_lif, decoders_lif)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    rmse_lif = rmse(sim.data[probe_direct], xhat_lif)

    plt.subplot(1, 1, 1)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff


def test_slice_post(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.3

    with nengo.Network() as network:

        stim1 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim/2,
                                neuron_type=nengo.Direct())

        trained_solver1 = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, lif1.n_neurons, n_syn)))
        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(lif1, bio[0],
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver1,
                         n_syn=n_syn)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=None)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    network = spike_match_train(network, method="1-N",
                                params=evo_params, plots=True)

    with Simulator(network, dt=dt_nengo) as sim:
        sim.run(t_final)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)

    plt.subplot(1, 1, 1)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff


def test_slice_pre_slice_post(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.3

    with nengo.Network() as network:

        stim1 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[0:dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                              seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim/2,
                                neuron_type=nengo.Direct())

        trained_solver1 = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, lif1.n_neurons, n_syn)))
        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(lif1[0], bio[0],
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver1,
                         n_syn=n_syn)
        nengo.Connection(stim1[0], direct[0], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=None)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    network = spike_match_train(network, method="1-N",
                                params=evo_params, plots=True)

    with Simulator(network, dt=dt_nengo) as sim:
        sim.run(t_final)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)

    plt.subplot(1, 1, 1)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff

def test_two_inputs_two_dims(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][1]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.3

    with nengo.Network() as network:

        stim1 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])
        stim2 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=2*pre_neurons, dimensions=dim/2,
                              seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        trained_solver1 = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, lif1.n_neurons, n_syn)))
        trained_solver2 = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, lif2.n_neurons, n_syn)))
        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(stim2, lif2, synapse=None)
        nengo.Connection(lif1, bio[0],
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver1,
                         n_syn=n_syn)
        nengo.Connection(lif2, bio[1],
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver2,
                         n_syn=n_syn)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)
        nengo.Connection(stim2, direct[1], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=None)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    network = spike_match_train(network, method="1-N",
                                params=evo_params, plots=True)

    with Simulator(network, dt=dt_nengo) as sim:
        sim.run(t_final)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)

    plt.subplot(1, 1, 1)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}

    assert rmse_bio < cutoff


def test_two_inputs_one_dim(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.3
    dim = 1
    max_freq = 5
    signal_seed  = 123
    bio_neurons = 50
    evo_params['generations'] = 200

    with nengo.Network() as network:

        stim1 = nengo.Node(
            lambda t: 0.5 * prime_sinusoids(t, dim, t_final) * (
                (t < t_final / 3) or (t > t_final * 2 / 3)))
        stim2 = nengo.Node(
            lambda t: 0.5 * equalpower(t, dt_nengo, t_final, max_freq, dim,
                                  mean=0.0, std=1.0, seed=signal_seed) * (
                (t < t_final / 3) or (t < t_final * 2 / 3)))

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                              seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=2*pre_neurons, dimensions=dim,
                              seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        trained_solver1 = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, lif1.n_neurons, n_syn)))
        trained_solver2 = TrainedSolver(
            weights_bio = np.zeros((bio.n_neurons, lif2.n_neurons, n_syn)))
        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(stim2, lif2, synapse=None)
        nengo.Connection(lif1, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver1,
                         n_syn=n_syn)
        nengo.Connection(lif2, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver2,
                         n_syn=n_syn)
        nengo.Connection(stim1, direct, synapse=tau_nengo)
        nengo.Connection(stim2, direct, synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=None)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    network = spike_match_train(network, method="1-N",
                                params=evo_params, plots=True)

    with Simulator(network, dt=dt_nengo) as sim:
        sim.run(t_final)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)

    plt.subplot(1, 1, 1)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff


def test_slice_out(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-[LIF1]
                                         -[LIF2]
                             -[Direct]-[direct2]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

    cutoff = 0.3

    def sim(w_train=1, weights_bio_in=None, decoders_bio_in=None, plots=False):

        if decoders_bio_in is None:
            decoders_bio = np.zeros((bio_neurons, dim))
        else:
        	decoders_bio = decoders_bio_in
        if weights_bio_in is None:
        	weights_bio = np.zeros((bio_neurons, pre_neurons, n_syn))
        else:
        	weights_bio = weights_bio_in

        with nengo.Network() as network:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            lif1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=post_seed, neuron_type=nengo.LIF())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=2*post_seed, neuron_type=nengo.LIF())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            oracle_solver1 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            oracle_solver2 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 1])
            trained_solver = TrainedSolver(weights_bio = weights_bio)

            nengo.Connection(stim, lif, synapse=None)
            # nengo.Connection(lif, bio, synapse=tau_neuron,
            #                  weights_bias_conn=True)
            conn1 = nengo.Connection(lif, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver = trained_solver,
                             n_syn=n_syn)
            nengo.Connection(bio[0], lif1, synapse=tau_nengo,
                             solver=oracle_solver1)
            nengo.Connection(bio[1], lif2, synapse=tau_nengo,
                             solver=oracle_solver2)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct2,
            				 synapse=tau_nengo)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=oracle_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=None)

        if weights_bio_in is None:  # don't retrain after the first simulation
            network = spike_match_train(network, method="1-N",
                                        params=evo_params, plots=True)

        with Simulator(network, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct2])
        weights_bio_new = conn1.solver.weights_bio
        # rmse_bio = rmse(sim.data[probe_bio], sim.data[probe_direct2][:,0])
        rmse_lif1 = rmse(sim.data[probe_lif1][:, 0],
                         sim.data[probe_direct2][:, 0])
        rmse_lif2 = rmse(sim.data[probe_lif2][:, 0],
                         sim.data[probe_direct2][:, 1])

        if plots:
            plt.subplot(1, 1, 1)
            plt.plot(sim.trange(), sim.data[probe_bio],
                     label='Bio[0] (probe)')
            plt.plot(sim.trange(), sim.data[probe_lif1][:, 0],
                     label='LIF1, rmse=%.5f' % rmse_lif1)
            plt.plot(sim.trange(), sim.data[probe_lif2][:, 0],
                     label='LIF2, rmse=%.5f' % rmse_lif2)
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 0],
                     label='oracle2[0]')
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 1],
                     label='oracle2[1]')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

        return decoders_bio_new, weights_bio_new, rmse_lif1, rmse_lif2

    decoders_bio_new, weights_bio_new, rmse_lif01, rmse_lif02 = sim(
    	w_train=1, weights_bio_in=None, decoders_bio_in=None)
    decoders_bio_new2, weights_bio_new2, rmse_lif1, rmse_lif2 = sim(
    	w_train=0, weights_bio_in=weights_bio_new,
    	decoders_bio_in=decoders_bio_new, plots=True)

    assert rmse_lif1 < cutoff
    assert rmse_lif2 < cutoff


def test_transform_out(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-(transform)-[BIO2]
                             -[Direct]-(transform)-[direct2]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    """

    cutoff = 0.3
    transform = -0.5
    evo_params['popsize'] = 10
    evo_params['generations'] = 200
    pre_neurons = 20
    bio_neurons = 20

    def sim(w_train=1, weights_bio_in=None, decoders_bio_in=None, plots=False):

        if decoders_bio_in is None:
            decoders_bio = np.zeros((bio_neurons, dim))
        else:
        	decoders_bio = decoders_bio_in
        if weights_bio_in is None:
        	weights_bio = np.zeros((bio_neurons, pre_neurons, n_syn))
        else:
        	weights_bio = weights_bio_in

        with nengo.Network(seed=1) as network:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim,
                                  seed=post_seed, neuron_type=nengo.LIF())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            oracle_solver = OracleSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
            trained_solver = TrainedSolver(weights_bio = weights_bio)

            nengo.Connection(stim, lif, synapse=None)
            # nengo.Connection(lif, bio, synapse=tau_neuron,
            #                  weights_bias_conn=True)
            conn1 = nengo.Connection(lif, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver = trained_solver,
                             n_syn=n_syn)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(bio, lif2, synapse=tau_nengo,
                             transform=transform, solver=oracle_solver)
            nengo.Connection(direct, direct2, synapse=tau_nengo,
                             transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=None)
            probe_direct2 = nengo.Probe(direct2, synapse=None)

        if weights_bio_in is None:  # don't retrain after the first simulation
            network = spike_match_train(network, method="1-N",
                                        params=evo_params, plots=True)

        with Simulator(network, dt=dt_nengo, seed=1) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
        weights_bio_new = conn1.solver.weights_bio
        rmse_lif = rmse(sim.data[probe_lif2], sim.data[probe_direct2])

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],
            #          label='BIO (probe), rmse=%.5f' % rmse_bio)
            # plt.plot(sim.trange(), xhat_bio,
            #          label='BIO (xhat), rmse=%.5f' % rmse_bio_xhat)
            # plt.plot(sim.trange(), sim.data[probe_direct],
            #          label='oracle')
            plt.plot(sim.trange(), sim.data[probe_lif2],
                     label='LIF2, rmse=%.5f' % rmse_lif)
            plt.plot(sim.trange(), sim.data[probe_direct2],
                     label='oracle2')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

        return decoders_bio_new, weights_bio_new, rmse_lif

    decoders_bio_new, weights_bio_new, rmse_lif = sim(
    	w_train=1, weights_bio_in=None, decoders_bio_in=None)
    decoders_bio_new2, weights_bio_new2, rmse_lif= sim(
    	w_train=0, weights_bio_in=weights_bio_new,
    	decoders_bio_in=decoders_bio_new, plots=True)

    assert rmse_lif < cutoff


def test_slice_and_transform_out(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-[LIF1]
                                         -[LIF2]
                             -[Direct]-[direct2]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

    cutoff = 0.3
    transform = -0.5

    def sim(w_train=1, weights_bio_in=None, decoders_bio_in=None, plots=False):

        if decoders_bio_in is None:
            decoders_bio = np.zeros((bio_neurons, dim))
        else:
        	decoders_bio = decoders_bio_in
        if weights_bio_in is None:
        	weights_bio = np.zeros((bio_neurons, pre_neurons, n_syn))
        else:
        	weights_bio = weights_bio_in

        with nengo.Network() as network:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            lif1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=post_seed, neuron_type=nengo.LIF())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=2*post_seed, neuron_type=nengo.LIF())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            oracle_solver1 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            oracle_solver2 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 1])
            trained_solver = TrainedSolver(weights_bio = weights_bio)

            nengo.Connection(stim, lif, synapse=None)
            # nengo.Connection(lif, bio, synapse=tau_neuron,
            #                  weights_bias_conn=True)
            conn1 = nengo.Connection(lif, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver = trained_solver,
                             n_syn=n_syn)
            nengo.Connection(bio[0], lif1, synapse=tau_nengo,
                             solver=oracle_solver1, transform=transform)
            nengo.Connection(bio[1], lif2, synapse=tau_nengo,
                             solver=oracle_solver2, transform=transform)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct2,
            				 synapse=tau_nengo, transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=oracle_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=None)

        if weights_bio_in is None:  # don't retrain after the first simulation
            network = spike_match_train(network, method="1-N",
                                        params=evo_params, plots=True)

        with Simulator(network, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct2])
        weights_bio_new = conn1.solver.weights_bio
        # rmse_bio = rmse(sim.data[probe_bio], sim.data[probe_direct2][:,0])
        rmse_lif1 = rmse(sim.data[probe_lif1][:, 0],
                         sim.data[probe_direct2][:, 0])
        rmse_lif2 = rmse(sim.data[probe_lif2][:, 0],
                         sim.data[probe_direct2][:, 1])

        if plots:
            plt.subplot(1, 1, 1)
            plt.plot(sim.trange(), sim.data[probe_bio],
                     label='Bio[0] (probe)')
            plt.plot(sim.trange(), sim.data[probe_lif1][:, 0],
                     label='LIF1, rmse=%.5f' % rmse_lif1)
            plt.plot(sim.trange(), sim.data[probe_lif2][:, 0],
                     label='LIF2, rmse=%.5f' % rmse_lif2)
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 0],
                     label='oracle2[0]')
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 1],
                     label='oracle2[1]')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

        return decoders_bio_new, weights_bio_new, rmse_lif1, rmse_lif2

    decoders_bio_new, weights_bio_new, rmse_lif01, rmse_lif02 = sim(
    	w_train=1, weights_bio_in=None, decoders_bio_in=None)
    decoders_bio_new2, weights_bio_new2, rmse_lif1, rmse_lif2 = sim(
    	w_train=0, weights_bio_in=weights_bio_new,
    	decoders_bio_in=decoders_bio_new, plots=True)

    assert rmse_lif1 < cutoff
    assert rmse_lif2 < cutoff

def test_bio_to_bio(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-[BIO2]
                             -[Direct]-[direct2]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

    dim = 1
    evo_params['popsize'] = 10  # 50 pop 50 evals 20 neurons = 18k sec
    evo_params['generations'] = 200  # 50 pop 50 evals 20 neurons = 18k sec
    bio_neurons = 100
    post_neurons = 100
    cutoff = 0.3
    transform = -0.5

    def sim(weights_bio1_in=None, weights_bio2_in=None, plots=False):

        if weights_bio1_in is None:
            weights_bio1 = np.zeros((bio_neurons, pre_neurons, n_syn))
        else:
            weights_bio1 = weights_bio1_in
        if weights_bio2_in is None:
            weights_bio2 = np.zeros((bio_neurons, bio_neurons, n_syn))
        else:
            weights_bio2 = weights_bio2_in

        with nengo.Network() as network:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            pre1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
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
            bio2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim,
                                 seed=post_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            lif2 = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            trained_solver1 = TrainedSolver(weights_bio = weights_bio1)
            trained_solver2 = TrainedSolver(weights_bio = weights_bio2)

            nengo.Connection(stim, pre1, synapse=None)
            nengo.Connection(pre1, lif, synapse=tau_nengo)
            conn1 = nengo.Connection(pre1, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver = trained_solver1,
                             n_syn=n_syn)
            nengo.Connection(pre1, direct, synapse=tau_nengo)
            conn2 = nengo.Connection(bio, bio2,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver = trained_solver2,
                             n_syn=n_syn,
                             transform = transform)
            nengo.Connection(lif, lif2,
                             synapse=tau_nengo, transform=transform)
            nengo.Connection(direct, direct2,
                             synapse=tau_nengo, transform=transform)

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_lif2_spikes = nengo.Probe(lif2.neurons, 'spikes')
            probe_direct = nengo.Probe(direct, synapse=None)
            probe_direct2 = nengo.Probe(direct2, synapse=None)

        if weights_bio1_in is None or weights_bio2_in is None:
            print 'training weights...'
            network = spike_match_train(network, method="1-N",
                                        params=evo_params, plots=True)

        with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
        act_lif2 = lpf.filt(sim.data[probe_lif2_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        assert np.sum(act_bio2) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
        decoders_bio_new2, info2 = solver(act_bio2, sim.data[probe_direct2])
        decoders_lif_new, info = solver(act_lif, sim.data[probe_direct])
        decoders_lif_new2, info2 = solver(act_lif2, sim.data[probe_direct2])
        weights_bio1_new = conn1.solver.weights_bio
        weights_bio2_new = conn2.solver.weights_bio
        xhat_bio = np.dot(act_bio, decoders_bio_new)
        xhat_bio2 = np.dot(act_bio2, decoders_bio_new2)
        xhat_lif = np.dot(act_lif, decoders_lif_new)
        xhat_lif2 = np.dot(act_lif2, decoders_lif_new2)
        rmse_bio = rmse(xhat_bio, sim.data[probe_direct])
        rmse_bio2 = rmse(xhat_bio2, sim.data[probe_direct2])
        rmse_lif = rmse(xhat_lif, sim.data[probe_direct])
        rmse_lif2 = rmse(xhat_lif2, sim.data[probe_direct2])

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],
            #          label='BIO (probe)')
            # plt.plot(sim.trange(), xhat_bio,
            #          label='BIO, rmse=%.5f' % rmse_bio)
            # plt.plot(sim.trange(), xhat_lif,
            #          label='lif, rmse=%.5f' % rmse_lif)
            plt.plot(sim.trange(), xhat_bio2,
                     label='bio 2, rmse=%.5f' % rmse_bio2)
            plt.plot(sim.trange(), xhat_lif2,
                     label='lif 2, rmse=%.5f' % rmse_lif2)
            # plt.plot(sim.trange(), sim.data[probe_direct],
            #          label='oracle')
            plt.plot(sim.trange(), sim.data[probe_direct2],
                     label='oracle 2')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

        return weights_bio1_new, weights_bio2_new, rmse_bio, rmse_bio2

    weights_bio1_new, weights_bio2_new, rmse_lif1, rmse_lif2 = sim(
    	weights_bio1_in=None, weights_bio2_in=None, plots=True)
    # weights_bio1_new2, weights_bio2_new2, rmse_lif1, rmse_lif2 = sim(
    # 	weights_bio1_in=weights_bio1_new, weights_bio2_in=weights_bio2_new,
    # 	plots=True)

    # assert rmse_lif1 < cutoff
    assert rmse_lif2 < cutoff