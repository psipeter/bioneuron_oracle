from functools32 import lru_cache

import numpy as np

import nengo
from nengo.utils.numpy import rmse

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input,
                              equalpower, BioSolver, TrainedSolver)
from bioneuron_oracle.tests.spike_match_train import spike_match_train


# Nengo Parameters
pre_neurons = 100
bio_neurons = 20
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

def test_transform_in_spike_matching(Simulator, plt):
    """
    Simulate a feedforward transformation into a bioensemble
    """

    transform = -0.5
    cutoff = 0.1

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
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)

    plt.subplot(1, 1, 1)
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff


def test_slice_post_spike_matching(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4

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

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
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
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff


def test_slice_pre_slice_post_spike_matching(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4

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

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
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
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff

def test_two_inputs_two_dims_spike_matching(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][1]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4

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

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
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
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}

    assert rmse_bio < cutoff


def test_two_inputs_one_dim_spike_matching(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4

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
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim/2,
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
        nengo.Connection(lif2, bio[0],
                         seed=conn_seed,
                         synapse=tau_neuron,
                         trained_weights=True,
                         solver = trained_solver2,
                         n_syn=n_syn)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)
        nengo.Connection(stim2, direct[0], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
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
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio < cutoff


def test_slice_out_spike_matching(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-[LIF1]
                                         -[LIF2]
                             -[Direct]-[Direct_out]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

    cutoff = 0.5

    def sim(w_train=1, weights_bio_in=None, decoders_bio_in=None, plots=False):

        if decoders_bio_in is None:
            decoders_bio = np.zeros((bio_neurons, dim))
        else:
        	decoders_bio = decoders_bio_in
        if weights_bio_in is None:
        	weights_bio = np.zeros((bio_neurons, post_neurons, n_syn))
        else:
        	weights_bio = weights_bio_in

        with nengo.Network() as network:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            lif1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=post_seed, neuron_type=nengo.LIF())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=2*post_seed, neuron_type=nengo.LIF())
            direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            bio_solver1 = BioSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            bio_solver2 = BioSolver(
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
                             solver=bio_solver1)
            nengo.Connection(bio[1], lif2, synapse=tau_nengo,
                             solver=bio_solver2)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct_out,
            				 synapse=tau_nengo)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=bio_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

        if weights_bio_in is None:  # don't retrain after the first simulation
            network = spike_match_train(network, method="1-N",
                                        params=evo_params, plots=True)

        with Simulator(network, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])
        weights_bio_new = conn1.solver.weights_bio
        # rmse_bio = rmse(sim.data[probe_bio], sim.data[probe_direct_out][:,0])
        rmse_lif1 = rmse(sim.data[probe_lif1][:, 0],
                         sim.data[probe_direct_out][:, 0])
        rmse_lif2 = rmse(sim.data[probe_lif2][:, 0],
                         sim.data[probe_direct_out][:, 1])

        if plots:
            plt.subplot(1, 1, 1)
            plt.plot(sim.trange(), sim.data[probe_bio],
                     label='[STIM]-[LIF]-[BIO][0]-[probe]')
            plt.plot(sim.trange(), sim.data[probe_lif1][:, 0],
                     label='[STIM]-[LIF]-[BIO]-[LIF1], rmse=%.5f' % rmse_lif1)
            plt.plot(sim.trange(), sim.data[probe_lif2][:, 0],
                     label='[STIM]-[LIF]-[BIO]-[LIF2], rmse=%.5f' % rmse_lif2)
            plt.plot(sim.trange(), sim.data[probe_direct_out][:, 0],
                     label='[STIM]-[LIF]-[LIF_EQ]-[Direct][0]')
            plt.plot(sim.trange(), sim.data[probe_direct_out][:, 1],
                     label='[STIM]-[LIF]-[LIF_EQ]-[Direct][1]')
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


def test_transform_out_spike_matching(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-(transform)-[BIO2]
                             -[Direct]-(transform)-[Direct_Out]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    """

    cutoff = 0.2
    transform = -0.5

    def sim(w_train=1, weights_bio_in=None, decoders_bio_in=None, plots=False):

        if decoders_bio_in is None:
            decoders_bio = np.zeros((bio_neurons, dim))
        else:
        	decoders_bio = decoders_bio_in
        if weights_bio_in is None:
        	weights_bio = np.zeros((bio_neurons, post_neurons, n_syn))
        else:
        	weights_bio = weights_bio_in

        with nengo.Network(seed=1) as network:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim,
                                  seed=post_seed, neuron_type=nengo.LIF())
            direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
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
                             transform=transform, solver=bio_solver)
            nengo.Connection(direct, direct_out, synapse=tau_nengo,
                             transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

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
        rmse_lif = rmse(sim.data[probe_lif2], sim.data[probe_direct_out])

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],
            #          label='BIO probe, rmse=%.5f' % rmse_bio)
            # plt.plot(sim.trange(), xhat_bio,
            #          label='BIO xhat, rmse=%.5f' % rmse_bio_xhat)
            # plt.plot(sim.trange(), sim.data[probe_direct],
            #          label='[STIM]-[Direct]')
            plt.plot(sim.trange(), sim.data[probe_lif2],
                     label='[STIM]-[LIF]-[BIO]-[LIF2], rmse=%.5f' % rmse_lif)
            plt.plot(sim.trange(), sim.data[probe_direct_out],
                     label='[STIM]-[Direct]-[Direct_Out]')
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


def test_slice_and_transform_out_spike_matching(Simulator, plt):
    """
    Simulate a network [stim]-[LIF]-[BIO]-[LIF1]
                                         -[LIF2]
                             -[Direct]-[Direct_out]
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
        	weights_bio = np.zeros((bio_neurons, post_neurons, n_syn))
        else:
        	weights_bio = weights_bio_in

        with nengo.Network() as network:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            lif1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=post_seed, neuron_type=nengo.LIF())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=2*post_seed, neuron_type=nengo.LIF())
            direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            bio_solver1 = BioSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            bio_solver2 = BioSolver(
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
                             solver=bio_solver1, transform=transform)
            nengo.Connection(bio[1], lif2, synapse=tau_nengo,
                             solver=bio_solver2, transform=transform)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct_out,
            				 synapse=tau_nengo, transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=bio_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

        if weights_bio_in is None:  # don't retrain after the first simulation
            network = spike_match_train(network, method="1-N",
                                        params=evo_params, plots=True)

        with Simulator(network, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])
        weights_bio_new = conn1.solver.weights_bio
        # rmse_bio = rmse(sim.data[probe_bio], sim.data[probe_direct_out][:,0])
        rmse_lif1 = rmse(sim.data[probe_lif1][:, 0],
                         sim.data[probe_direct_out][:, 0])
        rmse_lif2 = rmse(sim.data[probe_lif2][:, 0],
                         sim.data[probe_direct_out][:, 1])

        if plots:
            plt.subplot(1, 1, 1)
            plt.plot(sim.trange(), sim.data[probe_bio],
                     label='[STIM]-[LIF]-[BIO][0]-[probe]')
            plt.plot(sim.trange(), sim.data[probe_lif1][:, 0],
                     label='[STIM]-[LIF]-[BIO]-[LIF1], rmse=%.5f' % rmse_lif1)
            plt.plot(sim.trange(), sim.data[probe_lif2][:, 0],
                     label='[STIM]-[LIF]-[BIO]-[LIF2], rmse=%.5f' % rmse_lif2)
            plt.plot(sim.trange(), sim.data[probe_direct_out][:, 0],
                     label='[STIM]-[LIF]-[LIF_EQ]-[Direct][0]')
            plt.plot(sim.trange(), sim.data[probe_direct_out][:, 1],
                     label='[STIM]-[LIF]-[LIF_EQ]-[Direct][1]')
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

# def test_transform_out_trained_solver_and_bio_solver(Simulator, plt):
#     """
#     Simulate a network [stim]-[LIF]-[BIO]-[LIF1]
#                                          -[LIF2]
#                              -[Direct]-[Direct_out]
#     decoders_bio: decoders out of [BIO] that are trained
#                   by the oracle method (iterative)
#     w_train: soft-mux parameter that governs what fraction of
#              [BIO]-out decoders are computed
#              randomly vs from the oracle method
#     """

#     cutoff = 0.5

#     def sim(w_train=1, network=None, decoders_bio=None, plots=False):

#         if decoders_bio is None:
#             decoders_bio = np.zeros((bio_neurons, dim))

#         if network is None:
#             with nengo.Network() as network:

#                 stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

#                 lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                      seed=pre_seed, neuron_type=nengo.LIF())
#                 bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                      seed=bio_seed, neuron_type=BahlNeuron(),
#                                      max_rates=nengo.dists.Uniform(min_rate, max_rate),
#                                      intercepts=nengo.dists.Uniform(-1, 1))
#                 direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                         neuron_type=nengo.Direct())
#                 lif1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
#                                       seed=post_seed, neuron_type=nengo.LIF())
#                 lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
#                                       seed=2*post_seed, neuron_type=nengo.LIF())
#                 direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                             neuron_type=nengo.Direct())

#                 bio_solver1 = BioSolver(
#                     decoders_bio=(1. - w_train) * decoders_bio[:, 0])
#                 bio_solver2 = BioSolver(
#                     decoders_bio=(1. - w_train) * decoders_bio[:, 1])

#                 nengo.Connection(stim, lif, synapse=None)
#                 trained_solver = TrainedSolver(
#                     weights_bio = np.zeros((bio.n_neurons, lif1.n_neurons, n_syn)))
#                 nengo.Connection(lif, bio,
#                                  seed=conn_seed,
#                                  synapse=tau_neuron,
#                                  trained_weights=True,
#                                  solver = trained_solver,
#                                  n_syn=n_syn)
#                 nengo.Connection(bio[0], lif1, synapse=tau_nengo,
#                                  solver=bio_solver1)
#                 nengo.Connection(bio[1], lif2, synapse=tau_nengo,
#                                  solver=bio_solver2, transform=transform)
#                 nengo.Connection(stim, direct, synapse=tau_nengo)
#                 nengo.Connection(direct, direct_out,
#                 				 synapse=tau_nengo, transform=transform)
#                 # TODO: test output to node, direct, etc

#                 probe_bio = nengo.Probe(bio, synapse=tau_neuron,
#                                         solver=bio_solver1)
#                 # TODO: probe out of sliced bioensemble not filtering
#                 probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#                 probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
#                 probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
#                 probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

#         network = spike_match_train(network, method="1-N",
#                                     params=evo_params, plots=True)

#         with Simulator(network, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         assert np.sum(act_bio) > 0.0
#         solver = nengo.solvers.LstsqL2(reg=0.1)
#         decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])
#         # rmse_bio = rmse(sim.data[probe_bio], sim.data[probe_direct_out][:,0])
#         rmse_lif1 = rmse(sim.data[probe_lif1][:, 0],
#                          sim.data[probe_direct_out][:, 0])
#         rmse_lif2 = rmse(sim.data[probe_lif2][:, 0],
#                          sim.data[probe_direct_out][:, 1])

#         if plots:
#             plt.subplot(1, 1, 1)
#             plt.plot(sim.trange(), sim.data[probe_bio],
#                      label='[STIM]-[LIF]-[BIO][0]-[probe]')
#             plt.plot(sim.trange(), sim.data[probe_lif1][:, 0],
#                      label='[STIM]-[LIF]-[BIO]-[LIF1], rmse=%.5f' % rmse_lif1)
#             plt.plot(sim.trange(), sim.data[probe_lif2][:, 0],
#                      label='[STIM]-[LIF]-[BIO]-[LIF2], rmse=%.5f' % rmse_lif2)
#             plt.plot(sim.trange(), sim.data[probe_direct_out][:, 0],
#                      label='[STIM]-[LIF]-[LIF_EQ]-[Direct][0]')
#             plt.plot(sim.trange(), sim.data[probe_direct_out][:, 1],
#                      label='[STIM]-[LIF]-[LIF_EQ]-[Direct][1]')
#             plt.xlabel('time (s)')
#             plt.ylabel('$\hat{x}(t)$')
#             plt.title('decode')
#             plt.legend()  # prop={'size':8}

#         return decoders_bio_new, network, rmse_lif1, rmse_lif2

#     decoders_bio_new, network, _, _ = sim(w_train=1, network=None, decoders_bio=None)
#     _, _, rmse_lif1, rmse_lif2 = sim(w_train=0, network=network, decoders_bio=decoders_bio_new,
#                                   plots=True)

#     assert rmse_lif1 < cutoff
#     assert rmse_lif2 < cutoff


# def test_transform_out_trained_solver_and_bio_solver2(Simulator, plt):
#     """
#     Simulate a network [stim]-[LIF]-[BIO]-(transform)-[BIO2]
#                              -[Direct]-(transform)-[Direct_Out]
#     decoders_bio: decoders out of [BIO] that are trained
#                   by the oracle method (iterative)
#     w_train: soft-mux parameter that governs what fraction of
#              [BIO]-out decoders are computed
#              randomly vs from the oracle method
#     """

#     cutoff = 0.2
#     transform = -0.5

#     def sim(w_train=1, network=None, decoders_bio=None, plots=False):

#         if decoders_bio is None:
#             decoders_bio = np.zeros((bio_neurons, dim))

#         if network is None:
#             with nengo.Network() as network:

#                 stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
    
#                 lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                      seed=pre_seed, neuron_type=nengo.LIF())
#                 bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                      seed=bio_seed, neuron_type=BahlNeuron(),
#                                      max_rates=nengo.dists.Uniform(min_rate, max_rate),
#                                      intercepts=nengo.dists.Uniform(-1, 1))
#                 direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                         neuron_type=nengo.Direct())
#                 lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim,
#                                       seed=post_seed, neuron_type=nengo.LIF())
#                 direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                             neuron_type=nengo.Direct())
    
#                 bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
#                 trained_solver = TrainedSolver(
#                     weights_bio = np.zeros((bio.n_neurons, lif.n_neurons, n_syn)))
#                 nengo.Connection(stim, lif, synapse=None)
#                 nengo.Connection(lif, bio,
#                                  seed=conn_seed,
#                                  synapse=tau_neuron,
#                                  trained_weights=True,
#                                  solver = trained_solver,
#                                  n_syn=n_syn)
#                 nengo.Connection(lif, direct, synapse=tau_nengo)
#                 nengo.Connection(bio, lif2, synapse=tau_nengo,
#                                  transform=transform, solver=bio_solver)
#                 nengo.Connection(direct, direct_out, synapse=tau_nengo,
#                                  transform=transform)
#                 # TODO: test output to node, direct, etc

#                 network.probe_bio = nengo.Probe(bio, synapse=tau_neuron,
#                                         solver=bio_solver)
#                 # TODO: probe out of sliced bioensemble not filtering    
#                 network.probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#                 network.probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
#                 network.probe_direct = nengo.Probe(direct, synapse=tau_nengo)
#                 network.probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

#         network = spike_match_train(network, method="1-N",
#                                     params=evo_params, plots=True)

#         with Simulator(network, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[network.probe_bio_spikes], dt=dt_nengo)
#         assert np.sum(act_bio) > 0.0
#         solver = nengo.solvers.LstsqL2(reg=0.1)
#         decoders_bio_new, info = solver(act_bio, sim.data[network.probe_direct])
#     	xhat_bio = np.dot(act_bio, decoders_bio_new)
#         rmse_bio_probe = rmse(sim.data[network.probe_bio], sim.data[network.probe_direct_out])
#         rmse_bio_xhat = rmse(xhat_bio, sim.data[network.probe_direct_out])
#         rmse_lif = rmse(sim.data[network.probe_lif2], sim.data[network.probe_direct_out])

#         if plots:
#             plt.subplot(1, 1, 1)
#             plt.plot(sim.trange(), sim.data[network.probe_bio],
#                      label='BIO probe, rmse=%.5f' % rmse_bio_probe)
#             plt.plot(sim.trange(), xhat_bio,
#                      label='BIO xhat, rmse=%.5f' % rmse_bio_xhat)
#             plt.plot(sim.trange(), sim.data[network.probe_direct],
#                      label='[STIM]-[Direct]')
#             plt.plot(sim.trange(), sim.data[network.probe_lif2],
#                      label='[STIM]-[LIF]-[BIO]-[LIF2], rmse=%.5f' % rmse_lif)
#             plt.plot(sim.trange(), sim.data[network.probe_direct_out],
#                      label='[STIM]-[Direct]-[Direct_Out]')
#             plt.xlabel('time (s)')
#             plt.ylabel('$\hat{x}(t)$')
#             plt.title('decode')
#             plt.legend()  # prop={'size':8}

#         return decoders_bio_new, network, rmse_lif

#     decoders_bio_new, network, rmse_lif = sim(
#     	w_train=1, network=None, decoders_bio=None)
#     decoders_bio_new, network, rmse_lif = sim(
#     	w_train=0, network=network, decoders_bio=decoders_bio_new, plots=True)

#     assert rmse_lif < cutoff


# def test_bio_to_bio(Simulator, plt):
#     """
#     Simulate a network [stim]-[LIF]-[BIO]-[BIO2]
#                              -[Direct]-[Direct_Out]
#     decoders_bio: decoders out of [BIO] that are trained
#                   by the oracle method (iterative)
#     w_train: soft-mux parameter that governs what fraction of
#              [BIO]-out decoders are computed
#              randomly vs from the oracle method
#     jl_dims: extra dimensions for the oracle training
#              (Johnson-Lindenstrauss lemma)
#     """

#     cutoff = 0.3

#     def sim(w_train, decoders_bio=None, plots=False):

#         if decoders_bio is None:
#             decoders_bio = np.zeros((bio_neurons, dim))

#         with nengo.Network() as model:

#             stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

#             lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                  seed=pre_seed, neuron_type=nengo.LIF())
#             bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=BahlNeuron())
#             direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                     neuron_type=nengo.Direct())
#             pre = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=nengo.LIF())
#             bio2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                   seed=post_seed, neuron_type=BahlNeuron())
#             direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                         neuron_type=nengo.Direct())

#             bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

#             nengo.Connection(stim, lif, synapse=None)
#             nengo.Connection(lif, bio, synapse=tau_neuron,
#                              weights_bias_conn=True)
#             nengo.Connection(lif, direct, synapse=tau_nengo)
#             nengo.Connection(direct, pre, synapse=None)
#             nengo.Connection(pre, bio2, synapse=tau_nengo, transform=w_train)
#             nengo.Connection(bio, bio2, synapse=tau_nengo, solver=bio_solver)
#             nengo.Connection(direct, direct_out, synapse=tau_nengo)
#             # TODO: test output to node, direct, etc

#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
#             probe_direct = nengo.Probe(direct, synapse=tau_nengo)
#             probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

#         with Simulator(model, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
#         assert np.sum(act_bio) > 0.0
#         assert np.sum(act_bio2) > 0.0
#         solver = nengo.solvers.LstsqL2(reg=0.1)
#         decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])
#         decoders_bio_new2, info2 = solver(act_bio2, sim.data[probe_direct_out])
#         xhat_bio = np.dot(act_bio, decoders_bio_new)
#         xhat_bio2 = np.dot(act_bio2, decoders_bio_new2)
#         rmse_bio = rmse(xhat_bio, sim.data[probe_direct])
#         rmse_bio2 = rmse(xhat_bio2, sim.data[probe_direct_out])

#         if plots:
#             plt.subplot(1, 1, 1)
#             # plt.plot(sim.trange(), sim.data[probe_bio],
#             #          label='[STIM]-[LIF]-[BIO]-[probe]')
#             # plt.plot(sim.trange(), xhat_bio,
#             #          label='[STIM]-[LIF]-[BIO], rmse=%.5f' % rmse_bio)
#             plt.plot(sim.trange(), xhat_bio2,
#                      label='[STIM]-[LIF]-[BIO]-[BIO2], rmse=%.5f' % rmse_bio2)
#             # plt.plot(sim.trange(), sim.data[probe_direct],
#             #          label='[STIM]-[LIF]-[LIF_EQ]-[Direct]')
#             plt.plot(sim.trange(), sim.data[probe_direct_out],
#                      label='[STIM]-[Direct]-[Direct_Out]')
#             plt.xlabel('time (s)')
#             plt.ylabel('$\hat{x}(t)$')
#             plt.title('decode')
#             plt.legend()  # prop={'size':8}

#         return decoders_bio_new, rmse_bio, rmse_bio2

#     decoders_bio_new, _, _ = sim(w_train=1.0, decoders_bio=None)
#     _, rmse_bio, rmse_bio2 = sim(w_train=0.0, decoders_bio=decoders_bio_new,
#                                  plots=True)

#     assert rmse_bio < cutoff
#     assert rmse_bio2 < cutoff


# def test_bio_to_bio_transform(Simulator, plt):
#     """
#     Simulate a network [stim]-[LIF]-[BIO]-[BIO2]
#                              -[Direct]-[Direct_Out]
#     decoders_bio: decoders out of [BIO] that are trained
#                   by the oracle method (iterative)
#     w_train: soft-mux parameter that governs what fraction of
#              [BIO]-out decoders are computed
#              randomly vs from the oracle method
#     jl_dims: extra dimensions for the oracle training
#              (Johnson-Lindenstrauss lemma)
#     """

#     transform = -0.5
#     cutoff = 0.3

#     def sim(w_train, decoders_bio=None, plots=False):

#         if decoders_bio is None:
#             decoders_bio = np.zeros((bio_neurons, dim))

#         with nengo.Network() as model:

#             stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

#             lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                  seed=pre_seed, neuron_type=nengo.LIF())
#             bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=BahlNeuron())
#             direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                     neuron_type=nengo.Direct())
#             pre = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=nengo.LIF())
#             bio2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                   seed=post_seed, neuron_type=BahlNeuron())
#             direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                         neuron_type=nengo.Direct())

#             bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

#             nengo.Connection(stim, lif, synapse=None)
#             nengo.Connection(lif, bio, synapse=tau_neuron,
#                              weights_bias_conn=True)
#             nengo.Connection(lif, direct, synapse=tau_nengo)
#             nengo.Connection(direct, pre, synapse=None)
#             nengo.Connection(pre, bio2, synapse=tau_nengo, transform=w_train)
#             nengo.Connection(bio, bio2, synapse=tau_nengo,
#                              transform=transform, solver=bio_solver)
#             nengo.Connection(direct, direct_out, transform=transform,
#                              synapse=tau_nengo)
#             # TODO: test output to node, direct, etc

#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
#             probe_direct = nengo.Probe(direct, synapse=tau_nengo)
#             probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

#         with Simulator(model, dt=dt_nengo) as sim:
#             sim.run(t_final)

#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
#         assert np.sum(act_bio) > 0.0
#         assert np.sum(act_bio2) > 0.0
#         solver = nengo.solvers.LstsqL2(reg=0.1)
#         decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
#         decoders_bio_new2, info2 = solver(act_bio2, sim.data[probe_direct_out])
#         xhat_bio = np.dot(act_bio, decoders_bio_new)
#         xhat_bio2 = np.dot(act_bio2, decoders_bio_new2)
#         rmse_bio = rmse(xhat_bio, sim.data[probe_direct])
#         rmse_bio2 = rmse(xhat_bio2, sim.data[probe_direct_out])

#         if plots:
#             plt.subplot(1, 1, 1)
#             # plt.plot(sim.trange(), sim.data[probe_bio],
#             #          label='[STIM]-[LIF]-[BIO]-[probe]')
#             # plt.plot(sim.trange(), xhat_bio,
#             #          label='[STIM]-[LIF]-[BIO], rmse=%.5f' % rmse_bio)
#             plt.plot(sim.trange(), xhat_bio2,
#                      label='[STIM]-[LIF]-[BIO]-[BIO2], rmse=%.5f' % rmse_bio2)
#             # plt.plot(sim.trange(), sim.data[probe_direct],
#             #          label='[STIM]-[LIF]-[LIF_EQ]-[Direct]')
#             plt.plot(sim.trange(), sim.data[probe_direct_out],
#                      label='[STIM]-[Direct]-[Direct_Out]')
#             plt.xlabel('time (s)')
#             plt.ylabel('$\hat{x}(t)$')
#             plt.title('decode')
#             plt.legend()  # prop={'size':8}

#         return decoders_bio_new, rmse_bio, rmse_bio2

#     decoders_bio_new, _, _ = sim(w_train=1.0, decoders_bio=None)
#     _, rmse_bio, rmse_bio2 = sim(w_train=0.0, decoders_bio=decoders_bio_new,
#                                  plots=True)

#     assert rmse_bio < cutoff
#     assert rmse_bio2 < cutoff
