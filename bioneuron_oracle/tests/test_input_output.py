import numpy as np

import nengo
from nengo.utils.numpy import rmse

from bioneuron_oracle import BahlNeuron, prime_sinusoids, step_input, BioSolver

pre_neurons = 100
bio_neurons = 20
post_neurons = 100
tau_nengo = 0.01
tau_neuron = 0.01
dt_nengo = 0.001
dt_neuron = 0.0001
pre_seed = 3
bio_seed = 6
post_seed = 9
t_final = 1.0
dim = 2
assert dim % 2 == 0
n_syn = 10
signal = 'prime_sinusoids'
decoders_bio = None


def test_transform_in(Simulator, plt):
    cutoff = 0.3
    transform = -0.5
    with nengo.Network() as model:
        """
        Simulate a feedforward network [stim]-[LIF]-(transform)-[BIO]
        test passes if rmse_bio < cutoff
        """

        if signal == 'prime_sinusoids':
            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
        elif signal == 'step_input':
            stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                             seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, bio,
                         synapse=tau_neuron,
                         transform=transform,
                         weights_bias_conn=True)
        nengo.Connection(stim, direct,
                         transform=transform,
                         synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    with Simulator(model, dt=dt_nengo) as sim:
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


def test_two_inputs_two_dims(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][1]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4

    with nengo.Network() as model:

        stim1 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])
        stim2 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(stim2, lif2, synapse=None)
        nengo.Connection(lif1, bio[0], synapse=tau_neuron,
                         weights_bias_conn=True)
        nengo.Connection(lif2, bio[1], synapse=tau_neuron)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)
        nengo.Connection(stim2, direct[1], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    with Simulator(model, dt=dt_nengo) as sim:
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


def test_two_inputs_one_dim(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4

    with nengo.Network() as model:

        stim1 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])
        stim2 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim/2,
                             seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim/2,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(stim2, lif2, synapse=None)
        nengo.Connection(lif1, bio[0], synapse=tau_neuron,
                         weights_bias_conn=True)
        nengo.Connection(lif2, bio[0], synapse=tau_neuron)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)
        nengo.Connection(stim2, direct[0], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    with Simulator(model, dt=dt_nengo) as sim:
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


def test_slice_in(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][1]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4

    with nengo.Network() as model:

        stim1 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])
        stim2 = nengo.Node(
            lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(stim2, lif2, synapse=None)
        nengo.Connection(lif1, bio[0], synapse=tau_neuron,
                         weights_bias_conn=True)
        nengo.Connection(lif2, bio[1], synapse=tau_neuron)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)
        nengo.Connection(stim2, direct[1], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    with Simulator(model, dt=dt_nengo) as sim:
        sim.run(t_final)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)

    plt.subplot(1, 1, 1)
    rmse_bio_1 = rmse(sim.data[probe_direct][:, 0:dim/2],
                      xhat_bio[:, 0:dim/2])
    rmse_bio_2 = rmse(sim.data[probe_direct][:, dim/2:dim],
                      xhat_bio[:, dim/2:dim])
    plt.plot(sim.trange(), xhat_bio[:, 0:dim/2],
             label='bio dim 1, rmse=%.5f' % rmse_bio_1)
    plt.plot(sim.trange(), xhat_bio[:, dim/2:dim],
             label='bio dim 2, rmse=%.5f' % rmse_bio_2)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}
    assert rmse_bio_1 < cutoff
    assert rmse_bio_2 < cutoff


def test_slice_in_2(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1][0]-[BIO][0]

    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff = 0.4
    dim = 2

    with nengo.Network() as model:

        stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final)[0])

        lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                             seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim, lif[0], synapse=None)
        nengo.Connection(lif[0], bio[0], synapse=tau_neuron,
                         weights_bias_conn=True)
        nengo.Connection(stim, direct[0], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    with Simulator(model, dt=dt_nengo) as sim:
        sim.run(t_final)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)

    plt.subplot(1, 1, 1)
    rmse_bio = rmse(sim.data[probe_direct][:, 0], xhat_bio[:, 0])
    plt.plot(sim.trange(), xhat_bio[:, 0],
             label='bio dim 1, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct][:, 0], label='direct')
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

    cutoff = 0.5

    def sim(w_train=1, decoders_bio=None, plots=False):

        if decoders_bio is None:
            decoders_bio = np.zeros((bio_neurons, dim))

        with nengo.Network() as model:

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
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            bio_solver1 = BioSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            bio_solver2 = BioSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 1])

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron,
                             weights_bias_conn=True)
            nengo.Connection(bio[0], lif1, synapse=tau_nengo,
                             solver=bio_solver1)
            nengo.Connection(bio[1], lif2, synapse=tau_nengo,
                             solver=bio_solver2)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct2, synapse=tau_nengo)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=bio_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

        with Simulator(model, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct2])
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
                     label='Direct2[0]')
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 1],
                     label='Direct2[1]')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

        return decoders_bio_new, rmse_lif1, rmse_lif2

    decoders_bio_new, _, _ = sim(w_train=1, decoders_bio=None)
    _, rmse_lif1, rmse_lif2 = sim(w_train=0, decoders_bio=decoders_bio_new,
                                  plots=True)

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

    cutoff = 0.2
    transform = -0.5
    bio_neurons = 20

    def sim(w_train, decoders_bio=None, plots=False):

        if decoders_bio is None:
            decoders_bio = np.zeros((bio_neurons, dim))

        with nengo.Network(seed=1) as model:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim,
                                  seed=post_seed, neuron_type=nengo.LIF())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron,
                             weights_bias_conn=True)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(bio, lif2, synapse=tau_nengo,
                             transform=transform, solver=bio_solver)
            nengo.Connection(direct, direct2, synapse=tau_nengo,
                             transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

        with Simulator(model, dt=dt_nengo, seed=1) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
        rmse_lif = rmse(sim.data[probe_lif2], sim.data[probe_direct2])

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],
            #          label='BIO (probe), rmse=%.5f' % rmse_bio)
            # plt.plot(sim.trange(), xhat_bio,
            #          label='BIO (xhat), rmse=%.5f' % rmse_bio_xhat)
            # plt.plot(sim.trange(), sim.data[probe_direct],
            #          label='Direct')
            plt.plot(sim.trange(), sim.data[probe_lif2],
                     label='LIF2, rmse=%.5f' % rmse_lif)
            plt.plot(sim.trange(), sim.data[probe_direct2],
                     label='Direct2')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

        return decoders_bio_new, rmse_lif

    decoders_bio_new, _ = sim(w_train=1.0, decoders_bio=None)
    _, rmse_lif = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

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

    def sim(w_train=1, decoders_bio=None, plots=False):

        if decoders_bio is None:
            decoders_bio = np.zeros((bio_neurons, dim))

        with nengo.Network() as model:

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
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            bio_solver1 = BioSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            bio_solver2 = BioSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 1])

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron,
                             weights_bias_conn=True)
            nengo.Connection(bio[0], lif1, synapse=tau_nengo,
                             solver=bio_solver1, transform=transform)
            nengo.Connection(bio[1], lif2, synapse=tau_nengo,
                             solver=bio_solver2, transform=transform)
            nengo.Connection(lif, direct,
                             synapse=tau_nengo, transform=transform)
            nengo.Connection(direct, direct2, synapse=tau_nengo)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=bio_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

        with Simulator(model, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct2])
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
                     label='Direct2[0]')
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 1],
                     label='Direct2[1]')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}


        return decoders_bio_new, rmse_lif1, rmse_lif2

    decoders_bio_new, _, _ = sim(w_train=1, decoders_bio=None)
    _, rmse_lif1, rmse_lif2 = sim(w_train=0, decoders_bio=decoders_bio_new,
                                  plots=True)

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

    cutoff = 0.3
    transform = -0.5

    def sim(w_train, decoders_bio=None, plots=False):

        if decoders_bio is None:
            decoders_bio = np.zeros((bio_neurons, dim))

        with nengo.Network() as model:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            pre1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            pre2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF())
            bio2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                  seed=post_seed, neuron_type=BahlNeuron())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

            nengo.Connection(stim, pre1, synapse=None)
            nengo.Connection(pre1, bio, synapse=tau_neuron,
                             weights_bias_conn=True)
            nengo.Connection(pre1, direct, synapse=tau_nengo)
            nengo.Connection(direct, pre2, synapse=None)
            nengo.Connection(pre2, bio2, synapse=tau_nengo, transform=w_train)
            nengo.Connection(bio, bio2, synapse=tau_nengo,
                             solver=bio_solver, transform=transform)
            nengo.Connection(direct, direct2,
                             synapse=tau_nengo, transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

        with Simulator(model, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        assert np.sum(act_bio2) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct2])
        decoders_bio_new2, info2 = solver(act_bio2, sim.data[probe_direct2])
        xhat_bio = np.dot(act_bio, decoders_bio_new)
        xhat_bio2 = np.dot(act_bio2, decoders_bio_new2)
        rmse_bio = rmse(xhat_bio, sim.data[probe_direct])
        rmse_bio2 = rmse(xhat_bio2, sim.data[probe_direct2])

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],
            #          label='BIO (probe)')
            # plt.plot(sim.trange(), xhat_bio,
            #          label='BIO, rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), xhat_bio2,
                     label='BIO2, rmse=%.5f' % rmse_bio2)
            # plt.plot(sim.trange(), sim.data[probe_direct],
            #          label='Direct')
            plt.plot(sim.trange(), sim.data[probe_direct2],
                     label='Direct2')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

        return decoders_bio_new, rmse_bio, rmse_bio2

    decoders_bio_new, _, _ = sim(w_train=1.0, decoders_bio=None)
    _, rmse_bio, rmse_bio2 = sim(w_train=0.0, decoders_bio=decoders_bio_new,
                                 plots=True)

    assert rmse_bio < cutoff
    assert rmse_bio2 < cutoff
