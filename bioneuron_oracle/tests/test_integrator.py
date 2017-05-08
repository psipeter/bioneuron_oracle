import numpy as np

import nengo
from nengo.utils.numpy import rmse

from nengolib.signal import s

from bioneuron_oracle import BahlNeuron, prime_sinusoids, equalpower, BioSolver


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

    pre_neurons = 100
    bio_neurons = 50
    tau_nengo = 0.1
    tau_neuron = 0.1
    dt_nengo = 0.001
    lif_seed = 3
    bio_seed = 6
    pre_seed = 9
    t_final = 1.0
    dim = 1
    jl_dims = 0
    signal = 'white_noise'
    if signal == 'white_noise':
        max_freq = 5.0
        signal_seed = 123
    cutoff = 0.1

    def sim(w_train, decoders_bio=None, plots=False):

        if decoders_bio is None:
            decoders_bio = np.zeros((bio_neurons, dim))

        with nengo.Network() as model:

            if signal == 'prime_sinusoids':
                stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
            elif signal == 'white_noise':
                stim = nengo.Node(lambda t: equalpower(
                                      t, dt_nengo, t_final, max_freq, dim,
                                      mean=0.0, std=1.0, seed=signal_seed))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=lif_seed, neuron_type=nengo.LIF())
            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 max_rates=nengo.dists.Uniform(50, 100),
                                 intercepts=nengo.dists.Uniform(-1.0, 1.0))
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            compare = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                     seed=bio_seed, neuron_type=nengo.LIF())
            integral = nengo.Node(size_in=dim)

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
            if jl_dims > 0:
                # TODO: magnitude should scale with n_neurons (maybe 1./n^2)?
                jl_decoders = np.random.RandomState(seed=333).randn(
                    bio_neurons, jl_dims) * 1e-4
                bio_solver.decoders_bio = np.hstack(
                    (bio_solver.decoders_bio, jl_decoders))

            nengo.Connection(stim, lif, synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(lif, bio[:dim], weights_bias_conn=True,
                             synapse=tau_neuron, transform=tau_neuron)
            nengo.Connection(lif, compare, weights_bias_conn=True,
                             synapse=tau_nengo, transform=tau_nengo)
            nengo.Connection(bio, bio, synapse=tau_neuron, solver=bio_solver)
            nengo.Connection(compare, compare, synapse=tau_nengo)

            nengo.Connection(stim, integral, synapse=1/s)  # integrator
            nengo.Connection(integral, pre, synapse=None)
            nengo.Connection(pre, bio[:dim],  # oracle connection
                             synapse=tau_neuron, transform=w_train)

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_compare_spikes = nengo.Probe(compare.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=None)

        with Simulator(model, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_compare = lpf.filt(sim.data[probe_compare_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        assert np.sum(act_compare) > 0.0
        target = sim.data[probe_integral]
        solver_bio = nengo.solvers.LstsqL2(reg=0.1)
        solver_compare = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver_bio(act_bio, target)
        decoders_compare, info = solver_compare(act_compare, target)
        xhat_bio = np.dot(act_bio, decoders_bio_new)
        xhat_compare = np.dot(act_compare, decoders_compare)
        rmse_bio = rmse(xhat_bio, target)
        rmse_compare = rmse(xhat_compare, target)

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_stim], label='stim')
            plt.plot(sim.trange(), xhat_bio,
                     label='bio, rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), xhat_compare,
                     label='lif, rmse=%.5f' % rmse_compare)
            plt.plot(sim.trange(), target, label='target')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode, n_neurons=%s' % bio_neurons)
            plt.legend()

        return decoders_bio_new, rmse_bio

    decoders_bio_new, _ = sim(w_train=1.0, decoders_bio=None, plots=False)
    _, rmse_bio = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_bio < cutoff


def test_pre_switch(Simulator, plt):
    """
    simulate the same network as above, but change the LIF ensembles
    (LIF and pre) to have different seeds / rates between training and testing.
    """

    pre_neurons = 100
    bio_neurons = 50
    tau_nengo = 0.1
    tau_neuron = 0.1
    dt_nengo = 0.001
    lif_seed = 3
    bio_seed = 6
    pre_seed = 9
    t_final = 1.0
    dim = 2
    jl_dims = 0
    signal = 'prime_sinusoids'
    if signal == 'white_noise':
        max_freq = 5.0
        signal_seed = 123
    cutoff = 0.1

    def sim(w_train, decoders_bio=None, plots=False):

        if decoders_bio is None:
            decoders_bio = np.zeros((bio_neurons, dim))

        switch = int(1 + w_train)

        with nengo.Network() as model:

            if signal == 'prime_sinusoids':
                stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
            elif signal == 'white_noise':
                stim = nengo.Node(lambda t: equalpower(
                                      t, dt_nengo, t_final, max_freq, dim,
                                      mean=0.0, std=1.0, seed=signal_seed))

            max_rates = nengo.dists.Uniform(switch*50, switch*100)
            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=switch*lif_seed, neuron_type=nengo.LIF(),
                                 max_rates=max_rates,
                                 intercepts=nengo.dists.Uniform(-1.0, 1.0))
            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=switch*pre_seed, neuron_type=nengo.LIF(),
                                 max_rates=max_rates,
                                 intercepts=nengo.dists.Uniform(-1.0, 1.0))
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron())
            compare = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                     seed=bio_seed, neuron_type=nengo.LIF())
            integral = nengo.Node(size_in=dim)

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
            if jl_dims > 0:
                # TODO: magnitude should scale with n_neurons (maybe 1./n^2)?
                jl_decoders = np.random.RandomState(seed=333).randn(
                    bio_neurons, jl_dims) * 1e-4
                bio_solver.decoders_bio = np.hstack(
                    (bio_solver.decoders_bio, jl_decoders))

            nengo.Connection(stim, lif, synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(lif, bio[:dim], weights_bias_conn=True,
                             synapse=tau_neuron, transform=tau_neuron)
            nengo.Connection(lif, compare, weights_bias_conn=True,
                             synapse=tau_nengo, transform=tau_nengo)
            nengo.Connection(bio, bio, synapse=tau_neuron, solver=bio_solver)
            nengo.Connection(compare, compare, synapse=tau_nengo)

            nengo.Connection(stim, integral, synapse=1/s)  # integrator
            nengo.Connection(integral, pre, synapse=None)
            nengo.Connection(pre, bio[:dim],  # oracle connection
                             synapse=tau_neuron, transform=w_train)

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_compare_spikes = nengo.Probe(compare.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=None)

        with Simulator(model, dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_compare = lpf.filt(sim.data[probe_compare_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        assert np.sum(act_compare) > 0.0
        target = sim.data[probe_integral]
        solver_bio = nengo.solvers.LstsqL2(reg=0.1)
        solver_compare = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver_bio(act_bio, target)
        decoders_compare, info = solver_compare(act_compare, target)
        xhat_bio = np.dot(act_bio, decoders_bio_new)
        xhat_compare = np.dot(act_compare, decoders_compare)
        rmse_bio = rmse(xhat_bio, target)
        rmse_compare = rmse(xhat_compare, target)

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_stim], label='stim')
            plt.plot(sim.trange(), xhat_bio,
                     label='bio, rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), xhat_compare,
                     label='lif, rmse=%.5f' % rmse_compare)
            plt.plot(sim.trange(), target, label='target')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode, n_neurons=%s' % bio_neurons)
            plt.legend()

        return decoders_bio_new, rmse_bio

    decoders_bio_new, _ = sim(w_train=1.0, decoders_bio=None, plots=False)
    _, rmse_bio = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_bio < cutoff
