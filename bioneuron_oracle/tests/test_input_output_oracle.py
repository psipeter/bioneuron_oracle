import numpy as np

import nengo
from nengo.utils.numpy import rmse

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input,
                              equalpower, OracleSolver)

# Nengo Parameters
pre_neurons = 100
bio_neurons = 50
post_neurons = 50
tau_nengo = 0.05
tau_neuron = 0.05
dt_nengo = 0.001
min_rate = 300
max_rate = 400
pre_seed = 3
bio_seed = 6
conn_seed = 9
network_seed = 12
sim_seed = 15
post_seed = 18
t_final = 1.0
dim = 2
n_syn = 1
signal = 'prime_sinusoids'
max_freq = 5
signal_seed = 123


def test_transform_in(Simulator, plt):
    dim = 1
    cutoff = 0.3
    transform = -0.5

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

        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, lif, synapse=tau_nengo, transform=transform)
        nengo.Connection(pre, bio,
                         synapse=tau_neuron,
                         transform=transform,
                         weights_bias_conn=True)
        nengo.Connection(stim, direct,
                         transform=transform,
                         synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif, synapse=tau_nengo)
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

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}

def test_legendre_in(Simulator, plt):
    """
    Simulate a feedforward non transformation into a bioensemble
    use the legendre polynomials
    """

    dim = 1
    cutoff = 0.1
    plt.subplot(1, 1, 1)
    bio_neurons = 20
    signal = 'prime_sinusoids'
    orders = np.arange(1,5)
    import scipy.special as sp
    import pandas as pd
    import seaborn as sns

    columns = ('time', 'value', 'population', 'order')
    dataframe = pd.DataFrame(columns=columns)
    j=0
    for order in orders:
        with nengo.Network() as network:

            def legendre(x):
                return sp.legendre(order)(x)

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

            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, lif, synapse=tau_nengo, function=legendre)
            nengo.Connection(pre, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             function=legendre,
                             weights_bias_conn=True,
                             n_syn=n_syn)
            nengo.Connection(stim, direct, synapse=tau_nengo, function=legendre)

            probe_pre = nengo.Probe(pre, synapse=tau_nengo)
            probe_lif = nengo.Probe(lif, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

        with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
            sim.run(t_final)

        # compute oracle decoders
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
        oracle_decoders = oracle_solver(act_bio, sim.data[probe_direct])[0]

        # compute estimate on a new signal
        with network:
            stim.output = lambda t: prime_sinusoids(t, dim, t_final)
            # stim.output = lambda t: equalpower(
            #                 t, dt_nengo, t_final, max_freq, dim,
            #                 mean=0.0, std=1.0, seed=2*signal_seed)
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        xhat_bio = np.dot(act_bio, oracle_decoders)
        xhat_lif = sim.data[probe_lif]
        rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
        rmse_lif = rmse(sim.data[probe_direct], xhat_lif)

        # put into dataframe
        for t, time in enumerate(sim.trange()):
            dataframe.loc[j] = [time, xhat_bio[t][0], 'bio', order]
            dataframe.loc[j+1] = [time, xhat_lif[t][0], 'lif', order]
            dataframe.loc[j+2] = [time, sim.data[probe_direct][t][0], 'oracle', order]
            j+=3

        # plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f, order=%s' % (rmse_bio, order))
        # plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f, order=%s' % (rmse_lif, order))
        # plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
        # plt.xlabel('time (s)')
        # plt.ylabel('$\hat{x}(t)$')
        # plt.title('decode')
        # plt.legend()  # prop={'size':8}

    sns.set_style('white')
    sns.set_context('paper')
    g = sns.factorplot(x='time', y='value', hue='population', data=dataframe,
                       col='order', col_wrap=2, linestyle='-')
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

        if signal == 'prime_sinusoids':
            stim1 = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])
            stim2 = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim])
        elif signal == 'step_input':
            stim1 = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo)[0:dim/2])
            stim2 = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo)[dim/2:dim])
        elif signal == 'white_noise':
            stim1 = nengo.Node(lambda t: equalpower(
                                  t, dt_nengo, t_final, max_freq, dim/2,
                                  mean=0.0, std=1.0, seed=signal_seed))
            stim2 = nengo.Node(lambda t: equalpower(
                                  t, dt_nengo, t_final, max_freq, dim/2,
                                  mean=0.0, std=1.0, seed=3*signal_seed))

        pre1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=pre_seed, neuron_type=nengo.LIF())
        pre2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                             seed=bio.seed, neuron_type=nengo.LIF(),
                             max_rates=bio.max_rates, intercepts=bio.intercepts)
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim1, pre1, synapse=None)
        nengo.Connection(stim2, pre2, synapse=None)
        nengo.Connection(pre1, bio[0], synapse=tau_neuron,
                         n_syn=n_syn, weights_bias_conn=True)
        nengo.Connection(pre2, bio[1], synapse=tau_neuron,
                         n_syn=n_syn)
        nengo.Connection(pre1, lif[0], synapse=tau_nengo)
        nengo.Connection(pre2, lif[1], synapse=tau_nengo)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)
        nengo.Connection(stim2, direct[1], synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif, synapse=tau_nengo)
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
        stim1.output = lambda t: equalpower(
                        t, dt_nengo, t_final, max_freq, dim/2,
                        mean=0.0, std=1.0, seed=2*signal_seed)
        stim2.output = lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim]
        # TODO: bug where seed param doesn't affect shape
        # stim2.output = lambda t: equalpower(
        #                 t, dt_nengo, t_final, max_freq, dim/2,
        #                 mean=0.0, std=1.0, seed=4*signal_seed)
    with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_final)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio = np.dot(act_bio, oracle_decoders)
    xhat_lif = sim.data[probe_lif]
    rmse_bio1 = rmse(sim.data[probe_direct][:,0], xhat_bio[:,0])
    rmse_bio2 = rmse(sim.data[probe_direct][:,1], xhat_bio[:,1])
    rmse_lif1 = rmse(sim.data[probe_direct][:,0], xhat_lif[:,0])
    rmse_lif2 = rmse(sim.data[probe_direct][:,1], xhat_lif[:,1])

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio[:,0], label='bio dim 1, rmse=%.5f' % rmse_bio1)
    plt.plot(sim.trange(), xhat_bio[:,1], label='bio dim 2, rmse=%.5f' % rmse_bio2)
    plt.plot(sim.trange(), xhat_lif[:,0], label='lif dim 1, rmse=%.5f' % rmse_lif1)
    plt.plot(sim.trange(), xhat_lif[:,1], label='lif dim 2, rmse=%.5f' % rmse_lif2)
    plt.plot(sim.trange(), sim.data[probe_direct][:,0], label='oracle dim 1')
    plt.plot(sim.trange(), sim.data[probe_direct][:,1], label='oracle dim 2')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}


def test_two_inputs_one_dim(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO]
                       [stim2]-[LIF2]-[BIO]
    test passes if:
        - rmse_bio < cutoff
    """

    cutoff = 0.3
    dim = 1
    max_freq = 5
    signal_seed  = 123

    with nengo.Network() as network:

        if signal == 'prime_sinusoids':
            stim1 = nengo.Node(lambda t: 0.5*prime_sinusoids(t, dim, t_final))
            stim2 = nengo.Node(lambda t: 0.5*prime_sinusoids(t, dim, t_final))
        elif signal == 'step_input':
            stim1 = nengo.Node(lambda t: 0.5*step_input(t, dim, t_final, dt_nengo))
            stim2 = nengo.Node(lambda t: 0.5*step_input(t, dim, t_final, dt_nengo))
        elif signal == 'white_noise':
            stim1 = nengo.Node(lambda t: 0.5*equalpower(
                                  t, dt_nengo, t_final, max_freq, dim,
                                  mean=0.0, std=1.0, seed=signal_seed))
            stim2 = nengo.Node(lambda t: 0.5*equalpower(
                                  t, dt_nengo, t_final, max_freq, dim,
                                  mean=0.0, std=1.0, seed=3*signal_seed))

        pre1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                              seed=pre_seed, neuron_type=nengo.LIF())
        pre2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                              seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                             seed=bio.seed, neuron_type=nengo.LIF(),
                             max_rates=bio.max_rates, intercepts=bio.intercepts)
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim1, pre1, synapse=None)
        nengo.Connection(stim2, pre2, synapse=None)
        nengo.Connection(pre1, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         weights_bias_conn=True,
                         n_syn=n_syn)
        nengo.Connection(pre2, bio,
                         seed=conn_seed,
                         synapse=tau_neuron,
                         n_syn=n_syn)
        nengo.Connection(pre1, lif, synapse=tau_nengo)
        nengo.Connection(pre2, lif, synapse=tau_nengo)
        nengo.Connection(stim1, direct, synapse=tau_nengo)
        nengo.Connection(stim2, direct, synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif, synapse=tau_nengo)
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
        stim1.output = lambda t: 0.5*equalpower(
                        t, dt_nengo, t_final, max_freq, dim,
                        mean=0.0, std=1.0, seed=2*signal_seed)
        stim2.output = lambda t: 0.5*prime_sinusoids(t, dim, t_final)
        # TODO: bug where seed param doesn't affect shape
        # stim2.output = lambda t: 0.5*equalpower(
        #                 t, dt_nengo, t_final, max_freq, dim,
        #                 mean=0.0, std=1.0, seed=4*signal_seed)
    with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_final)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio = np.dot(act_bio, oracle_decoders)
    xhat_lif = sim.data[probe_lif]
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    rmse_lif = rmse(sim.data[probe_direct], xhat_lif)

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
    plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}


def test_slice_in(Simulator, plt):
    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][0]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """
    dim = 2
    cutoff = 0.3
    bio_neurons = 20
    signal = 'white_noise'

    with nengo.Network(seed=network_seed) as network:

        if signal == 'prime_sinusoids':
            stim = nengo.Node(lambda t: prime_sinusoids(t, dim/2, t_final))
        elif signal == 'step_input':
            stim = nengo.Node(lambda t: step_input(t, dim/2, t_final, dt_nengo))
        elif signal == 'white_noise':
            stim = nengo.Node(lambda t: equalpower(
                                  t, dt_nengo, t_final, max_freq, dim/2,
                                  mean=0.0, std=1.0, seed=signal_seed))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                              seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                             seed=bio_seed, neuron_type=BahlNeuron(),
                             max_rates=nengo.dists.Uniform(min_rate, max_rate),
                             intercepts=nengo.dists.Uniform(-1, 1))
        lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                             seed=bio.seed, neuron_type=nengo.LIF(),
                             max_rates=bio.max_rates, intercepts=bio.intercepts)     
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim/2,
                                neuron_type=nengo.Direct())

        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, lif[0], synapse=tau_nengo)
        pre_to_bio = nengo.Connection(pre, bio[0],
                         seed=conn_seed,
                         synapse=tau_neuron,
                         weights_bias_conn=True,
                         n_syn=n_syn)
        nengo.Connection(stim, direct, synapse=tau_nengo)

        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_lif = nengo.Probe(lif, synapse=tau_nengo)
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
        # stim.output = lambda t: prime_sinusoids(t, dim/2, t_final)
        stim.output = lambda t: equalpower(
                        t, dt_nengo, t_final, max_freq, dim/2,
                        mean=0.0, std=1.0, seed=2*signal_seed)
    with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_final)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio = np.dot(act_bio, oracle_decoders)[:,0]
    xhat_lif = sim.data[probe_lif][:,0]
    rmse_bio = rmse(sim.data[probe_direct][:,0], xhat_bio)
    rmse_lif = rmse(sim.data[probe_direct][:,0], xhat_lif)

    plt.subplot(1, 1, 1)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
    plt.plot(sim.trange(), sim.data[probe_direct][:,0], label='oracle')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    plt.legend()  # prop={'size':8}

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

    def sim(w_train=1, decoders_bio=None, plots=False, signal='prime_sinusoids', t_final=1.0):

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
            # lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
            #                      seed=bio.seed, neuron_type=nengo.LIF(),
            #                      max_rates=bio.max_rates, intercepts=bio.intercepts)     
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            post1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=post_seed, neuron_type=nengo.LIF())
            post2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=2*post_seed, neuron_type=nengo.LIF())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            oracle_solver1 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            oracle_solver2 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 1])

            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, bio, synapse=tau_neuron,
                             n_syn=n_syn, weights_bias_conn=True)
            nengo.Connection(bio[0], post1, synapse=tau_nengo,
                             solver=oracle_solver1)
            nengo.Connection(bio[1], post2, synapse=tau_nengo,
                             solver=oracle_solver2)
            nengo.Connection(stim, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct2, synapse=tau_nengo)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=oracle_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_post1 = nengo.Probe(post1, synapse=tau_nengo)
            probe_post2 = nengo.Probe(post2, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

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
        xhat_post1 = sim.data[probe_post1]
        xhat_post2 = sim.data[probe_post2]
        rmse_post1 = rmse(sim.data[probe_direct2][:, 0], xhat_post1[:, 0])
        rmse_post2 = rmse(sim.data[probe_direct2][:, 1], xhat_post2[:, 0])

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],label='bio probe')
            plt.plot(sim.trange(), xhat_post1, label='post1, rmse=%.5f' % rmse_post1)
            plt.plot(sim.trange(), xhat_post2, label='post2, rmse=%.5f' % rmse_post2)
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 0],label='oracle dim 1')
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 1], label='oracle dim 2')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

            assert rmse_post1 < cutoff
            assert rmse_post2 < cutoff

        return oracle_decoders

    oracle_decoders = sim(w_train=1, decoders_bio=np.zeros((bio_neurons, dim)), signal='prime_sinusoids')
    oracle_decoders = sim(w_train=0, decoders_bio=oracle_decoders, plots=True, signal='white_noise')


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
    dim = 1

    def sim(w_train=1, decoders_bio=None, plots=False, signal='prime_sinusoids', t_final=1.0):

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
            # lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
            #                      seed=bio.seed, neuron_type=nengo.LIF(),
            #                      max_rates=bio.max_rates, intercepts=bio.intercepts)     
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            post = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim,
                                  seed=post_seed, neuron_type=nengo.LIF())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            oracle_solver = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio)

            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, bio, synapse=tau_neuron,
                             n_syn=n_syn, weights_bias_conn=True)
            nengo.Connection(bio, post, synapse=tau_nengo,
                             solver=oracle_solver, transform=transform)
            nengo.Connection(stim, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct2, synapse=tau_nengo, transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio, synapse=tau_neuron,
                                    solver=oracle_solver)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_post = nengo.Probe(post, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

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
        xhat_post = sim.data[probe_post]
        rmse_post = rmse(sim.data[probe_direct2], xhat_post)

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],label='bio probe')
            plt.plot(sim.trange(), xhat_post, label='post, rmse=%.5f' % rmse_post)
            plt.plot(sim.trange(), sim.data[probe_direct2],label='oracle')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

            assert rmse_post < cutoff

        return oracle_decoders

    oracle_decoders = sim(w_train=1, decoders_bio=np.zeros((bio_neurons, dim)), signal='prime_sinusoids')
    oracle_decoders = sim(w_train=0, decoders_bio=oracle_decoders, plots=True, signal='white_noise')


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

    def sim(w_train=1, decoders_bio=None, plots=False, signal='prime_sinusoids', t_final=1.0):

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
            # lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
            #                      seed=bio.seed, neuron_type=nengo.LIF(),
            #                      max_rates=bio.max_rates, intercepts=bio.intercepts)     
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            post1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=post_seed, neuron_type=nengo.LIF())
            post2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                  seed=2*post_seed, neuron_type=nengo.LIF())
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            oracle_solver1 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 0])
            oracle_solver2 = OracleSolver(
                decoders_bio=(1. - w_train) * decoders_bio[:, 1])

            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, bio, synapse=tau_neuron,
                             n_syn=n_syn, weights_bias_conn=True)
            nengo.Connection(bio[0], post1, synapse=tau_nengo,
                             solver=oracle_solver1, transform=transform)
            nengo.Connection(bio[1], post2, synapse=tau_nengo,
                             solver=oracle_solver2, transform=transform)
            nengo.Connection(stim, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct2, synapse=tau_nengo, transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio = nengo.Probe(bio[0], synapse=tau_neuron,
                                    solver=oracle_solver1)
            # TODO: probe out of sliced bioensemble not filtering
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_post1 = nengo.Probe(post1, synapse=tau_nengo)
            probe_post2 = nengo.Probe(post2, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

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
        xhat_post1 = sim.data[probe_post1]
        xhat_post2 = sim.data[probe_post2]
        rmse_post1 = rmse(sim.data[probe_direct2][:, 0], xhat_post1[:, 0])
        rmse_post2 = rmse(sim.data[probe_direct2][:, 1], xhat_post2[:, 0])

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio],label='bio probe')
            plt.plot(sim.trange(), xhat_post1, label='post1, rmse=%.5f' % rmse_post1)
            plt.plot(sim.trange(), xhat_post2, label='post2, rmse=%.5f' % rmse_post2)
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 0],label='oracle dim 1')
            plt.plot(sim.trange(), sim.data[probe_direct2][:, 1], label='oracle dim 2')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}

            assert rmse_post1 < cutoff
            assert rmse_post2 < cutoff

        return oracle_decoders

    oracle_decoders = sim(w_train=1, decoders_bio=np.zeros((bio_neurons, dim)), signal='prime_sinusoids')
    oracle_decoders = sim(w_train=0, decoders_bio=oracle_decoders, plots=True, signal='white_noise')


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
    dim = 2
    bio_neurons = 20
    post_neurons = 20

    def sim(w_train, decoders_bio=None, plots=False, t_final=1.0,
            signal_train='prime_sinusoids', signal_test='white_noise', signal_seed=123):

        with nengo.Network(seed=network_seed) as network:

            if signal_train == 'prime_sinusoids':
                stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
            elif signal_train == 'step_input':
                stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
            elif signal_train == 'white_noise':
                stim = nengo.Node(lambda t: equalpower(
                                      t, dt_nengo, t_final, max_freq, dim,
                                      mean=0.0, std=1.0, seed=signal_seed))

            pre1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            lif1 = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)     
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            # pre2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
            #                      seed=bio_seed, neuron_type=nengo.LIF())
            bio2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim,
                                  seed=post_seed, neuron_type=BahlNeuron())
            lif2 = nengo.Ensemble(n_neurons=bio2.n_neurons, dimensions=bio2.dimensions,
                                 seed=bio2.seed, neuron_type=nengo.LIF(),
                                 max_rates=bio2.max_rates, intercepts=bio2.intercepts)     
            direct2 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                        neuron_type=nengo.Direct())

            oracle_solver = OracleSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

            nengo.Connection(stim, pre1, synapse=None)
            nengo.Connection(pre1, bio, synapse=tau_neuron,
                             n_syn=n_syn, weights_bias_conn=True)
            nengo.Connection(pre1, lif1, synapse=tau_nengo)
            nengo.Connection(stim, direct, synapse=tau_nengo)
            # nengo.Connection(direct, pre2, synapse=None)
            # nengo.Connection(pre2, bio2, synapse=tau_nengo, transform=w_train)
            nengo.Connection(lif1, lif2, synapse=tau_nengo, transform=transform)
            nengo.Connection(bio, bio2, synapse=tau_nengo,
                             n_syn=n_syn, solver=oracle_solver,
                             weights_bias_conn=True, transform=transform)
            nengo.Connection(direct, direct2,
                             synapse=tau_nengo, transform=transform)
            # TODO: test output to node, direct, etc

            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif1.neurons, 'spikes')
            probe_lif2_spikes = nengo.Probe(lif2.neurons, 'spikes')
            probe_lif = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct2 = nengo.Probe(direct2, synapse=tau_nengo)

        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)

        # compute oracle decoders
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
        oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
        oracle_solver2 = nengo.solvers.LstsqL2(reg=0.01)
        oracle_decoders = oracle_solver(act_bio, sim.data[probe_direct])[0]
        oracle_decoders2 = oracle_solver2(act_bio2, sim.data[probe_direct2])[0]

        # compute estimate on a new signal
        with network:
            if signal_test == 'prime_sinusoids':
                stim.output = lambda t: prime_sinusoids(t, dim, t_final)
            elif signal_test == 'step_input':
                stim.output = lambda t: step_input(t, dim, t_final, dt_nengo)
            elif signal_test == 'white_noise':
                stim.output = lambda t: equalpower(
                                      t, dt_nengo, t_final, max_freq, dim,
                                      mean=0.0, std=1.0, seed=2*signal_seed)
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
        xhat_bio = np.dot(act_bio, oracle_decoders)
        xhat_lif = sim.data[probe_lif]
        xhat_bio2 = np.dot(act_bio2, oracle_decoders2)
        xhat_lif2 = sim.data[probe_lif2]
        rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
        rmse_bio2 = rmse(sim.data[probe_direct2], xhat_bio2)
        rmse_lif = rmse(sim.data[probe_direct], xhat_lif)
        rmse_lif2 = rmse(sim.data[probe_direct2], xhat_lif2)

        if plots:
            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio], label='bio probe')
            plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
            plt.plot(sim.trange(), sim.data[probe_direct], label='oracle')
            plt.plot(sim.trange(), xhat_bio2, label='bio 2, rmse=%.5f' % rmse_bio2)
            plt.plot(sim.trange(), xhat_lif2, label='lif 2, rmse=%.5f' % rmse_lif2)
            plt.plot(sim.trange(), sim.data[probe_direct2], label='oracle 2')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}
            assert rmse_bio < cutoff
            assert rmse_bio2 < cutoff

        return oracle_decoders

    oracle_decoders = sim(w_train=1.0, decoders_bio=np.zeros((bio_neurons, dim)),
        signal_train='prime_sinusoids', signal_test='white_noise', plots=False,
        signal_seed=123, t_final=1.0)
    oracle_decoders = sim(w_train=0.0, decoders_bio=oracle_decoders,
        signal_train='prime_sinusoids', signal_test='white_noise', plots=True,
        signal_seed=2*123, t_final=1.0)
