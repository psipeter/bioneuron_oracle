import numpy as np

import nengo
from nengo.utils.numpy import rmse

from nengolib.signal import s

from bioneuron_oracle import BahlNeuron, prime_sinusoids, equalpower, OracleSolver


def test_integrator_1d(Simulator, plt):
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
    inter_seed = 21
    t_train = 1.0
    t_test = 1.0
    dim = 1
    n_syn = 1
    max_freq = 5

    jl_dims = 3
    cutoff = 0.1

    def sim(w_train, decoders_bio=None, plots=False, t_train=1.0, t_test=1.0,
            signal_train='prime_sinusoids', signal_test='white_noise', signal_seed=123):

        """
        Define the network, with extra dimensions in bio for JL-feedback
        Set the recurrent decoders set to decoders_bio,
        which are initially set to zero or the result of a previous sim() call
        """
        with nengo.Network(seed=network_seed) as network:

            amp = 4*np.pi
            if signal_train == 'prime_sinusoids':
                stim = nengo.Node(lambda t: amp * prime_sinusoids(t, dim, t_train, f_0=2.0))
            elif signal_train == 'step_input':
                stim = nengo.Node(lambda t: step_input(t, dim, t_train, dt_nengo))
            elif signal_train == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_train, high=max_freq, rms=1.0, seed=signal_seed))
                # stim = nengo.Node(lambda t: equalpower(
                #                       t, dt_nengo, t_train, max_freq, dim,
                #                       mean=0.0, std=1.0, seed=signal_seed))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 radius=amp)
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            inter = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=inter_seed, neuron_type=nengo.LIF(),
                                 max_rates=pre.max_rates, intercepts=pre.intercepts)
            lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            integral = nengo.Node(size_in=dim)

            oracle_solver = OracleSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
            if jl_dims > 0:
                # TODO: magnitude should scale with n_neurons (maybe 1./n^2)?
                jl_decoders = np.random.RandomState(seed=conn_seed).randn(
                    bio_neurons, jl_dims) * 5e-4
                oracle_solver.decoders_bio = np.hstack(
                    (oracle_solver.decoders_bio, jl_decoders))

            nengo.Connection(stim, pre, synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(pre, bio[:dim], weights_bias_conn=True,
                             seed=conn_seed, synapse=tau_neuron,
                             transform=tau_neuron)
            nengo.Connection(pre, lif, weights_bias_conn=True,
                             synapse=tau_nengo, transform=tau_nengo)
            nengo.Connection(bio, bio, synapse=tau_neuron, seed=conn_seed,
                             n_syn=n_syn, solver=oracle_solver)
            nengo.Connection(lif, lif, synapse=tau_nengo)

            nengo.Connection(stim, integral, synapse=1/s)  # integrator
            nengo.Connection(integral, inter, synapse=None)
            nengo.Connection(inter, bio[:dim],  # oracle connection
                             synapse=tau_neuron, transform=w_train)

            probe_lif = nengo.Probe(lif, synapse=tau_nengo)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=tau_nengo)


        """
        Simulate the network, collect bioneuron activities and target values,
        and apply the oracle method to calculate recurrent decoders
        """
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_train)
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
        oracle_decoders = oracle_solver(act_bio, sim.data[probe_integral])[0]
        if jl_dims > 0:
            full_decoders = np.hstack((oracle_decoders, jl_decoders))


        """
        Run the simulation on a new signal, and use the full_decoders
        calculated above to estimate the bioneurons' outputs for plotting
        """
        if plots:
            with network:
                amp2 = 2*np.pi
                if signal_test == 'prime_sinusoids':
                    stim.output = lambda t: amp2*prime_sinusoids(t, dim, t_test, f_0=1.0)
                elif signal_test == 'step_input':
                    stim.output = lambda t: step_input(t, dim, t_test, dt_nengo)
                elif signal_test == 'white_noise':
                    stim = nengo.Node(nengo.processes.WhiteSignal(
                        period=t_train, high=max_freq, rms=1.0, seed=signal_seed))
            #         stim.output = lambda t: equalpower(
            #                         t, dt_nengo, t_test, max_freq, dim,
            #                         mean=0.0, std=1.0, seed=2*signal_seed)
            with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
                sim.run(t_test)
            act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
            if jl_dims > 0:
                xhat_bio = np.dot(act_bio, full_decoders)
            else:
                xhat_bio = np.dot(act_bio, oracle_decoders)                
            xhat_lif = sim.data[probe_lif]
            rmse_bio = rmse(sim.data[probe_integral][:,0], xhat_bio[:,0])
            rmse_lif = rmse(sim.data[probe_integral], xhat_lif)

            plt.subplot(1, 1, 1)
            # plt.plot(sim.trange(), sim.data[probe_bio], label='bio probe')
            plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
            if jl_dims > 0:
                plt.plot(sim.trange(), xhat_bio[:,1:], label='jm_dims')
            # plt.plot(sim.trange(), xhat_bio, label='jm_dims')
            plt.plot(sim.trange(), sim.data[probe_integral], label='oracle')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}
            assert rmse_bio < cutoff

        return oracle_decoders


    oracle_decoders = sim(w_train=1.0, decoders_bio=np.zeros((bio_neurons, dim)),
        signal_train='prime_sinusoids', signal_test='prime_sinusoids', plots=False,
        signal_seed=3, t_train=t_train, t_test=t_test)
    oracle_decoders = sim(w_train=0.0, decoders_bio=oracle_decoders,
        signal_train='white_noise', signal_test='white_noise', plots=True,
        signal_seed=123, t_train=t_train, t_test=t_test)



def test_integrator_2d(Simulator, plt):

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
    inter_seed = 21
    t_train = 10.0
    t_test = 1.0
    dim = 2
    n_syn = 1
    max_freq = 5

    jl_dims = 0
    jl_dim_mag = 2e-4
    cutoff = 0.1


    def sim(w_train, decoders_bio=None, plots=False, t_train=1.0, t_test=1.0,
            signal_train='prime_sinusoids', signal_test='white_noise', signal_seed=123):

        """
        Define the network, with extra dimensions in bio for JL-feedback
        Set the recurrent decoders set to decoders_bio,
        which are initially set to zero or the result of a previous sim() call
        """
        with nengo.Network(seed=network_seed) as network:

            amp = 2*np.pi
            amp2 = 4*np.pi
            rms = 5.0
            if signal_train == 'prime_sinusoids':
                stim = nengo.Node(lambda t: amp * prime_sinusoids(t, dim/2, t_train, f_0=1))
                stim2 = nengo.Node(lambda t: amp2 * prime_sinusoids(t, dim/2, t_train, f_0=2))
            elif signal_train == 'step_input':
                stim = nengo.Node(lambda t: step_input(t, dim/2, t_train, dt_nengo))
                stim2 = nengo.Node(lambda t: step_input(t, dim/2, t_train, dt_nengo))
            elif signal_train == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_train, high=max_freq, rms=rms, seed=signal_seed))
                # stim = nengo.Node(lambda t: equalpower(
                #                       t, dt_nengo, t_train, max_freq, dim/2,
                #                       mean=0.0, std=3.0, seed=signal_seed))
                stim2 = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_train, high=max_freq, rms=rms, seed=2*signal_seed))
                # stim2 = nengo.Node(lambda t: equalpower(
                #                       t, dt_nengo, t_train, max_freq, dim/2,
                #                       mean=0.0, std=3.0, seed=2*signal_seed))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 radius=amp+amp2)
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            inter = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            integral = nengo.Node(size_in=dim)

            oracle_solver = OracleSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
            if jl_dims > 0:
                # TODO: magnitude should scale with n_neurons (maybe 1./n^2)?
                jl_decoders = np.random.RandomState(seed=conn_seed).randn(
                    bio_neurons, jl_dims) * jl_dim_mag
                oracle_solver.decoders_bio = np.hstack(
                    (oracle_solver.decoders_bio, jl_decoders))

            nengo.Connection(stim, pre[0], synapse=None)
            nengo.Connection(stim2, pre[1], synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(pre, bio[:dim], weights_bias_conn=True,
                             seed=conn_seed, synapse=tau_neuron,
                             transform=tau_neuron)
            nengo.Connection(pre, lif, weights_bias_conn=True,
                             synapse=tau_nengo, transform=tau_nengo)
            nengo.Connection(bio, bio, synapse=tau_neuron, seed=conn_seed,
                             n_syn=n_syn, solver=oracle_solver)
            nengo.Connection(lif, lif, synapse=tau_nengo)

            nengo.Connection(stim, integral[0], synapse=1/s)  # integrator
            nengo.Connection(stim2, integral[1], synapse=1/s)  # integrator
            nengo.Connection(integral, inter, synapse=None)
            nengo.Connection(inter, bio[:dim],  # oracle connection
                             synapse=tau_neuron, transform=w_train, seed=conn_seed)

            probe_lif = nengo.Probe(lif, synapse=tau_nengo)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=tau_nengo)

        """
        Simulate the network, collect bioneuron activities and target values,
        and apply the oracle method to calculate recurrent decoders
        """
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_train)
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
        oracle_decoders = oracle_solver(act_bio, sim.data[probe_integral])[0]
        if jl_dims > 0:
            full_decoders = np.hstack((oracle_decoders, jl_decoders))

        """
        Run the simulation on a new signal, and use the full_decoders
        calculated above to estimate the bioneurons' outputs for plotting
        """
        if plots:
            with network:
                if signal_test == 'prime_sinusoids':
                    stim.output = lambda t: amp2 * prime_sinusoids(t, dim/2, t_test, f_0=2)
                    stim2.output = lambda t: amp * prime_sinusoids(t, dim/2, t_test, f_0=1)
                elif signal_test == 'step_input':
                    stim.output = lambda t: step_input(t, dim/2, t_test, dt_nengo)
                    stim2.output = lambda t: step_input(t, dim/2, t_test, dt_nengo)
                elif signal_test == 'white_noise':
                    stim.output = nengo.processes.WhiteSignal(
                        period=t_test, high=max_freq, rms=rms, seed=3*signal_seed)
                    # stim.output = lambda t: equalpower(
                    #                       t, dt_nengo, t_test, max_freq, dim/2,
                    #                       mean=0.0, std=3.0, seed=3*signal_seed)
                    stim2.output = nengo.processes.WhiteSignal(
                        period=t_test, high=max_freq, rms=rms, seed=4*signal_seed)
                    # stim2.output = lambda t: equalpower(
                    #                       t, dt_nengo, t_test, max_freq, dim/2,
                    #                       mean=0.0, std=3.0, seed=4*signal_seed)

            with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
                sim.run(t_test)
            act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
            if jl_dims > 0:
                xhat_bio = np.dot(act_bio, full_decoders)
            else:
                xhat_bio = np.dot(act_bio, oracle_decoders)  
            xhat_lif = sim.data[probe_lif]
            rmse_bio = rmse(sim.data[probe_integral][:,0], xhat_bio[:,0])
            rmse_bio2 = rmse(sim.data[probe_integral][:,1], xhat_bio[:,1])
            rmse_lif = rmse(sim.data[probe_integral][:,0], xhat_lif[:,0])
            rmse_lif2 = rmse(sim.data[probe_integral][:,1], xhat_lif[:,1])

            plt.subplot(1, 1, 1)
            plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), xhat_bio[:,1], label='bio, rmse=%.5f' % rmse_bio2)
            plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
            plt.plot(sim.trange(), xhat_lif[:,1], label='lif, rmse=%.5f' % rmse_lif2)
            if jl_dims > 0:
                plt.plot(sim.trange(), xhat_bio[:,2:], label='jm_dims')
            plt.plot(sim.trange(), sim.data[probe_integral], label='oracle')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            plt.legend()  # prop={'size':8}
            assert rmse_bio < cutoff

        return oracle_decoders

    # Try repeated soft-mux training
    w_trains = np.array([1.0, 0.0])
    seeds = np.random.RandomState(seed=bio_seed).randint(999, size=w_trains.shape)
    recurrent_decoders = np.zeros((bio_neurons, dim))
    for i in range(len(w_trains)):
        recurrent_decoders = sim(
            w_train=w_trains[i],
            decoders_bio=recurrent_decoders,
            signal_train='white_noise',
            signal_test='white_noise',
            t_train=t_train,
            t_test=t_test,
            signal_seed = seeds[i],
            plots=(w_trains[i] == 0.0))

    # oracle_decoders = sim(w_train=1.0, decoders_bio=np.zeros((bio_neurons, dim)),
    #     signal_train='white_noise', signal_test='white_noise', plots=False,
    #     signal_seed=3, t_train=t_train, t_test=t_test)
    # oracle_decoders = sim(w_train=0.0, decoders_bio=oracle_decoders,
    #     signal_train='white_noise', signal_test='white_noise', plots=True,
    #     signal_seed=123, t_train=t_train, t_test=t_test)