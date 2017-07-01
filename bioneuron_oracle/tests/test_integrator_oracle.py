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
    n_syn = 1
    t_final = 1.0

    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    network_seed = 12
    sim_seed = 15
    post_seed = 18
    inter_seed = 21

    jl_rng = np.random.RandomState(seed=conn_seed)
    plot_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/'

    dim = 1
    jl_dims = 3
    jl_dim_mag = 2e-4

    cutoff = 0.1

    def sim(w_train=0.0, d_recurrent=None, d_JL=None, d_readout=None,
        t_final=1.0, signal='prime_sinusoids', p_signal=1,
        plot_dir='/home/pduggins/'):

        """
        Define the network, with extra dimensions in bio for JL-feedback
        Set the recurrent decoders set to decoders_bio,
        which are initially set to zero or the result of a previous sim() call
        """
        if jl_dims > 0:
            d_full = np.hstack(((1.0 - w_train) * d_recurrent, d_JL))
        else:
            d_full = (1.0 - w_train) * d_recurrent
        amp = 2 * np.pi * p_signal
        rms = 3.0
        max_freq = 5
        radius = max(amp, rms)  # changing radius changes pre-bio weights (bias emulation)

        with nengo.Network(seed=network_seed) as network:

            if signal == 'prime_sinusoids':
                stim = nengo.Node(lambda t: amp * prime_sinusoids(t, dim, t_final, f_0=p_signal))
            elif signal == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=p_signal))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 radius=radius, label='pre')
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1),
                                 label='bio')
            inter = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts,
                                 label='inter')
            lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            integral = nengo.Node(size_in=dim)

            oracle_solver = OracleSolver(decoders_bio=d_full)

            nengo.Connection(stim, pre, synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(pre, bio[:dim], weights_bias_conn=True,
                             seed=conn_seed, synapse=tau_neuron,
                             transform=tau_neuron)
            nengo.Connection(pre, lif,
                             synapse=tau_nengo, transform=tau_nengo)
            # connect recurrently to normal and jl dims of bio
            nengo.Connection(bio, bio, synapse=tau_neuron, seed=conn_seed,
                             n_syn=n_syn, solver=oracle_solver)
            nengo.Connection(lif, lif, synapse=tau_nengo)

            nengo.Connection(stim, integral, synapse=1/s)  # integrator
            # if w_train != 0.0:
            nengo.Connection(integral, inter, synapse=None)
            nengo.Connection(inter, bio[:dim],  # oracle training connection
                             synapse=tau_neuron, transform=w_train)  # todo: does this nullify weights?

            probe_lif = nengo.Probe(lif, synapse=tau_nengo)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=tau_nengo)


        """
        Simulate the network, collect bioneuron activities and target values,
        and apply the oracle method to calculate recurrent decoders
        """
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        d_recurrent_new = nengo.solvers.LstsqL2(reg=0.01)(act_bio, sim.data[probe_integral])[0]
        if jl_dims > 0:
            d_full_new = np.hstack((d_recurrent_new, d_JL))
        else:
            d_full_new = d_recurrent_new
        d_readout_new = d_full_new

        # print 'w_train', w_train
        # for nrn in sim.data[bio.neurons]:
        #     print 'neuron', nrn
        #     for conn_pre in nrn.synapses.iterkeys():
        #         print 'conn_pre', conn_pre
        #         for pre in range(nrn.synapses[conn_pre].shape[0]):
        #             for syn in range(nrn.synapses[conn_pre][pre].shape[0]):
        #                 print 'syn weight', nrn.synapses[conn_pre][pre][syn].weight

        """
        Use either the old or new decoders to estimate the bioneurons' outputs for plotting
        """
        xhat_bio = np.dot(act_bio, d_readout)
        xhat_lif = sim.data[probe_lif]
        rmse_bio = rmse(sim.data[probe_integral][:,0], xhat_bio[:,0])
        rmse_lif = rmse(sim.data[probe_integral], xhat_lif)

        fig, ax1 = plt.subplots(1,1)
        ax1.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
        ax1.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
        if jl_dims > 0:
            ax1.plot(sim.trange(), xhat_bio[:,1:], label='jm_dims')
        ax1.plot(sim.trange(), sim.data[probe_integral], label='oracle')
        ax1.set(xlabel='time (s)', ylabel='$\hat{x}(t)$')
        ax1.legend()
        fig.savefig(plot_dir+'dim=%s_wtrain=%s_jldims=%s_%s_%s.png' %
            (dim, w_train, jl_dims, signal, p_signal))

        # plt.subplot(1, 1, 1)
        # plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
        # plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
        # if jl_dims > 0:
        #     plt.plot(sim.trange(), xhat_bio[:,1:], label='jm_dims')
        # plt.plot(sim.trange(), sim.data[probe_integral], label='oracle')
        # plt.xlabel('time (s)')
        # plt.ylabel('$\hat{x}(t)$')
        # plt.title('decode')
        # plt.legend()  # prop={'size':8}
        # assert rmse_bio < cutoff

        return d_recurrent_new, d_JL, d_readout_new, rmse_bio


    d_recurrent_init = np.zeros((bio_neurons, dim))
    d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
    d_readout_init = np.hstack((d_recurrent_init, d_JL))

    d_recurrent_new, d_JL, d_readout_extra, rmse_bio = sim(
        w_train=1.0,
        d_recurrent=d_recurrent_init,
        d_JL=d_JL,
        d_readout=d_readout_init,
        signal='prime_sinusoids',
        p_signal = 1.0,  # float for f_0 for sinusoid, int for seed for whitesignal
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_extra, d_JL, d_readout_new, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_init,
        signal='prime_sinusoids',
        p_signal = 1.5,  # float for f_0 for sinusoid, int for seed for whitesignal
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_extra, d_JL, d_readout_extra, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_new,
        signal='prime_sinusoids',
        p_signal = 2.0,  # float for f_0 for sinusoid, int for seed for whitesignal
        t_final=t_final,
        plot_dir=plot_dir)

    assert rmse_bio < cutoff
    assert False



def test_integrator_2d(Simulator, plt):
    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 100
    tau_nengo = 0.1
    tau_neuron = 0.1
    dt_nengo = 0.001
    min_rate = 150
    max_rate = 200
    n_syn = 1
    t_final = 1.0

    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    network_seed = 12
    sim_seed = 15
    post_seed = 18
    inter_seed = 21

    jl_rng = np.random.RandomState(seed=conn_seed)
    plot_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/'

    dim = 2
    jl_dims = 1
    jl_dim_mag = 3e-4

    cutoff = 0.1

    def sim(w_train=0.0, d_recurrent=None, d_JL=None, d_readout=None,
        t_final=1.0, signal='prime_sinusoids', p_signal=[1,2],
        plot_dir='/home/pduggins/'):

        """
        Define the network, with extra dimensions in bio for JL-feedback
        Set the recurrent decoders set to decoders_bio,
        which are initially set to zero or the result of a previous sim() call
        """
        if jl_dims > 0:
            d_full = np.hstack(((1.0 - w_train) * d_recurrent, d_JL))
        else:
            d_full = (1.0 - w_train) * d_recurrent
        amp = 2 * np.pi * p_signal[0]
        amp2 = 2 * np.pi * p_signal[1]
        rms = 3.0
        max_freq = 5
        radius = max(amp, rms)  # changing radius changes pre-bio weights (bias emulation)

        with nengo.Network(seed=network_seed) as network:

            if signal == 'prime_sinusoids':
                stim = nengo.Node(lambda t: amp * prime_sinusoids(t, dim/2, t_final, f_0=p_signal[0]))
                stim2 = nengo.Node(lambda t: amp2 * prime_sinusoids(t, dim/2, t_final, f_0=p_signal[1]))
            elif signal == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=p_signal[0]))
                stim2 = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=p_signal[1]))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 radius=radius, label='pre')
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1),
                                 label='bio')
            inter = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts,
                                 label='inter')
            lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            integral = nengo.Node(size_in=dim)

            oracle_solver = OracleSolver(decoders_bio=d_full)

            nengo.Connection(stim, pre[0], synapse=None)
            nengo.Connection(stim2, pre[1], synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(pre, bio[:dim], weights_bias_conn=True,
                             seed=conn_seed, synapse=tau_neuron,
                             transform=tau_neuron)
            nengo.Connection(pre, lif,
                             synapse=tau_nengo, transform=tau_nengo)
            # connect recurrently to normal and jl dims of bio
            nengo.Connection(bio, bio, synapse=tau_neuron, seed=conn_seed,
                             n_syn=n_syn, solver=oracle_solver)
            nengo.Connection(lif, lif, synapse=tau_nengo)

            nengo.Connection(stim, integral[0], synapse=1/s)  # integrator
            nengo.Connection(stim2, integral[1], synapse=1/s)  # integrator
            nengo.Connection(integral, inter, synapse=None)
            nengo.Connection(inter, bio[:dim],  # oracle training connection
                             synapse=tau_neuron, transform=w_train)  # todo: does this nullify weights?

            probe_lif = nengo.Probe(lif, synapse=tau_nengo)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=tau_nengo)


        """
        Simulate the network, collect bioneuron activities and target values,
        and apply the oracle method to calculate recurrent decoders
        """
        with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)
        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        d_recurrent_new = nengo.solvers.LstsqL2(reg=0.01)(act_bio, sim.data[probe_integral])[0]
        if jl_dims > 0:
            d_full_new = np.hstack((d_recurrent_new, d_JL))
        else:
            d_full_new = d_recurrent_new
        d_readout_new = d_full_new

        # print 'w_train', w_train
        # for nrn in sim.data[bio.neurons]:
        #     print 'neuron', nrn
        #     for conn_pre in nrn.synapses.iterkeys():
        #         print 'conn_pre', conn_pre
        #         for pre in range(nrn.synapses[conn_pre].shape[0]):
        #             for syn in range(nrn.synapses[conn_pre][pre].shape[0]):
        #                 print 'syn weight', nrn.synapses[conn_pre][pre][syn].weight

        """
        Use either the old or new decoders to estimate the bioneurons' outputs for plotting
        """
        xhat_bio = np.dot(act_bio, d_readout)
        xhat_lif = sim.data[probe_lif]
        rmse_bio = rmse(sim.data[probe_integral][:,0], xhat_bio[:,0])
        rmse_bio2 = rmse(sim.data[probe_integral][:,1], xhat_bio[:,1])
        rmse_lif = rmse(sim.data[probe_integral][:,0], xhat_lif[:,0])
        rmse_lif2 = rmse(sim.data[probe_integral][:,1], xhat_lif[:,1])

        fig, ax1 = plt.subplots(1,1)
        ax1.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
        ax1.plot(sim.trange(), xhat_bio[:,1], label='bio, rmse=%.5f' % rmse_bio2)
        ax1.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
        ax1.plot(sim.trange(), xhat_lif[:,1], label='lif, rmse=%.5f' % rmse_lif2)
        if jl_dims > 0:
            ax1.plot(sim.trange(), xhat_bio[:,2:], label='jm_dims')
        ax1.plot(sim.trange(), sim.data[probe_integral], label='oracle')
        ax1.set(xlabel='time (s)', ylabel='$\hat{x}(t)$')
        ax1.legend()
        fig.savefig(plot_dir+'dim=%s_wtrain=%s_jldims=%s_%s_%s_%s.png' %
            (dim, w_train, jl_dims, signal, p_signal[0], p_signal[1]))

        # plt.subplot(1, 1, 1)
        # plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
        # plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
        # if jl_dims > 0:
        #     plt.plot(sim.trange(), xhat_bio[:,1:], label='jm_dims')
        # plt.plot(sim.trange(), sim.data[probe_integral], label='oracle')
        # plt.xlabel('time (s)')
        # plt.ylabel('$\hat{x}(t)$')
        # plt.title('decode')
        # plt.legend()  # prop={'size':8}
        # assert rmse_bio < cutoff

        return d_recurrent_new, d_JL, d_readout_new, rmse_bio


    d_recurrent_init = np.zeros((bio_neurons, dim))
    d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
    d_readout_init = np.hstack((d_recurrent_init, d_JL))

    d_recurrent_new, d_JL, d_readout_extra, rmse_bio = sim(
        w_train=1.0,
        d_recurrent=d_recurrent_init,
        d_JL=d_JL,
        d_readout=d_readout_init,
        signal='prime_sinusoids',
        p_signal = [1.0, 2.0],  # float for f_0 for sinusoid, int for seed for whitesignal
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_extra, d_JL, d_readout_new, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_init,
        signal='prime_sinusoids',
        p_signal = [1.0, 2.0],  # float for f_0 for sinusoid, int for seed for whitesignal
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_extra, d_JL, d_readout_extra, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_new,
        signal='prime_sinusoids',
        p_signal = [0.5, 1.5],  # float for f_0 for sinusoid, int for seed for whitesignal
        t_final=t_final,
        plot_dir=plot_dir)

    assert rmse_bio < cutoff
    assert False





# def test_integrator_2d(Simulator, plt):

#     # Nengo Parameters
#     pre_neurons = 100
#     bio_neurons = 200
#     tau_nengo = 0.1
#     tau_neuron = 0.1
#     dt_nengo = 0.001
#     min_rate = 150
#     max_rate = 200
#     pre_seed = 3
#     bio_seed = 6
#     conn_seed = 9
#     network_seed = 12
#     sim_seed = 15
#     post_seed = 18
#     inter_seed = 21
#     t_train = 5.0
#     t_test = 1.0
#     n_syn = 1
#     cutoff = 0.1

#     dim = 2
#     jl_dims = 2
#     jl_dim_mag = 1e-4
            
#     f_0 = 1.0
#     f_0_2 = 2.0
#     amp = 2*np.pi * f_0
#     amp2 = 2*np.pi * f_0_2
#     max_freq = 5
#     rms = 3.0
#     radius = amp2

#     def sim(w_train, decoders_bio=None, plots=False, t_train=1.0, t_test=1.0,
#             signal_train='prime_sinusoids', signal_test='white_noise', signal_seed=123):

#         """
#         Define the network, with extra dimensions in bio for JL-feedback
#         Set the recurrent decoders set to decoders_bio,
#         which are initially set to zero or the result of a previous sim() call
#         """
#         with nengo.Network(seed=network_seed) as network:

#             if signal_train == 'prime_sinusoids':
#                 stim = nengo.Node(lambda t: amp * prime_sinusoids(t, dim/2, t_train, f_0=f_0))
#                 stim2 = nengo.Node(lambda t: amp2 * prime_sinusoids(t, dim/2, t_train, f_0=f_0_2))
#             elif signal_train == 'step_input':
#                 stim = nengo.Node(lambda t: step_input(t, dim/2, t_train, dt_nengo))
#                 stim2 = nengo.Node(lambda t: step_input(t, dim/2, t_train, dt_nengo))
#             elif signal_train == 'white_noise':
#                 stim = nengo.Node(nengo.processes.WhiteSignal(
#                     period=t_train, high=max_freq, rms=rms, seed=signal_seed))
#                 # stim = nengo.Node(lambda t: equalpower(
#                 #                       t, dt_nengo, t_train, max_freq, dim/2,
#                 #                       mean=0.0, std=3.0, seed=signal_seed))
#                 stim2 = nengo.Node(nengo.processes.WhiteSignal(
#                     period=t_train, high=max_freq, rms=rms, seed=2*signal_seed))
#                 # stim2 = nengo.Node(lambda t: equalpower(
#                 #                       t, dt_nengo, t_train, max_freq, dim/2,
#                 #                       mean=0.0, std=3.0, seed=2*signal_seed))

#             pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                                  seed=pre_seed, neuron_type=nengo.LIF(),
#                                  radius=radius)
#             bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
#                                  seed=bio_seed, neuron_type=BahlNeuron(),
#                                  max_rates=nengo.dists.Uniform(min_rate, max_rate),
#                                  intercepts=nengo.dists.Uniform(-1, 1))
#             inter = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=nengo.LIF(),
#                                  max_rates=bio.max_rates, intercepts=bio.intercepts)
#             lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
#                                  seed=bio_seed, neuron_type=nengo.LIF(),
#                                  max_rates=bio.max_rates, intercepts=bio.intercepts)
#             integral = nengo.Node(size_in=dim)

#             oracle_solver = OracleSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
#             if jl_dims > 0:
#                 # TODO: magnitude should scale with n_neurons (maybe 1./n^2)?
#                 jl_decoders = np.random.RandomState(seed=conn_seed).randn(
#                     bio_neurons, jl_dims) * jl_dim_mag
#                 oracle_solver.decoders_bio = np.hstack(
#                     (oracle_solver.decoders_bio, jl_decoders))

#             nengo.Connection(stim, pre[0], synapse=None)
#             nengo.Connection(stim2, pre[1], synapse=None)
#             # connect feedforward to non-jl_dims of bio
#             nengo.Connection(pre, bio[:dim], weights_bias_conn=True,
#                              seed=conn_seed, synapse=tau_neuron,
#                              transform=tau_neuron)
#             nengo.Connection(pre, lif,
#                              synapse=tau_nengo, transform=tau_nengo)
#             nengo.Connection(bio, bio, synapse=tau_neuron, seed=conn_seed,
#                              n_syn=n_syn, solver=oracle_solver)
#             nengo.Connection(lif, lif, synapse=tau_nengo)

#             nengo.Connection(stim, integral[0], synapse=1/s)  # integrator
#             nengo.Connection(stim2, integral[1], synapse=1/s)  # integrator
#             nengo.Connection(integral, inter, synapse=None)
#             nengo.Connection(inter, bio[:dim],  # oracle connection
#                              synapse=tau_neuron, transform=w_train, seed=conn_seed)

#             probe_lif = nengo.Probe(lif, synapse=tau_nengo)
#             probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
#             probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
#             probe_integral = nengo.Probe(integral, synapse=tau_nengo)

#         """
#         Simulate the network, collect bioneuron activities and target values,
#         and apply the oracle method to calculate recurrent decoders
#         """
#         with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
#             sim.run(t_train)
#         lpf = nengo.Lowpass(tau_nengo)
#         act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#         oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
#         oracle_decoders = oracle_solver(act_bio, sim.data[probe_integral])[0]
#         if jl_dims > 0:
#             full_decoders = np.hstack((oracle_decoders, jl_decoders))

#         """
#         Run the simulation on a new signal, and use the full_decoders
#         calculated above to estimate the bioneurons' outputs for plotting
#         """
#         if plots:
#             with network:
#                 if signal_test == 'prime_sinusoids':
#                     stim.output = lambda t: amp2 * prime_sinusoids(t, dim/2, t_test, f_0=f_0_2)
#                     stim2.output = lambda t: amp * prime_sinusoids(t, dim/2, t_test, f_0=f_0)
#                 elif signal_test == 'step_input':
#                     stim.output = lambda t: step_input(t, dim/2, t_test, dt_nengo)
#                     stim2.output = lambda t: step_input(t, dim/2, t_test, dt_nengo)
#                 elif signal_test == 'white_noise':
#                     stim.output = nengo.processes.WhiteSignal(
#                         period=t_test, high=max_freq, rms=rms, seed=3*signal_seed)
#                     # stim.output = lambda t: equalpower(
#                     #                       t, dt_nengo, t_test, max_freq, dim/2,
#                     #                       mean=0.0, std=3.0, seed=3*signal_seed)
#                     stim2.output = nengo.processes.WhiteSignal(
#                         period=t_test, high=max_freq, rms=rms, seed=4*signal_seed)
#                     # stim2.output = lambda t: equalpower(
#                     #                       t, dt_nengo, t_test, max_freq, dim/2,
#                     #                       mean=0.0, std=3.0, seed=4*signal_seed)

#             with Simulator(network, dt=dt_nengo, progress_bar=False, seed=sim_seed) as sim:
#                 sim.run(t_test)
#             act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
#             if jl_dims > 0:
#                 xhat_bio = np.dot(act_bio, full_decoders)
#             else:
#                 xhat_bio = np.dot(act_bio, oracle_decoders)  
#             xhat_lif = sim.data[probe_lif]
#             rmse_bio = rmse(sim.data[probe_integral][:,0], xhat_bio[:,0])
#             rmse_bio2 = rmse(sim.data[probe_integral][:,1], xhat_bio[:,1])
#             rmse_lif = rmse(sim.data[probe_integral][:,0], xhat_lif[:,0])
#             rmse_lif2 = rmse(sim.data[probe_integral][:,1], xhat_lif[:,1])

#             fig, ax1 = plt.subplots(1,1)
#             ax1.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
#             ax1.plot(sim.trange(), xhat_bio[:,1], label='bio, rmse=%.5f' % rmse_bio2)
#             ax1.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
#             ax1.plot(sim.trange(), xhat_lif[:,1], label='lif, rmse=%.5f' % rmse_lif2)
#             if jl_dims > 0:
#                 ax1.plot(sim.trange(), xhat_bio[:,2:], label='jm_dims')
#             ax1.plot(sim.trange(), sim.data[probe_integral], label='oracle')
#             ax1.set(xlabel='time (s)', ylabel='$\hat{x}(t)$')
#             ax1.legend()
#             fig.savefig(plots)

#             # plt.subplot(1, 1, 1)
#             # plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
#             # plt.plot(sim.trange(), xhat_bio[:,1], label='bio, rmse=%.5f' % rmse_bio2)
#             # plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
#             # plt.plot(sim.trange(), xhat_lif[:,1], label='lif, rmse=%.5f' % rmse_lif2)
#             # if jl_dims > 0:
#             #     plt.plot(sim.trange(), xhat_bio[:,2:], label='jm_dims')
#             # plt.plot(sim.trange(), sim.data[probe_integral], label='oracle')
#             # plt.xlabel('time (s)')
#             # plt.ylabel('$\hat{x}(t)$')
#             # plt.title('decode')
#             # plt.legend()  # prop={'size':8}

#         return oracle_decoders

#     # Try repeated soft-mux training
#     w_trains = np.array([1.0, 0.0])

#     seeds = np.random.RandomState(seed=bio_seed).randint(999, size=w_trains.shape)
#     recurrent_decoders = np.zeros((bio_neurons, dim))
#     for i in range(len(w_trains)):
#         recurrent_decoders = sim(
#             w_train=w_trains[i],
#             decoders_bio=recurrent_decoders,
#             signal_train='prime_sinusoids',
#             signal_test='prime_sinusoids',
#             t_train=t_train,
#             t_test=t_test,
#             signal_seed = seeds[i],
#             plots='/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/JL_dims_2_nrn_200_5s_2d/sinusoid_sinusoid.png')

#     seeds = np.random.RandomState(seed=bio_seed).randint(999, size=w_trains.shape)
#     recurrent_decoders = np.zeros((bio_neurons, dim))
#     for i in range(len(w_trains)):
#         recurrent_decoders = sim(
#             w_train=w_trains[i],
#             decoders_bio=recurrent_decoders,
#             signal_train='white_noise',
#             signal_test='white_noise',
#             t_train=t_train,
#             t_test=t_test,
#             signal_seed = seeds[i],
#             plots='/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/JL_dims_2_nrn_200_5s_2d/whitenoise_whitenoise.png')

#     seeds = np.random.RandomState(seed=bio_seed).randint(999, size=w_trains.shape)
#     recurrent_decoders = np.zeros((bio_neurons, dim))
#     for i in range(len(w_trains)):
#         recurrent_decoders = sim(
#             w_train=w_trains[i],
#             decoders_bio=recurrent_decoders,
#             signal_train='white_noise',
#             signal_test='prime_sinusoids',
#             t_train=t_train,
#             t_test=t_test,
#             signal_seed = seeds[i],
#             plots='/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/JL_dims_2_nrn_200_5s_2d/whitenose_sinusoid.png')

#     seeds = np.random.RandomState(seed=bio_seed).randint(999, size=w_trains.shape)
#     recurrent_decoders = np.zeros((bio_neurons, dim))
#     for i in range(len(w_trains)):
#         recurrent_decoders = sim(
#             w_train=w_trains[i],
#             decoders_bio=recurrent_decoders,
#             signal_train='prime_sinusoids',
#             signal_test='white_noise',
#             t_train=t_train,
#             t_test=t_test,
#             signal_seed = seeds[i],
#             plots='/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/JL_dims_2_nrn_200_5s_2d/sinusoid_whitenoise.png')