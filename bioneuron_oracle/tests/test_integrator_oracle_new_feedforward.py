import numpy as np

import nengo
from nengo.utils.numpy import rmse

from nengolib.signal import s

from bioneuron_oracle import BahlNeuron, OracleSolver


def test_integrator_1d(Simulator, plt):
    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 20
    tau = 0.1
    dt = 0.001
    min_rate = 150
    max_rate = 200
    radius = np.sqrt(2)
    intercept = 1.0
    n_syn = 1
    t_final = 1.0

    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    network_seed = 12
    sim_seed = 15
    post_seed = 18
    inter_seed = 21

    max_freq = 5
    rms = 0.5

    jl_rng = np.random.RandomState(seed=conn_seed)
    plot_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/'

    dim = 1
    jl_dims = 0
    jl_dim_mag = 1e-4
    reg = 0.1

    cutoff = 0.1

    def sim(w_train=0.0, d_recurrent=None, d_JL=None, d_readout=None,
        t_final=1.0, signal='sinusoids', freq=1, mag=1, seeds=1,
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

        if signal == 'sinusoids':
            amp = 2 * np.pi * mag
        elif signal == 'white_noise':
            amp = mag

        with nengo.Network(seed=network_seed) as network:

            if signal == 'sinusoids':
                stim = nengo.Node(lambda t: np.cos(amp * t))
                # stim = nengo.Node(lambda t: amp * prime_sinusoids(t, dim, t_final, f_0=p_signal))
            elif signal == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=seeds))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 radius=radius, label='pre')
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron(), label='bio')
            inter = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-intercept, intercept),
                                 radius=radius, label='inter')
            lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=dim,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-intercept, intercept),
                                 radius=radius, label='lif')
            integral = nengo.Node(size_in=dim)

            oracle_solver = OracleSolver(decoders_bio=d_full)

            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, bio[0],  # connect feedforward to non-jl_dims of bio
                            weights_bias_conn=True,
                            seed=conn_seed,
                            synapse=tau,
                            transform=amp*tau)
            nengo.Connection(pre, lif[0],
                            synapse=tau,
                            transform=amp*tau)
            # nengo.Connection(bio, bio,  # connect recurrently to normal and jl dims of bio
            #                 synapse=tau,
            #                 seed=conn_seed,
            #                 n_syn=n_syn,
            #                 solver=oracle_solver)
            # nengo.Connection(lif, lif, synapse=tau)

            nengo.Connection(stim, integral[0],  # feedforward
                            synapse=tau,
                            transform=amp*tau)
            # nengo.Connection(stim, integral,  # integrator
            #                 synapse=1/s,
            #                 transform=amp)
            # nengo.Connection(integral, inter, synapse=None)
            # nengo.Connection(inter, bio[:dim],  # oracle training connection
            #                 synapse=tau,
            #                 transform=w_train)

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_lif = nengo.Probe(lif, synapse=tau)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=tau)


        """
        Simulate the network, collect bioneuron activities and target values,
        and apply the oracle method to calculate recurrent decoders
        """
        with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)
        lpf = nengo.Lowpass(tau)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
        d_recurrent_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_integral])[0]
        if jl_dims > 0:
            d_full_new = np.hstack((d_recurrent_new, d_JL))
        else:
            d_full_new = d_recurrent_new
        d_readout_new = d_full_new

        # print 'encoders', bio.encoders
        # assert False
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
        fig.savefig(plot_dir+'dim=%s_wtrain=%s_jldims=%s_signal=%s_freq=%s_mag=%s_seed=%s.png' %
            (dim, w_train, jl_dims, signal, freq, mag, seeds))


        """
        Make a KDE plot of the bioneurons' activities
        """
        # import pandas
        # import seaborn
        # columns = ('time', 'nrn', 'act_bio', 'act_lif', 'encoder', 'x_dot_e')
        # df_list = []
        # times = np.arange(dt, t_final, dt)
        # for i in range(5):  # len(sim.data[bio.neurons])
        #     # encoder = bio.encoders[i][:dim]  # ignore JL_dims for decoding state
        #     encoder = sim.data[bio].encoders[i] ## including JL_dims
        #     for t, time in enumerate(times):
        #         act_bio_i = act_bio[t,i]
        #         act_lif_i = act_lif[t,i]
        #         value = sim.data[probe_stim][t] + sim.data[probe_integral][t]
        #         if jl_dims > 0:
        #             jl_feedback = xhat_bio[t,1:]
        #             value = np.hstack((value, jl_feedback))
        #         x_dot_e = np.dot(value, encoder)
        #         df_temp = pandas.DataFrame(
        #             [[time, i, act_bio_i, act_lif_i, encoder[0], x_dot_e]],
        #             columns=columns)
        #         df_list.append(df_temp)
        # df = pandas.concat(df_list, ignore_index=True)
        # for i in range(5):  # len(sim.data[bio.neurons])
        #     df_nrn = pandas.DataFrame(df.query("nrn==%s"%i)).reset_index()
        #     fig1, ax1, = plt.subplots(1,1)
        #     if np.sum(df_nrn['act_bio']) > 0 and np.sum(df_nrn['x_dot_e']) > 0:
        #         seaborn.kdeplot(df_nrn['x_dot_e'], df_nrn['act_bio'],
        #             cmap='Blues', shade=True, shade_lowest=False, label='bio')
        #     ax1.legend()
        #     fig1.savefig(plot_dir+'dim=%s_wtrain=%s_jldims=%s_nrn=%s_signal=%s_%s_%s_%s_kdeplot.png' %
        #         (dim, w_train, jl_dims, i, signal, freq, mag, seeds))

        return d_recurrent_new, d_JL, d_readout_new, rmse_bio


    d_recurrent_init = np.zeros((bio_neurons, dim))
    d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
    d_readout_init = np.hstack((d_recurrent_init, d_JL))

    d_recurrent_new, d_JL, d_readout_new, rmse_bio = sim(
        w_train=1.0,
        d_recurrent=d_recurrent_init,
        d_JL=d_JL,
        d_readout=d_readout_init,
        signal='sinusoids',
        freq = 2,
        mag = 2,
        seeds = 2,
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_new, d_JL, d_readout_new, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_new,
        signal='sinusoids',
        freq = 4,
        mag = 4,
        seeds = 4,
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_extra, d_JL, d_readout_extra, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_new,
        signal='sinusoids',
        freq = 3,
        mag = 3,
        seeds = 3,
        t_final=1.0,
        plot_dir=plot_dir)

    assert rmse_bio < cutoff


def test_integrator_2d(Simulator, plt):
    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 100
    tau = 0.1
    dt = 0.001
    min_rate = 150
    max_rate = 200
    radius = np.sqrt(2)
    intercept = 1.0
    n_syn = 1
    t_final = 1.0

    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    network_seed = 12
    sim_seed = 15
    post_seed = 18
    inter_seed = 21

    max_freq = 5
    rms = 0.5

    jl_rng = np.random.RandomState(seed=conn_seed)
    plot_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/plots/'

    dim = 2
    jl_dims = 0
    jl_dim_mag = 1e-4
    reg = 0.01

    cutoff = 0.1

    def sim(w_train=0.0, d_recurrent=None, d_JL=None, d_readout=None,
        t_final=1.0, signal='sinusoids', freq=[1,2], mag=[1,2], seeds=[1,2],
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

        if signal == 'sinusoids':
            amp = 2 * np.pi * mag[0]
            amp2 = 2 * np.pi * mag[1]
        elif signal == 'white_noise':
            amp = mag[0]
            amp2 = mag[1]

        with nengo.Network(seed=network_seed) as network:

            if signal == 'sinusoids':
                stim = nengo.Node(lambda t: np.cos(amp * t))
                stim2 = nengo.Node(lambda t: np.cos(amp2 * t))
                # stim = nengo.Node(lambda t: prime_sinusoids(t, dim/2, t_final, f_0=p_signal[0]))
                # stim2 = nengo.Node(lambda t: prime_sinusoids(t, dim/2, t_final, f_0=p_signal[1]))
            elif signal == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=freq[0]))
                stim2 = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=freq[1]))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 radius=radius, label='pre')
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                 seed=bio_seed, neuron_type=BahlNeuron(), label='bio')
            inter = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=nengo.LIF(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-intercept, intercept),
                                 radius=radius, label='inter')
            lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=dim,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-intercept, intercept),
                                 radius=radius, label='lif')
            integral = nengo.Node(size_in=dim)

            oracle_solver = OracleSolver(decoders_bio=d_full)

            nengo.Connection(stim, pre[0], synapse=None)
            nengo.Connection(stim2, pre[1], synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(pre[0], bio[0],
                            weights_bias_conn=True,
                            seed=conn_seed,
                            synapse=tau,
                            transform=amp*tau)
            nengo.Connection(pre[1], bio[1],
                            weights_bias_conn=False,
                            seed=conn_seed,
                            synapse=tau,
                            transform=amp2*tau)
            nengo.Connection(pre[0], lif[0],
                            synapse=tau,
                            transform=amp*tau)
            nengo.Connection(pre[1], lif[1],
                            synapse=tau,
                            transform=amp2*tau)
            # connect recurrently to normal and jl dims of bio
            # nengo.Connection(bio, bio,
                            # synapse=tau,
                            # seed=conn_seed,
                            # n_syn=n_syn,
                            # solver=oracle_solver)
            # nengo.Connection(lif, lif, synapse=tau)

            nengo.Connection(stim, integral[0],
                            synapse=tau,
                            transform=amp*tau)  # feedforward
            nengo.Connection(stim2, integral[1],
                            synapse=tau,
                            transform=amp2*tau)  # feedforward
            # nengo.Connection(stim, integral[0],
                            # synapse=1/s, transform=amp)  # integrator
            # nengo.Connection(stim2, integral[1],
                            # synapse=1/s, transform=amp2)  # integrator
            # nengo.Connection(integral, inter, synapse=None)
            # nengo.Connection(inter, bio[:dim],  # oracle training connection
                            # synapse=tau,
                            # transform=w_train)  # todo: does this nullify weights?

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_stim2 = nengo.Probe(stim2, synapse=None)
            probe_lif = nengo.Probe(lif, synapse=tau)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=tau)


        """
        Simulate the network, collect bioneuron activities and target values,
        and apply the oracle method to calculate recurrent decoders
        """
        with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
            sim.run(t_final)
        lpf = nengo.Lowpass(tau)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
        d_recurrent_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_integral])[0]
        if jl_dims > 0:
            d_full_new = np.hstack((d_recurrent_new, d_JL))
        else:
            d_full_new = d_recurrent_new
        d_readout_new = d_full_new

        # print 'encoders', bio.encoders
        # assert False
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
            (dim, w_train, jl_dims, signal, freq[0], freq[1]))

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

        """
        Make a KDE plot of the bioneurons' activities
        """
        # import pandas
        # import seaborn
        # columns = ('time', 'nrn', 'act_bio', 'act_lif', 'encoder_0', 'encoder_1', 'x_dot_e')
        # df_list = []
        # times = np.arange(dt, t_final, dt)
        # for i in range(5):  # len(sim.data[bio.neurons])
        #     encoder = sim.data[bio].encoders[i]  # including JL_dims
        #     for t, time in enumerate(times):
        #         act_bio_i = act_bio[t,i]
        #         act_lif_i = act_lif[t,i]
        #         value = np.array([
        #             sim.data[probe_stim][t,0] + sim.data[probe_integral][t, 0],
        #             sim.data[probe_stim2][t,0] + sim.data[probe_integral][t, 1]])
        #         if jl_dims > 0:
        #             jl_feedback = xhat_bio[t,2:]
        #             value = np.hstack((value, jl_feedback))
        #         x_dot_e = np.dot(value, encoder)
        #         df_temp = pandas.DataFrame(
        #             [[time, i, act_bio_i, act_lif_i, encoder[0], encoder[1], x_dot_e]],
        #             columns=columns)
        #         df_list.append(df_temp)
        # df = pandas.concat(df_list, ignore_index=True)
        # for i in range(5):  # len(sim.data[bio.neurons])
        #     df_nrn = pandas.DataFrame(df.query("nrn==%s"%i)).reset_index()
        #     fig1, ax1, = plt.subplots(1,1)
        #     if np.sum(df_nrn['act_bio']) > 0 and np.sum(df_nrn['x_dot_e']) > 0:
        #         seaborn.kdeplot(df_nrn['x_dot_e'], df_nrn['act_bio'],
        #             cmap='Blues', shade=True, shade_lowest=False, label='bio')
        #     # if np.sum(df_nrn['act_lif']) > 0 and np.sum(df_nrn['x_dot_e']) > 0:
        #     #     seaborn.kdeplot(df_nrn['x_dot_e'], df_nrn['act_lif'],
        #     #         cmap='Reds', shade=True, shade_lowest=False, label='lif')
        #     ax1.legend()
        #     fig1.savefig(plot_dir+'dim=%s_wtrain=%s_jldims=%s_nrn=%s_%s_%s_%s_kdeplot.png' %
        #         (dim, w_train, jl_dims, i, signal, freq[0], freq[1]))

        return d_recurrent_new, d_JL, d_readout_new, rmse_bio


    d_recurrent_init = np.zeros((bio_neurons, dim))
    d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
    d_readout_init = np.hstack((d_recurrent_init, d_JL))

    d_recurrent_new, d_JL, d_readout_new, rmse_bio = sim(
        w_train=1.0,
        d_recurrent=d_recurrent_init,
        d_JL=d_JL,
        d_readout=d_readout_init,
        signal='sinusoids',
        freq = [2, 4],
        mag = [2, 4],
        seeds = [2, 4],
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_new, d_JL, d_readout_new, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_new,
        signal='sinusoids',
        freq = [4, 1],
        mag = [4, 1],
        seeds = [4, 1],
        t_final=t_final,
        plot_dir=plot_dir)
    d_recurrent_extra, d_JL, d_readout_extra, rmse_bio = sim(
        w_train=0.0,
        d_recurrent=d_recurrent_new,
        d_JL=d_JL,
        d_readout=d_readout_new,
        signal='sinusoids',
        freq = [3, 2],
        mag = [3, 2],
        seeds = [3, 2],
        t_final=1.0,
        plot_dir=plot_dir)

    assert rmse_bio < cutoff
    assert False