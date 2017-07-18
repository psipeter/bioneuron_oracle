from functools32 import lru_cache

import numpy as np

from scipy.stats import linregress

import neuron

import nengo
from nengo.utils.numpy import rmse
from nengo.utils.matplotlib import rasterplot

from seaborn import set_palette, color_palette, tsplot

from bioneuron_oracle import (BahlNeuron, TrainedSolver, OracleSolver, spike_train)
from bioneuron_oracle.builder import Bahl, ExpSyn

def test_somatic_attenuation(Simulator, plt):

	locations = np.arange(0,1,0.01)
	weight = 1e-3
	tau = 0.05
	t_spike = 250  # ms, run out transients
	t_final = 500  # ms
	voltages = []
	times = []

	for l in locations:
		bioneuron = Bahl()
		bioneuron.cell.apical.nseg = 20
		section = bioneuron.cell.apical(l)
		synapse = ExpSyn(section, weight, tau, l)
		bioneuron.synapses[l] = synapse
		neuron.init()
		bioneuron.synapses[l].spike_in.event(t_spike)
		neuron.run(t_final)
		voltages.append(np.array(bioneuron.v_record))
		times.append(np.array(bioneuron.t_record))
		bioneuron.cleanup()

	# delta v measured from time of spike initiation
	delta_v = np.array([np.amax(voltages[i][int(t_spike / neuron.h.dt):]) -
					np.amin(voltages[i][int(t_spike / neuron.h.dt):])
				for i in range(len(locations))])
	# normalize
	delta_v /= np.max(delta_v)

	# perform a linear regression on the attenuation curve and check the slope
	# and r-squared
	slope, intercept, r_value, p_value, std_err = linregress(locations, delta_v)
	assert -0.4 < slope < -0.3
	assert 0.95 < intercept < 1.0
	assert r_value**2 > 0.9

	# plt.subplot(1,1,1)
	# for i in range(len(locations)):
	# 	plt.plot(times[0], voltages[i], label='syn @ %s' % locations[i]) 
	# plt.xlabel('time (ms)')
	# plt.ylabel('$V_{soma}(t)$')
	# plt.legend()

	plt.subplot(1,1,1)
	plt.scatter(locations, delta_v)
	plt.plot(locations, slope * locations + intercept,
		label='$\hat{\Delta V} = %.3f %s + %.3f$, $r^2=%.3f$'
		%(slope, 'd', intercept, r_value**2))
	plt.xlabel('$d_{apical}$')
	# plt.ylabel('$\Delta V_{soma}$')
	plt.ylabel('Somatic Attenuation \n($\Delta V_{soma} \quad / \quad \Delta V_{soma}^{d=0}$)')
	plt.legend()


def test_tuning_curves(Simulator, plt):

    """
    For trained weights:
        Train synaptic weights using spike_train and see how well they match

    For untrained weights:
        Naively assume that using the optimal decoders from pre
        in conjuction with the encoders, gains, and emulated biases
        from bio (the same as from ideal) will construct a weight
        matrix that performs spike matching.
        The synaptic weights will also be multiplied by an attenuation
        factor to account for the known attenuation that occurs for a 
        single spike traveling to the soma, which is approximated by
        v(d) = 1.0 - 0.355*d   ==>   attenuation(d) = 1/0.335 d = 2.817d

    NOTE: only works when gains, biases, and encoders are generated from 
    using the LIF methods in the builder
    """

    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 10
    tau = 0.01
    tau_readout = 0.01
    dt = 0.001
    min_rate = 150
    max_rate = 200
    radius = 1
    bio_radius = 1
    n_syn = 1

    pre_seed = 1
    bio_seed = 6
    conn_seed = 3
    network_seed = 4
    sim_seed = 5
    post_seed = 6
    inter_seed = 7

    max_freq = 5
    rms = 1.0

    dim = 1
    reg = 0.1
    t_final = 10.0
    cutoff = 0.1

    evo_params = {
        'dt': dt,
        'tau_readout': tau_readout,
        'sim_seed': sim_seed,
        'n_processes': 10,
        'popsize': 10,
        'generations' : 10,
        'w_0': 1e-1,
        'delta_w' :1e-1,
        'evo_seed' :1,
        'evo_t_final' :1.0,
    }

    def sim(
        w_trained,
        d_ideal,
        evo_params,
        readout = 'LIF',
        signal='sinusoids',
        t_final=1.0,
        freq=1,
        seeds=1,
        transform=1,
        train=False,
        plot=False):

        """
        Define the network
        """
        with nengo.Network(seed=network_seed) as network:

            if signal == 'sinusoids':
                stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq * t))
            elif signal == 'white_noise':
                stim = nengo.Node(nengo.processes.WhiteSignal(
                    period=t_final, high=max_freq, rms=rms, seed=seeds))
            elif signal == 'step':
                stim = nengo.Node(lambda t:
                    np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
            elif signal == 'constant':
                stim = nengo.Node(lambda t: freq)

            pre = nengo.Ensemble(
                n_neurons=pre_neurons,
                dimensions=dim,
                seed=pre_seed,
                neuron_type=nengo.LIF(),
                radius=radius,
                label='pre')
            untrained_bio = nengo.Ensemble(
                n_neurons=bio_neurons,
                dimensions=dim,
                seed=bio_seed,
                neuron_type=BahlNeuron(),
                # neuron_type=nengo.LIF(),
                radius=bio_radius,
                max_rates=nengo.dists.Uniform(min_rate, max_rate),
                label='untrained_bio')
            trained_bio = nengo.Ensemble(
                n_neurons=bio_neurons,
                dimensions=dim,
                seed=bio_seed,
                neuron_type=BahlNeuron(),
                # neuron_type=nengo.LIF(),
                radius=bio_radius,
                max_rates=nengo.dists.Uniform(min_rate, max_rate),
                label='trained_bio')
            lif = nengo.Ensemble(
                n_neurons=bio_neurons,
                dimensions=dim,
                seed=bio_seed,
                max_rates=nengo.dists.Uniform(min_rate, max_rate),
                radius=bio_radius,
                neuron_type=nengo.LIF(),
                label='lif')
            oracle = nengo.Node(size_in=dim)
            temp = nengo.Node(size_in=dim)

            """
            The OracleSolver is used to manually set weights when no training is occuring
            The TrainedSolver is used to train weights using 1+$lambda$ ES
            """
            untrained_solver = OracleSolver(decoders_bio = d_ideal)
            trained_solver = TrainedSolver(weights_bio = w_trained)

            nengo.Connection(stim, pre, synapse=None)
            conn_untrained = nengo.Connection(pre, untrained_bio,
                weights_bias_conn=True,
                seed=conn_seed,
                synapse=tau,
                transform=transform,
                n_syn=n_syn)
            conn_trained = nengo.Connection(pre, trained_bio,
                seed=conn_seed,
                synapse=tau,
                transform=transform,
                trained_weights=True,
                solver = trained_solver,
                n_syn=n_syn)

            conn_ideal = nengo.Connection(pre, lif,
                synapse=tau,
                transform=transform)
            conn_lif = nengo.Connection(lif, temp,
                synapse=tau,
                solver=nengo.solvers.LstsqL2(reg=reg))
            conn_oracle = nengo.Connection(stim, oracle,
                synapse=tau,
                transform=transform)


            probe_stim = nengo.Probe(stim, synapse=None)
            probe_pre = nengo.Probe(pre, synapse=tau_readout)
            probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
            probe_untrained_bio_spikes = nengo.Probe(untrained_bio.neurons, 'spikes')
            probe_trained_bio_spikes = nengo.Probe(trained_bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_oracle = nengo.Probe(oracle, synapse=tau_readout)


        """
        Perform spike-match training on the conn_trained weights
        """
        if train:
            network = spike_train(network, evo_params, plots=False)
            w_trained_new = conn_trained.solver.weights_bio
        else:
            w_trained_new = w_trained

        """
        Run the simulation with the w_trained_new on conn_trained
        and d_ideal on conn_untrained
        """
        with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
            """
            Modulate the synaptic weights of conn_untrained by dendritic location
            to account for dendritic attenuation,
            and undo the multiplications in builder.build_connection()
            """
            times = np.arange(0, 1.0, 0.001)  # to 1s by dt=0.001
            k_norm = np.linalg.norm(np.exp((-times/tau)),1)
            for nrn in sim.data[untrained_bio.neurons]:
                for conn_pre in nrn.synapses.iterkeys():
                    for pre in range(nrn.synapses[conn_pre].shape[0]):
                        for syn in range(nrn.synapses[conn_pre][pre].shape[0]):
                            synapse = nrn.synapses[conn_pre][pre][syn]
                            # if plots: print 'weight before', synapse.weight
                            w_new = synapse.weight * synapse.loc * 2.817 * k_norm / 1e+1
                            nrn.synapses[conn_pre][pre][syn] = ExpSyn(
                                nrn.cell.apical(synapse.loc), w_new, synapse.tau, synapse.loc)
                            # if plots: print 'weight after', nrn.synapses[conn_pre][pre][syn].weight
            neuron.init()
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_readout)
        act_untrained_bio = lpf.filt(sim.data[probe_untrained_bio_spikes], dt=dt)
        act_trained_bio = lpf.filt(sim.data[probe_trained_bio_spikes], dt=dt)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
        d_ideal_new = sim.data[conn_ideal].weights.T

        """
        Plot tuning curves with the trained weights
        """
        if plot:
            n_eval_points = 20
            for i in range(untrained_bio.n_neurons):
                x_dot_e_untrained_bio = np.dot(
                    sim.data[probe_pre],
                    untrained_bio.encoders[i])
                x_dot_e_trained_bio = np.dot(
                    sim.data[probe_pre],
                    trained_bio.encoders[i])
                x_dot_e_lif = np.dot(
                    sim.data[probe_pre],
                    sim.data[lif].encoders[i])
                x_dot_e_vals_untrained_bio = np.linspace(
                    np.min(x_dot_e_untrained_bio),
                    np.max(x_dot_e_untrained_bio), 
                    num=n_eval_points)
                x_dot_e_vals_trained_bio = np.linspace(
                    np.min(x_dot_e_trained_bio),
                    np.max(x_dot_e_trained_bio), 
                    num=n_eval_points)
                x_dot_e_vals_lif = np.linspace(
                    np.min(x_dot_e_lif),
                    np.max(x_dot_e_lif), 
                    num=n_eval_points)
                Hz_mean_untrained_bio = np.zeros((x_dot_e_vals_untrained_bio.shape[0]))
                Hz_stddev_untrained_bio = np.zeros_like(Hz_mean_untrained_bio)
                Hz_mean_trained_bio = np.zeros((x_dot_e_vals_trained_bio.shape[0]))
                Hz_stddev_trained_bio = np.zeros_like(Hz_mean_trained_bio)
                Hz_mean_lif = np.zeros((x_dot_e_vals_lif.shape[0]))
                Hz_stddev_lif = np.zeros_like(Hz_mean_lif)

                for xi in range(x_dot_e_vals_untrained_bio.shape[0] - 1):
                    ts_greater = np.where(x_dot_e_vals_untrained_bio[xi] < sim.data[probe_pre])[0]
                    ts_smaller = np.where(sim.data[probe_pre] < x_dot_e_vals_untrained_bio[xi + 1])[0]
                    ts = np.intersect1d(ts_greater, ts_smaller)
                    if ts.shape[0] > 0: Hz_mean_untrained_bio[xi] = np.average(act_untrained_bio[ts, i])
                    if ts.shape[0] > 1: Hz_stddev_untrained_bio[xi] = np.std(act_untrained_bio[ts, i])
                for xi in range(x_dot_e_vals_trained_bio.shape[0] - 1):
                    ts_greater = np.where(x_dot_e_vals_trained_bio[xi] < sim.data[probe_pre])[0]
                    ts_smaller = np.where(sim.data[probe_pre] < x_dot_e_vals_trained_bio[xi + 1])[0]
                    ts = np.intersect1d(ts_greater, ts_smaller)
                    if ts.shape[0] > 0: Hz_mean_trained_bio[xi] = np.average(act_trained_bio[ts, i])
                    if ts.shape[0] > 1: Hz_stddev_trained_bio[xi] = np.std(act_trained_bio[ts, i])
                for xi in range(x_dot_e_vals_lif.shape[0] - 1):
                    ts_greater = np.where(x_dot_e_vals_lif[xi] < sim.data[probe_pre])[0]
                    ts_smaller = np.where(sim.data[probe_pre] < x_dot_e_vals_lif[xi + 1])[0]
                    ts = np.intersect1d(ts_greater, ts_smaller)
                    if ts.shape[0] > 0: Hz_mean_lif[xi] = np.average(act_lif[ts, i])
                    if ts.shape[0] > 1: Hz_stddev_lif[xi] = np.std(act_lif[ts, i])

                rmse_tuning_curve_untrained = rmse(Hz_mean_untrained_bio[:-2], Hz_mean_lif[:-2])
                rmse_tuning_curve_trained = rmse(Hz_mean_trained_bio[:-2], Hz_mean_lif[:-2])
                lifplot = plt.plot(x_dot_e_vals_lif[:-2], Hz_mean_lif[:-2],
                    label='ideal',
                    ls='dotted')
                plt.fill_between(x_dot_e_vals_lif[:-2],
                    Hz_mean_lif[:-2]+Hz_stddev_lif[:-2],
                    Hz_mean_lif[:-2]-Hz_stddev_lif[:-2],
                    alpha=0.25,
                    facecolor=lifplot[0].get_color())
                untrained_bioplot = plt.plot(x_dot_e_vals_untrained_bio[:-2], Hz_mean_untrained_bio[:-2],
                    # marker='.',
                    ls='dashed',
                    color=lifplot[0].get_color(),
                    label='untrained bioneuron %s, RMSE=%.1f' % (i, rmse_tuning_curve_untrained))
                plt.fill_between(x_dot_e_vals_untrained_bio[:-2],
                    Hz_mean_untrained_bio[:-2]+Hz_stddev_untrained_bio[:-2],
                    Hz_mean_untrained_bio[:-2]-Hz_stddev_untrained_bio[:-2],
                    alpha=0.5,
                    facecolor=lifplot[0].get_color())
                trained_bioplot = plt.plot(x_dot_e_vals_trained_bio[:-2], Hz_mean_trained_bio[:-2],
                    # marker='o',
                    ls='solid',
                    color=lifplot[0].get_color(),
                    label='trained bioneuron %s, RMSE=%.1f' % (i, rmse_tuning_curve_trained))
                plt.fill_between(x_dot_e_vals_trained_bio[:-2],
                    Hz_mean_trained_bio[:-2]+Hz_stddev_trained_bio[:-2],
                    Hz_mean_trained_bio[:-2]-Hz_stddev_trained_bio[:-2],
                    alpha=0.75,
                    facecolor=lifplot[0].get_color())

            plt.ylim(ymin=0)
            plt.xlabel('$x \cdot e$')
            plt.ylabel('firing rate')
            plt.title('Tuning Curves')
            plt.legend()

        return w_trained_new, d_ideal_new


    """
    Run the test
    """
    weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
    weight_filename = 'w_tuning_curves.npz'
    try:
        w_trained_init = np.load(weight_dir+weight_filename)['weights_bio']
        to_train = False
    except IOError:
        w_trained_init = np.zeros((bio_neurons, pre_neurons, n_syn))
        to_train = True
    d_ideal_init = np.zeros((bio_neurons, dim))

    w_trained_new, d_ideal_new = sim(
        w_trained=w_trained_init,
        d_ideal=d_ideal_init,
        evo_params=evo_params,
        signal='sinusoids',
        freq=1,
        seeds=1,
        transform=1,
        t_final=t_final,
        train=to_train,
        plot=False)
    w_pre_bio_extra, d_ideal_extra = sim(
        w_trained=w_trained_new,
        d_ideal=d_ideal_new,
        evo_params=evo_params,
        signal='sinusoids',
        freq=1,
        seeds=1,
        transform=1,
        t_final=10*t_final,
        train=False,
        plot=True)

    # np.savez(weight_dir+weight_filename, weights_bio=w_trained_new)


# def test_pairwise(plt):

# 	l_0 = 0.0
# 	l_test = 0.0
# 	l_test_2 = 0.5
# 	l_test_3 = 1.0
# 	weight = 1e-3
# 	tau = 0.05
# 	t_spike = 250  # ms, run out transients
# 	t_final = 500  # ms

# 	# somatic response to a single spike at l_0
# 	bahl1 = Bahl()
# 	section = bahl1.cell.apical(l_0)
# 	synapse = ExpSyn(section, weight, tau)
# 	bahl1.synapses[l_0] = synapse

# 	# somatic response to one spike at l_0 and one spike at l_test
# 	bahl2 = Bahl()
# 	section_0 = bahl2.cell.apical(l_0)
# 	section_test = bahl2.cell.apical(l_test)
# 	synapse_0 = ExpSyn(section_0, weight, tau)
# 	synapse_test = ExpSyn(section_test, weight, tau)
# 	bahl2.synapses[l_0] = synapse_0
# 	bahl2.synapses[l_test] = synapse_test

# 	# somatic response to one spike at l_0 and one spike at l_test_2
# 	bahl3 = Bahl()
# 	section_0_2 = bahl3.cell.apical(l_0)
# 	section_test_2 = bahl3.cell.apical(l_test_2)
# 	synapse_0_2 = ExpSyn(section_0_2, weight, tau)
# 	synapse_test_2 = ExpSyn(section_test_2, weight, tau)
# 	bahl3.synapses[l_0] = synapse_0_2
# 	bahl3.synapses[l_test_2] = synapse_test_2

# 	# somatic response to one spike at l_0 and one spike at l_test_3
# 	bahl4 = Bahl()
# 	section_0_3 = bahl4.cell.apical(l_0)
# 	section_test_3 = bahl4.cell.apical(l_test_3)
# 	synapse_0_3 = ExpSyn(section_0_3, weight, tau)
# 	synapse_test_3 = ExpSyn(section_test_3, weight, tau)
# 	bahl4.synapses[l_0] = synapse_0_3
# 	bahl4.synapses[l_test_3] = synapse_test_3

# 	# somatic response to one spike at l_test_2 and one spike at l_test_3
# 	bahl5 = Bahl()
# 	section_0_4 = bahl5.cell.apical(l_test_2)
# 	section_test_4 = bahl5.cell.apical(l_test_3)
# 	synapse_0_4 = ExpSyn(section_0_4, weight, tau)
# 	synapse_test_4 = ExpSyn(section_test_4, weight, tau)
# 	bahl5.synapses[l_test_2] = synapse_0_4
# 	bahl5.synapses[l_test_3] = synapse_test_4

# 	# somatic response to one spike at l_test_2 and one spike at l_test_2
# 	bahl6 = Bahl()
# 	section_0_5 = bahl6.cell.apical(l_test_2)
# 	section_test_5 = bahl6.cell.apical(l_test_2)
# 	synapse_0_5 = ExpSyn(section_0_5, weight, tau)
# 	synapse_test_5 = ExpSyn(section_test_5, weight, tau)
# 	bahl6.synapses['l_test_a'] = synapse_0_5
# 	bahl6.synapses['l_test_b'] = synapse_test_5

# 	# somatic response to one spike at l_test_3 and one spike at l_test_2
# 	bahl7 = Bahl()
# 	section_0_6 = bahl7.cell.apical(l_test_3)
# 	section_test_6 = bahl7.cell.apical(l_test_3)
# 	synapse_0_6 = ExpSyn(section_0_6, weight, tau)
# 	synapse_test_6 = ExpSyn(section_test_6, weight, tau)
# 	bahl7.synapses['l_test_a'] = synapse_0_6
# 	bahl7.synapses['l_test_b'] = synapse_test_6

# 	neuron.init()

# 	bahl1.synapses[l_0].spike_in.event(t_spike)
# 	bahl2.synapses[l_0].spike_in.event(t_spike)
# 	bahl2.synapses[l_test].spike_in.event(t_spike)
# 	bahl3.synapses[l_0].spike_in.event(t_spike)
# 	bahl3.synapses[l_test_2].spike_in.event(t_spike)
# 	bahl4.synapses[l_0].spike_in.event(t_spike)
# 	bahl4.synapses[l_test_3].spike_in.event(t_spike)
# 	bahl5.synapses[l_test_2].spike_in.event(t_spike)
# 	bahl5.synapses[l_test_3].spike_in.event(t_spike)
# 	bahl6.synapses['l_test_a'].spike_in.event(t_spike)
# 	bahl6.synapses['l_test_b'].spike_in.event(t_spike)
# 	bahl7.synapses['l_test_a'].spike_in.event(t_spike)
# 	bahl7.synapses['l_test_b'].spike_in.event(t_spike)

# 	neuron.run(t_final)

# 	time = np.array(bahl1.t_record)
# 	v_0 = np.array(bahl1.v_record)
# 	v_test = np.array(bahl2.v_record)
# 	v_test_2 = np.array(bahl3.v_record)
# 	v_test_3 = np.array(bahl4.v_record)
# 	v_test_4 = np.array(bahl5.v_record)
# 	v_test_5 = np.array(bahl6.v_record)
# 	v_test_6 = np.array(bahl7.v_record)
# 	bahl1.cleanup()
# 	bahl2.cleanup()
# 	bahl3.cleanup()
# 	bahl4.cleanup()
# 	bahl5.cleanup()
# 	bahl6.cleanup()
# 	bahl7.cleanup()

# 	# Compute expected v(t) from pairwise spike assuming linear addition and
# 	# the attenuation from the test above, v(d) = 1.0 - 0.355*d
# 	# (shift up the v_0 curve to zero when adding)
# 	v_eq = v_0[int(t_spike / neuron.h.dt) - 1]
# 	v_test_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_0) + (0.970 - 0.355 * l_test)) + v_eq
# 	v_test2_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_0) + (0.970 - 0.355 * l_test_2)) + v_eq
# 	v_test3_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_0) + (0.970 - 0.355 * l_test_3)) + v_eq
# 	v_test4_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_test_2) + (0.970 - 0.355 * l_test_3)) + v_eq
# 	v_test5_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_test_2) + (0.970 - 0.355 * l_test_2)) + v_eq
# 	v_test6_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_test_3) + (0.970 - 0.355 * l_test_3)) + v_eq

# 	plt.subplot(1,1,1)
# 	plt.plot(time, v_0, label='spike @ %s' % l_0) 
# 	a = plt.plot(time, v_test, label='spikes @ %s and %s' % (l_0, l_test)) 
# 	b = plt.plot(time, v_test_2, label='spikes @ %s and %s' % (l_0, l_test_2)) 
# 	c = plt.plot(time, v_test_3, label='spikes @ %s and %s' % (l_0, l_test_3)) 
# 	d = plt.plot(time, v_test_4, label='spikes @ %s and %s' % (l_test_2, l_test_3))
# 	e = plt.plot(time, v_test_5, label='spikes @ %s and %s' % (l_test_2, l_test_2))
# 	f = plt.plot(time, v_test_6, label='spikes @ %s and %s' % (l_test_3, l_test_3))
# 	a_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test_hat[int(t_spike / neuron.h.dt):],
# 		color = a[0].get_color(), linestyle='--',
# 		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_0, l_test))
# 	b_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test2_hat[int(t_spike / neuron.h.dt):],
# 		color = b[0].get_color(), linestyle='--',
# 		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_0, l_test_2))
# 	c_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test3_hat[int(t_spike / neuron.h.dt):],
# 		color = c[0].get_color(), linestyle='--',
# 		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_0, l_test_3))
# 	d_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test4_hat[int(t_spike / neuron.h.dt):],
# 		color = d[0].get_color(), linestyle='--',
# 		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_test_2, l_test_3))
# 	e_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test5_hat[int(t_spike / neuron.h.dt):],
# 		color = e[0].get_color(), linestyle='--',
# 		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_test_2, l_test_2))
# 	f_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test6_hat[int(t_spike / neuron.h.dt):],
# 		color = f[0].get_color(), linestyle='--',
# 		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_test_3, l_test_3))
# 	plt.xlim((t_spike - 50, t_final))
# 	plt.xlabel('time (ms)')
# 	plt.ylabel('$V_{soma}(t)$')
# 	plt.legend()

# 	assert True