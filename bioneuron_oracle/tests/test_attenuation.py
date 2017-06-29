from functools32 import lru_cache

import numpy as np

from scipy.stats import linregress

import neuron

import nengo
from nengo.utils.numpy import rmse
from nengo.utils.matplotlib import rasterplot

from seaborn import set_palette, color_palette, tsplot

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input, equalpower,
	                          TrainedSolver, OracleSolver, spike_match_train)
from bioneuron_oracle.builder import Bahl, ExpSyn

def test_feedforward(Simulator, plt):

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
	plt.ylabel('Somatic Attenuation ($\Delta V_{soma} \quad / \quad \Delta V_{soma}^{d=0}$)')
	plt.legend()


def test_untrained_feedforward(Simulator, plt):

    """
    Naively assume that using the optimal decoders from pre
    in conjuction with the encoders, gains, and emulated biases
    from bio (the same as from ideal) will construct a weight
    matrix that performs spike matching.
    The synaptic weights will also be multiplied by an attenuation
    factor to account for the known attenuation that occurs for a 
    single spike traveling to the soma, which is approximated by
    v(d) = 1.0 - 0.355*d   ==>   attenuation(d) = 1/0.335 d = 2.817d
    """

    from bioneuron_oracle.builder import ExpSyn
    import neuron

    # Nengo Parameters
    pre_neurons = 100
    bio_neurons = 2
    tau_nengo = 0.05
    tau_neuron = 0.05
    dt_nengo = 0.001
    min_rate = 150
    max_rate = 200
    pre_seed = 3
    bio_seed = 6
    conn_seed = 9
    network_seed = 12
    sim_seed = 15
    t_final = 1.0
    dim = 1
    n_syn = 1
    max_freq = 5
    signal_seed  = 123

    def sim(decoders_bio=None, decoders_bio2=None, plots=False):

        if decoders_bio is None:
            decoders_bio = np.zeros((pre_neurons, dim))
        if decoders_bio2 is None:
            decoders_bio2 = np.zeros((pre_neurons, dim))

        with nengo.Network(seed=network_seed) as network:
            """
            Simulate a feedforward network [stim]-[LIF]-[BIO]
            and compare to [stim]-[LIF]-[LIF].
            """

            stim1 = nengo.Node(
                # lambda t: 0.25 * equalpower(t, dt_nengo, t_final, max_freq, dim,
                #                       mean=0.0, std=1.0, seed=signal_seed))
                lambda t: prime_sinusoids(t, dim, t_final))

            # stim2 = nengo.Node(
                # lambda t: 0.25 * equalpower(t, dt_nengo, t_final, max_freq, dim,
                #                       mean=0.0, std=1.0, seed=2*signal_seed))
                # lambda t: prime_sinusoids(t, 2*dim, t_final)[dim:2*dim])

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF(),
                                 label='pre1')
            # pre2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
            #                      seed=2*pre_seed, neuron_type=nengo.LIF(),
            #                      label='pre2')
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1),
                                 label='bio')
            lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                neuron_type=nengo.Direct())

            oracle_solver = OracleSolver(decoders_bio=decoders_bio)
            # oracle_solver2 = OracleSolver(decoders_bio=decoders_bio2)

            nengo.Connection(stim1, pre, synapse=None)
            # nengo.Connection(stim2, pre2, synapse=None)
            conn_in = nengo.Connection(pre, bio,
                             synapse=tau_neuron,
                             n_syn=n_syn,
                             seed=conn_seed,
                             weights_bias_conn=True,
                             solver=oracle_solver)  # set decoders manually
            # conn_in2 = nengo.Connection(pre2, bio,
            #                  synapse=tau_neuron,
            #                  n_syn=n_syn,
            #                  # weights_bias_conn=True,
            #                  solver=oracle_solver2)  # set decoders manually
            conn_ideal = nengo.Connection(pre, lif, synapse=tau_nengo)
            # conn_ideal2 = nengo.Connection(pre2, lif, synapse=tau_nengo)
            nengo.Connection(stim1, direct, synapse=tau_nengo)
            # nengo.Connection(stim2, direct, synapse=tau_nengo)

            probe_stim = nengo.Probe(stim1, synapse=None)
            probe_pre = nengo.Probe(pre, synapse=tau_nengo)
            probe_lif = nengo.Probe(lif, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_pre_spikes = nengo.Probe(pre.neurons, 'spikes')
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
            probe_lif_voltage = nengo.Probe(lif.neurons, 'voltage')

        with Simulator(network, dt=dt_nengo, seed=sim_seed) as sim:
            for nrn in sim.data[bio.neurons]:
                for conn_pre in nrn.synapses.iterkeys():
                    for pre in range(nrn.synapses[conn_pre].shape[0]):
                        for syn in range(nrn.synapses[conn_pre][pre].shape[0]):
                            synapse = nrn.synapses[conn_pre][pre][syn]
                            # if plots: print 'weight before', synapse.weight
                            w_new = synapse.weight * synapse.loc * 2.817
                            nrn.synapses[conn_pre][pre][syn] = ExpSyn(
                                nrn.cell.apical(synapse.loc), w_new, synapse.tau, synapse.loc)
                            # if plots: print 'weight after', nrn.synapses[conn_pre][pre][syn].weight
            neuron.init()
            sim.run(t_final)

        decoders_ideal = sim.data[conn_ideal].weights.T
        # decoders_ideal2 = sim.data[conn_ideal2].weights.T
        decoders_ideal2 = decoders_bio2

        if plots:
            plt.subplot(1,1,1)
            # copy tuning curve plotting from above
            n_eval_points = 20
            cutoff = 10.0

            lpf = nengo.Lowpass(tau_nengo)
            act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
            act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
            rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
                                 for i in range(bio.n_neurons)])

            for i in range(bio.n_neurons):
                x_dot_e_bio = np.dot(sim.data[probe_pre], bio.encoders[i])
                x_dot_e_lif = np.dot(sim.data[probe_pre], sim.data[lif].encoders[i])
                x_dot_e_vals_bio = np.linspace(np.min(x_dot_e_bio),
                                               np.max(x_dot_e_bio), 
                                               num=n_eval_points)
                x_dot_e_vals_lif = np.linspace(np.min(x_dot_e_lif),
                                               np.max(x_dot_e_lif), 
                                               num=n_eval_points)
                Hz_mean_bio = np.zeros((x_dot_e_vals_bio.shape[0]))
                Hz_stddev_bio = np.zeros_like(Hz_mean_bio)
                Hz_mean_lif = np.zeros((x_dot_e_vals_lif.shape[0]))
                Hz_stddev_lif = np.zeros_like(Hz_mean_lif)
                for xi in range(x_dot_e_vals_bio.shape[0] - 1):
                    ts_greater = np.where(x_dot_e_vals_bio[xi] < sim.data[probe_pre])[0]
                    ts_smaller = np.where(sim.data[probe_pre] < x_dot_e_vals_bio[xi + 1])[0]
                    ts = np.intersect1d(ts_greater, ts_smaller)
                    if ts.shape[0] > 0: Hz_mean_bio[xi] = np.average(act_bio[ts, i])
                    if ts.shape[0] > 1: Hz_stddev_bio[xi] = np.std(act_bio[ts, i])
                for xi in range(x_dot_e_vals_lif.shape[0] - 1):
                    ts_greater = np.where(x_dot_e_vals_lif[xi] < sim.data[probe_pre])[0]
                    ts_smaller = np.where(sim.data[probe_pre] < x_dot_e_vals_lif[xi + 1])[0]
                    ts = np.intersect1d(ts_greater, ts_smaller)
                    if ts.shape[0] > 0: Hz_mean_lif[xi] = np.average(act_lif[ts, i])
                    if ts.shape[0] > 1: Hz_stddev_lif[xi] = np.std(act_lif[ts, i])

                rmse_tuning_curve = rmse(Hz_mean_bio[:-2], Hz_mean_lif[:-2])
                bioplot = plt.errorbar(x_dot_e_vals_bio[:-2], Hz_mean_bio[:-2],
                             yerr=Hz_stddev_bio[:-2], fmt='-o',
                             label='BIO %s, RMSE=%.5f' % (i, rmse_tuning_curve))
                lifplot = plt.errorbar(x_dot_e_vals_lif[:-2], Hz_mean_lif[:-2],
                             yerr=Hz_stddev_lif[:-2], fmt='--', label='LIF %s' % i,
                             color=bioplot[0].get_color())
                lifplot[-1][0].set_linestyle('--')
            plt.xlabel('$x \cdot e$')
            plt.ylabel('firing rate')
            plt.title('Tuning Curves')
            plt.legend()


            assert rmse_tuning_curve < cutoff

        return decoders_ideal, decoders_ideal2  

    decoders_ideal, decoders_ideal2 = sim(decoders_bio=None, decoders_bio2=None, plots=False)
    decoders_ideal, decoders_ideal2 = sim(decoders_bio=decoders_ideal, decoders_bio2=decoders_ideal2, plots=True)


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