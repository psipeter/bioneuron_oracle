from functools32 import lru_cache

import numpy as np

from scipy.stats import linregress

import neuron

import nengo
from nengo.utils.numpy import rmse

from bioneuron_oracle.builder import Bahl, ExpSyn

def test_feedforward(plt):

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
		synapse = ExpSyn(section, weight, tau)
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



def test_pairwise(plt):

	l_0 = 0.0
	l_test = 0.0
	l_test_2 = 0.5
	l_test_3 = 1.0
	weight = 1e-3
	tau = 0.05
	t_spike = 250  # ms, run out transients
	t_final = 500  # ms

	# somatic response to a single spike at l_0
	bahl1 = Bahl()
	section = bahl1.cell.apical(l_0)
	synapse = ExpSyn(section, weight, tau)
	bahl1.synapses[l_0] = synapse

	# somatic response to one spike at l_0 and one spike at l_test
	bahl2 = Bahl()
	section_0 = bahl2.cell.apical(l_0)
	section_test = bahl2.cell.apical(l_test)
	synapse_0 = ExpSyn(section_0, weight, tau)
	synapse_test = ExpSyn(section_test, weight, tau)
	bahl2.synapses[l_0] = synapse_0
	bahl2.synapses[l_test] = synapse_test

	# somatic response to one spike at l_0 and one spike at l_test_2
	bahl3 = Bahl()
	section_0_2 = bahl3.cell.apical(l_0)
	section_test_2 = bahl3.cell.apical(l_test_2)
	synapse_0_2 = ExpSyn(section_0_2, weight, tau)
	synapse_test_2 = ExpSyn(section_test_2, weight, tau)
	bahl3.synapses[l_0] = synapse_0_2
	bahl3.synapses[l_test_2] = synapse_test_2

	# somatic response to one spike at l_0 and one spike at l_test_3
	bahl4 = Bahl()
	section_0_3 = bahl4.cell.apical(l_0)
	section_test_3 = bahl4.cell.apical(l_test_3)
	synapse_0_3 = ExpSyn(section_0_3, weight, tau)
	synapse_test_3 = ExpSyn(section_test_3, weight, tau)
	bahl4.synapses[l_0] = synapse_0_3
	bahl4.synapses[l_test_3] = synapse_test_3

	# somatic response to one spike at l_test_2 and one spike at l_test_3
	bahl5 = Bahl()
	section_0_4 = bahl5.cell.apical(l_test_2)
	section_test_4 = bahl5.cell.apical(l_test_3)
	synapse_0_4 = ExpSyn(section_0_4, weight, tau)
	synapse_test_4 = ExpSyn(section_test_4, weight, tau)
	bahl5.synapses[l_test_2] = synapse_0_4
	bahl5.synapses[l_test_3] = synapse_test_4

	# somatic response to one spike at l_test_2 and one spike at l_test_2
	bahl6 = Bahl()
	section_0_5 = bahl6.cell.apical(l_test_2)
	section_test_5 = bahl6.cell.apical(l_test_2)
	synapse_0_5 = ExpSyn(section_0_5, weight, tau)
	synapse_test_5 = ExpSyn(section_test_5, weight, tau)
	bahl6.synapses['l_test_a'] = synapse_0_5
	bahl6.synapses['l_test_b'] = synapse_test_5

	# somatic response to one spike at l_test_3 and one spike at l_test_2
	bahl7 = Bahl()
	section_0_6 = bahl7.cell.apical(l_test_3)
	section_test_6 = bahl7.cell.apical(l_test_3)
	synapse_0_6 = ExpSyn(section_0_6, weight, tau)
	synapse_test_6 = ExpSyn(section_test_6, weight, tau)
	bahl7.synapses['l_test_a'] = synapse_0_6
	bahl7.synapses['l_test_b'] = synapse_test_6

	neuron.init()

	bahl1.synapses[l_0].spike_in.event(t_spike)
	bahl2.synapses[l_0].spike_in.event(t_spike)
	bahl2.synapses[l_test].spike_in.event(t_spike)
	bahl3.synapses[l_0].spike_in.event(t_spike)
	bahl3.synapses[l_test_2].spike_in.event(t_spike)
	bahl4.synapses[l_0].spike_in.event(t_spike)
	bahl4.synapses[l_test_3].spike_in.event(t_spike)
	bahl5.synapses[l_test_2].spike_in.event(t_spike)
	bahl5.synapses[l_test_3].spike_in.event(t_spike)
	bahl6.synapses['l_test_a'].spike_in.event(t_spike)
	bahl6.synapses['l_test_b'].spike_in.event(t_spike)
	bahl7.synapses['l_test_a'].spike_in.event(t_spike)
	bahl7.synapses['l_test_b'].spike_in.event(t_spike)

	neuron.run(t_final)

	time = np.array(bahl1.t_record)
	v_0 = np.array(bahl1.v_record)
	v_test = np.array(bahl2.v_record)
	v_test_2 = np.array(bahl3.v_record)
	v_test_3 = np.array(bahl4.v_record)
	v_test_4 = np.array(bahl5.v_record)
	v_test_5 = np.array(bahl6.v_record)
	v_test_6 = np.array(bahl7.v_record)
	bahl1.cleanup()
	bahl2.cleanup()
	bahl3.cleanup()
	bahl4.cleanup()
	bahl5.cleanup()
	bahl6.cleanup()
	bahl7.cleanup()

	# Compute expected v(t) from pairwise spike assuming linear addition and
	# the attenuation from the test above, v(d) = 1.0 - 0.355*d
	# (shift up the v_0 curve to zero when adding)
	v_eq = v_0[int(t_spike / neuron.h.dt) - 1]
	v_test_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_0) + (0.970 - 0.355 * l_test)) + v_eq
	v_test2_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_0) + (0.970 - 0.355 * l_test_2)) + v_eq
	v_test3_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_0) + (0.970 - 0.355 * l_test_3)) + v_eq
	v_test4_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_test_2) + (0.970 - 0.355 * l_test_3)) + v_eq
	v_test5_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_test_2) + (0.970 - 0.355 * l_test_2)) + v_eq
	v_test6_hat = (v_0 - v_eq) * ((0.970 - 0.355 * l_test_3) + (0.970 - 0.355 * l_test_3)) + v_eq

	plt.subplot(1,1,1)
	plt.plot(time, v_0, label='spike @ %s' % l_0) 
	a = plt.plot(time, v_test, label='spikes @ %s and %s' % (l_0, l_test)) 
	b = plt.plot(time, v_test_2, label='spikes @ %s and %s' % (l_0, l_test_2)) 
	c = plt.plot(time, v_test_3, label='spikes @ %s and %s' % (l_0, l_test_3)) 
	d = plt.plot(time, v_test_4, label='spikes @ %s and %s' % (l_test_2, l_test_3))
	e = plt.plot(time, v_test_5, label='spikes @ %s and %s' % (l_test_2, l_test_2))
	f = plt.plot(time, v_test_6, label='spikes @ %s and %s' % (l_test_3, l_test_3))
	a_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test_hat[int(t_spike / neuron.h.dt):],
		color = a[0].get_color(), linestyle='--',
		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_0, l_test))
	b_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test2_hat[int(t_spike / neuron.h.dt):],
		color = b[0].get_color(), linestyle='--',
		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_0, l_test_2))
	c_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test3_hat[int(t_spike / neuron.h.dt):],
		color = c[0].get_color(), linestyle='--',
		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_0, l_test_3))
	d_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test4_hat[int(t_spike / neuron.h.dt):],
		color = d[0].get_color(), linestyle='--',
		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_test_2, l_test_3))
	e_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test5_hat[int(t_spike / neuron.h.dt):],
		color = e[0].get_color(), linestyle='--',
		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_test_2, l_test_2))
	f_hat = plt.plot(time[int(t_spike / neuron.h.dt):], v_test6_hat[int(t_spike / neuron.h.dt):],
		color = f[0].get_color(), linestyle='--',
		label='$\hat{v}(t)$ w/ spikes @ %s and %s' % (l_test_3, l_test_3))
	plt.xlim((t_spike - 50, t_final))
	plt.xlabel('time (ms)')
	plt.ylabel('$V_{soma}(t)$')
	plt.legend()

	assert True