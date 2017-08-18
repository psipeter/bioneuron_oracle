import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, get_stim_deriv, train_filters_decoders, build_filter
import scipy.special as sp
import seaborn as sns
import pandas as pd

def test_nonlinear_1d(Simulator, plt):

	def simulate_legendre(order):
		# Nengo Parameters
		pre_neurons = 100
		bio_neurons = 10
		tau = 0.1
		tau_readout = 0.1
		tau_decoders = 0.1
		tau_JL = 0.1
		min_rate = 150
		max_rate = 200
		radius = 1
		bio_radius = np.sqrt(2)
		n_syn = 1
		dim = 1
		reg = 0.01
		cutoff = 0.1

		pre_seed = 1
		bio_seed = 2
		conn_seed = 3
		network_seed = 4
		sim_seed = 5
		post_seed = 6
		inter_seed = 7

		freq = 10
		max_freq = 5
		rms = 0.5
		t_transient = 0.5
		t_train = 1.0
		t_test = 1.0
		dt = 0.001
		signal_type = 'white_noise'
		seed_train = 1
		seed_test = 2

		n_processes = 10
		evo_popsize = 200
		evo_gen = 20
		evo_seed = 1
		zeros = [1e2]
		poles = [-1e2, -1e2]
		delta_zeros = [1e1]
		delta_poles = [1e1, 1e1]
		filter_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/deriv_filters/'

		filter_filename = 'order_%s_test1.npz' % order

		signal_train, deriv_train = get_stim_deriv(
			signal_type, network_seed, sim_seed, freq, seed_train, t_transient, t_train, max_freq, rms, tau, dt)
		signal_test, deriv_test = get_stim_deriv(
			signal_type, network_seed, sim_seed, freq, seed_test, t_transient, t_test, max_freq, rms, tau, dt)
		# derivatives of stim passed through legendre then normalized.
		# This probably won't work, need to compute deriv of each order legendre
		def legendre(x):
			return sp.legendre(order)(x)
		deriv_train = legendre(deriv_train)
		deriv_train /= max(abs(deriv_train))
		deriv_test = legendre(deriv_test)
		deriv_test /= max(abs(deriv_test))


		def make_network():
			"""
			Define the network
			"""
			with nengo.Network(seed=network_seed) as network:
				def legendre(x):
					return sp.legendre(order)(x)

				stim = nengo.Node(lambda t: signal_test[int(t/dt)])
				pre = nengo.Ensemble(
					n_neurons=pre_neurons,
					dimensions=dim,
					seed=pre_seed,
					neuron_type=nengo.LIF(),
					radius=radius,
					label='pre')
				pre_deriv = nengo.Ensemble(
					n_neurons=pre_neurons,
					dimensions=dim,
					seed=pre_seed,
					neuron_type=nengo.LIF(),
					radius=radius,
					label='pre_deriv')
				bio = nengo.Ensemble(
					n_neurons=bio_neurons,
					dimensions=dim+1,
					seed=bio_seed,
					neuron_type=BahlNeuron(),
					radius=bio_radius,
					max_rates=nengo.dists.Uniform(min_rate, max_rate),
					label='bio')
				lif = nengo.Ensemble(
					n_neurons=bio.n_neurons,
					dimensions=dim,
					seed=bio.seed,
					max_rates=nengo.dists.Uniform(min_rate, max_rate),
					radius=bio.radius,
					neuron_type=nengo.LIF(),
					label='lif')
				oracle = nengo.Node(size_in=dim)
				temp = nengo.Node(size_in=dim)

				nengo.Connection(stim, pre, synapse=None)
				nengo.Connection(stim, oracle, synapse=tau, function=legendre)
				nengo.Connection(pre, bio[0],
					weights_bias_conn=True,
					seed=conn_seed,
					synapse=tau,
					n_syn=n_syn,
					function=legendre)
				nengo.Connection(pre_deriv, bio[1],
					weights_bias_conn=False,
					seed=2*conn_seed,
					synapse=tau,
					n_syn=n_syn)
				nengo.Connection(pre, lif, synapse=tau, function=legendre)
				network.conn_lif = nengo.Connection(lif, temp, synapse=None)

				probe_stim = nengo.Probe(stim, synapse=None)
				probe_pre = nengo.Probe(pre, synapse=tau_readout)
				probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
				probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
				probe_lif_activity = nengo.Probe(lif.neurons, 'spikes', synapse=tau_readout)
				probe_bio_activity = nengo.Probe(bio.neurons, 'spikes', synapse=tau_readout)
				probe_oracle = nengo.Probe(oracle, synapse=tau_readout)

			network.bio = bio
			network.stim = stim
			network.probe_pre = probe_pre
			network.target_probe = probe_oracle
			network.bio_probe = probe_bio_activity
			network.lif_probe = probe_lif_activity

			return network

		network = make_network()

		"""
		Construct dictionary for training the bio_filters
		"""
		filters_to_train = {
			'bio': {
				'filter_dir': filter_dir,
				'filter_filename': filter_filename,
				'stim': network.stim,
				'bio_probe': network.bio_probe,
				'target_probe': network.target_probe,
			}
		}

		""" 
		Use 1 1+lambda evolutionary algorithm to optimize the readout filters and readout decoders
		for the bioneurons and alifs, then add probes with those filters into the network
		"""
		for filt in filters_to_train.iterkeys():
			filter_dir = filters_to_train[filt]['filter_dir']
			filter_filename = filters_to_train[filt]['filter_filename']
			stim = filters_to_train[filt]['stim']
			bio_probe = filters_to_train[filt]['bio_probe']
			target_probe = filters_to_train[filt]['target_probe']
			try:
				filter_info = np.load(filter_dir+filter_filename)
				zeros = filter_info['zeros']
				poles = filter_info['poles']
				d_bio = filter_info['d_bio']
			except IOError:
				zeros, poles, d_bio = train_filters_decoders(
					network,
					Simulator,
					sim_seed,
					signal_train,
					t_transient,
					t_train,
					dt,
					reg,
					n_processes,
					evo_popsize,
					evo_gen,
					evo_seed,
					zeros,
					poles,
					delta_zeros,
					delta_poles,
					bio_probe,
					target_probe,
					)
				np.savez(filter_dir+filter_filename,
					zeros=zeros,
					poles=poles,
					d_bio=d_bio)
			f_bio = build_filter(zeros, poles)
			with network:
				stim.output = lambda t: signal_test[int(t/dt)]
				bio_probe.synapse = f_bio


		"""
		Simulate the network, collect the filtered bioneuron activities and target values,
		and decode the activities to estimate the state 
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_test)
		act_bio = sim.data[network.bio_probe]
		act_lif = sim.data[network.lif_probe]
		x_target = sim.data[network.target_probe][int(t_transient/dt):]
		xhat_bio = np.dot(act_bio, d_bio)[int(t_transient/dt):]
		xhat_lif = np.dot(act_lif, sim.data[network.conn_lif].weights.T)[int(t_transient/dt):]
		# rmse_bio = rmse(x_target, xhat_bio)
		# rmse_lif = rmse(x_target, xhat_lif)

		return xhat_bio, xhat_lif, x_target


	"""
	Run the Experiment
	"""

	orders = [1, 2, 3, 4]
	columns = ('time', 'value', 'population', 'order')
	df_list = []
	t_transient = 0.1
	t_train = 0.2
	t_test = 0.2
	dt = 0.001
	times=np.arange(dt, t_test, dt)
	for order in orders:
		xhat_bio, xhat_lif, x_target = simulate_legendre(order)
		df = pd.DataFrame(columns=columns, index=range(3*len(times)))
		j=0
		for t, time in enumerate(times):
			df.loc[j] = [time, xhat_bio[t][0], 'bio', order]
			df.loc[j+1] = [time, xhat_lif[t][0], 'lif', order]
			df.loc[j+2] = [time, x_target[t][0], 'oracle', order]
			j+=3
		df_list.append(df)
	df_final = pd.concat(df_list, ignore_index=True)

	figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True, sharey=True)
	axes = [ax1, ax2, ax3, ax4]
	for o, order in enumerate(orders):
		time = np.array(df_final.query(
			"order==%s & population=='bio'" % order).reset_index()['time'])
		bio = np.array(df_final.query(
			"order==%s & population=='bio'" % order).reset_index()['value'])
		lif = np.array(df_final.query(
			"order==%s & population=='lif'" % order).reset_index()['value'])
		oracle = np.array(df_final.query(
			"order==%s & population=='oracle'" % order).reset_index()['value'])
		rmse_bio = rmse(bio, oracle)
		rmse_lif = rmse(lif, oracle)
		axes[o].plot(time, bio, label='bio, rmse=%0.3f' % rmse_bio)
		axes[o].plot(time, lif, label='lif, rmse=%0.3f' % rmse_lif)
		axes[o].plot(time, oracle, label='oracle')
		axes[o].set(title='order = %s' %order)
		axes[o].legend(loc='lower left')
	ax1.set(ylabel='$\hat{x}(t)$')
	ax3.set(xlabel='time (s)', ylabel='$\hat{x}(t)$')
	ax4.set(xlabel='time (s)')