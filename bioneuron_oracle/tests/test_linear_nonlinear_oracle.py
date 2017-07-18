import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron

def test_linear_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
	bio_radius = 1
	n_syn = 1

	pre_seed = 1
	bio_seed = 2
	conn_seed = 3
	network_seed = 4
	sim_seed = 5
	post_seed = 6
	inter_seed = 7

	max_freq = 5
	rms = 1.0
	freq_train = 1
	freq_test = 2
	seed_train = 1
	seed_test = 2

	dim = 1
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1
	transform = -0.5

	def sim(
		d_readout_bio,
		d_readout_lif,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1,
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

			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre')
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio')
			lif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=bio.dimensions,
				seed=bio.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif')
			oracle = nengo.Node(size_in=dim)
			temp = nengo.Node(size_in=dim)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=transform)
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)


		"""
		Simulate the network, collect bioneuron activities and target values,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
		# bio readout is always "oracle" for the oracle method training
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][:,0]
		xhat_bio = np.dot(act_bio, d_readout_bio)[:,0]
		xhat_lif = np.dot(act_lif, d_readout_lif)[:,0]
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)

		if plot:
			plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target, label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_readout_bio_new, d_readout_lif_new, rmse_bio


	"""
	Run the test
	"""
	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		plot=False)
	d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		signal='sinusoids',
		freq=freq_train,
		seeds=seed_train,
		transform=transform,
		t_final=t_final,
		plot=True)

	assert rmse_bio < cutoff



def test_nonlinear_1d(Simulator, plt):

	import scipy.special as sp
	import seaborn as sns
	import pandas as pd

	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
	bio_radius = 1
	n_syn = 1

	pre_seed = 1
	bio_seed = 2
	conn_seed = 3
	network_seed = 4
	sim_seed = 5
	post_seed = 6
	inter_seed = 7

	max_freq = 5
	rms = 1.0
	freq_train = 1
	freq_test = 1
	seed_train = 1
	seed_test = 1

	dim = 1
	reg = 0.1
	transform = 1
	t_final = 1.0
	cutoff = 0.1

	def sim(
		d_readout_bio,
		d_readout_lif,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		order=1,
		transform=1):

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			def legendre(x):
				return sp.legendre(order)(x)

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq * t))
			elif signal == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds))

			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre')
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio')
			lif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=bio.dimensions,
				seed=bio.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif')
			oracle = nengo.Node(size_in=dim)
			temp = nengo.Node(size_in=dim)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				function=legendre,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				function=legendre)
			nengo.Connection(stim, oracle,
				synapse=tau,
				function=legendre)
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)

		"""
		Simulate the network, collect bioneuron activities and target values,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
		# bio readout is always "oracle" for the oracle method training
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][:,0]
		xhat_bio = np.dot(act_bio, d_readout_bio)
		xhat_lif = np.dot(act_lif, d_readout_lif)
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)

		return [d_readout_bio_new,
			d_readout_lif_new,
			sim.trange(),
			x_target,
			xhat_bio,
			xhat_lif,
			rmse_bio,
			rmse_lif]



	"""
	Run the test
	"""
	orders = [1, 2, 3, 4]
	columns = ('time', 'value', 'population', 'order')
	df_list = []
	for order in orders:
		d_readout_bio_init = np.zeros((bio_neurons, dim))
		d_readout_lif_init = np.zeros((bio_neurons, dim))

		[d_readout_bio_new,
			d_readout_lif_new,
			times,
			x_target,
			xhat_bio,
			xhat_lif,
			rmse_bio,
			rmse_lif] = sim(
			d_readout_bio=d_readout_bio_init,
			d_readout_lif=d_readout_lif_init,
			signal='sinusoids',
			freq=freq_test,
			seeds=seed_test,
			transform=transform,
			order=order,
			t_final=t_final)
		[d_readout_bio_extra,
			d_readout_lif_extra,
			times,
			x_target,
			xhat_bio,
			xhat_lif,
			rmse_bio,
			rmse_lif] = sim(
			d_readout_bio=d_readout_bio_new,
			d_readout_lif=d_readout_lif_new,
			signal='sinusoids',
			freq=freq_train,
			seeds=seed_train,
			transform=transform,
			order=order,
			t_final=t_final)

		# assert rmse_bio_static < cutoff
		df = pd.DataFrame(columns=columns, index=range(3*len(times)))
		j=0
		times=np.arange(dt, t_final, dt)
		for t, time in enumerate(times):
			print [time, xhat_bio[t][0], 'bio', order]
			df.loc[j] = [time, xhat_bio[t][0], 'bio', order]
			df.loc[j+1] = [time, xhat_lif[t][0], 'lif', order]
			df.loc[j+2] = [time, x_target[t], 'oracle', order]
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
	# g = sns.factorplot(x='time', y='value', hue='population', col='order', data=df_final,
	# 	col_wrap=2)

	assert rmse_bio < cutoff
