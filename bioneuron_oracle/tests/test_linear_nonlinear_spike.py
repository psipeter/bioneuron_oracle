import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, TrainedSolver, spike_train

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

	evo_params = {
		'dt': dt,
		'tau_readout': tau_readout,
		'sim_seed': sim_seed,
		'n_processes': 10,
		'popsize': 10,
		'generations' : 10,
		'w_0': 1e-1,
		'delta_w' :1e-1,
		'evo_seed' :9,
		'evo_t_final' :1.0,
	}

	def sim(
		w_pre_bio,
		d_readout,
		evo_params,
		readout = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=-0.5,
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

			pre_bio_solver = TrainedSolver(weights_bio = w_pre_bio)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				trained_weights=True,
				solver = pre_bio_solver,
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
		Perform spike-match training on the pre-bio weights
		"""
		if train:
			network = spike_train(network, evo_params, plots=False)
			w_pre_bio_new = pre_bio.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio

		"""
		Run the simulation with the new w_pre_bio
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)

		"""
		Calculate readout decoders by either
			- grabbing the ideal decoders from the LIF population, OR
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		if readout == 'LIF':
			d_readout_new = sim.data[conn_lif].weights.T
		elif readout == 'oracle':
			d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle], xhat_lif)
		error_bio = xhat_bio[:,0]-sim.data[probe_oracle][:,0]
		error_lif = xhat_lif[:,0]-sim.data[probe_oracle][:,0]
		dft_bio = np.fft.fft(error_bio)				
		dft_lif = np.fft.fft(error_lif)

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.3f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.3f' % rmse_lif)
			plt.plot(sim.trange(), sim.data[probe_oracle][:,0], label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return w_pre_bio_new, d_readout_new, rmse_bio


	"""
	Run the test
	"""
	weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
	weight_filename = 'w_linear_1d.npz'
	try:
		w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
		to_train = False
	except IOError:
		w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
		to_train = True
	d_readout_init = np.zeros((bio_neurons, dim))

	w_pre_bio_new, d_readout_new, rmse_bio = sim(
		w_pre_bio=w_pre_bio_init,
		d_readout=d_readout_init,
		evo_params=evo_params,
		signal='sinusoids',
		freq=freq_train,
		seeds=seed_train,
		transform=transform,
		t_final=t_final,
		train=to_train,
		plot=False)
	w_pre_bio_extra, d_readout_extra, rmse_bio = sim(
		w_pre_bio=w_pre_bio_new,
		d_readout=d_readout_new,
		evo_params=evo_params,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		train=False,
		plot=True)

	np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)

	assert rmse_bio < cutoff




def test_linear_2d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 20
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 2
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
	freq_train = [1,2]
	freq_test = [1,2]
	seed_train = [1,2]
	seed_test = [1,2]

	dim = 2
	reg = 0.01
	t_final = 1.0
	cutoff = 0.1
	transform = -0.5

	evo_params = {
		'dt': dt,
		'tau_readout': tau_readout,
		'sim_seed': sim_seed,
		'n_processes': 10,
		'popsize': 10,
		'generations' : 50,
		'w_0': 1e-1,
		'delta_w' :1e-2,
		'evo_seed' :9,
		'evo_t_final' :1.0,
		'evo_cutoff' :50.0,
	}

	def sim(
		w_pre_bio,
		d_readout,
		evo_params,
		readout = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=[1,1],
		seeds=[1,1],
		transform=-0.5,
		train=False,
		plot=False):

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq[0] * t))
				stim2 = nengo.Node(lambda t: np.cos(2 * np.pi * freq[1] * t))
			elif signal == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds[0]))
				stim2 = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds[1]))

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

			pre_bio_solver = TrainedSolver(weights_bio = w_pre_bio)

			nengo.Connection(stim, pre[0], synapse=None)
			nengo.Connection(stim2, pre[1], synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				trained_weights=True,
				solver = pre_bio_solver,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle[0],
				synapse=tau,
				transform=transform)
			nengo.Connection(stim2, oracle[1],
				synapse=tau,
				transform=transform)
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_stim2 = nengo.Probe(stim2, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)


		"""
		Perform spike-match training on the pre-bio weights
		"""
		if train:
			network = spike_train(network, evo_params, plots=False)
			w_pre_bio_new = pre_bio.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio

		"""
		Run the simulation with the new w_pre_bio
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)

		"""
		Calculate readout decoders by either
			- grabbing the ideal decoders from the LIF population, OR
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		if readout == 'LIF':
			d_readout_new = sim.data[conn_lif].weights.T
		elif readout == 'oracle':
			d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle][:,0], xhat_lif[:,0])
		rmse_bio2 = rmse(sim.data[probe_oracle][:,1], xhat_bio[:,1])
		rmse_lif2 = rmse(sim.data[probe_oracle][:,1], xhat_lif[:,1])

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), xhat_bio[:,1], label='bio2, rmse=%.5f' % rmse_bio2)
			plt.plot(sim.trange(), xhat_lif[:,1], label='lif2, rmse=%.5f' % rmse_lif2)
			plt.plot(sim.trange(), sim.data[probe_oracle][:,0], label='oracle')
			plt.plot(sim.trange(), sim.data[probe_oracle][:,1], label='oracle2')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return w_pre_bio_new, d_readout_new, rmse_bio


	"""
	Run the test
	"""
	weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
	weight_filename = 'w_transform_in_pre_to_bio.npz'
	try:
		w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
		to_train = False
	except IOError:
		w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
		to_train = True
	d_readout_init = np.zeros((bio_neurons, dim))

	w_pre_bio_new, d_readout_new, rmse_bio = sim(
		w_pre_bio=w_pre_bio_init,
		d_readout=d_readout_init,
		evo_params=evo_params,
		signal='sinusoids',
		freq=freq_train,
		seeds=seed_train,
		transform=transform,
		t_final=t_final,
		train=to_train,
		plot=False)
	w_pre_bio_extra, d_readout_extra, rmse_bio = sim(
		w_pre_bio=w_pre_bio_new,
		d_readout=d_readout_new,
		evo_params=evo_params,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		train=False,
		plot=True)

	np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)

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
	t_final = 1.0
	cutoff = 0.1

	evo_params = {
		'dt': dt,
		'tau_readout': tau_readout,
		'sim_seed': sim_seed,
		'n_processes': 10,
		'popsize': 20,
		'generations' : 20,
		'w_0': 1e-1,
		'delta_w' :1e-1,
		'evo_seed' :9,
		'evo_t_final' :2.0,
	}

	def sim(
		w_pre_bio,
		d_readout_static,
		d_readout_oracle_bio,
		d_readout_oracle_lif,
		evo_params,
		readout = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		order=1,
		train=False,
		plot=False):

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

			pre_bio_solver = TrainedSolver(weights_bio = w_pre_bio)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				seed=conn_seed,
				synapse=tau,
				trained_weights=True,
				function=legendre,
				solver = pre_bio_solver,
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
		Perform spike-match training on the pre-bio weights
		"""
		if train:
			network = spike_train(network, evo_params, plots=False)
			w_pre_bio_new = pre_bio.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio

		"""
		Run the simulation with the new w_pre_bio, then
		Calculate readout decoders by
			- grabbing the ideal decoders from the LIF population
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
		d_readout_static_new = sim.data[conn_lif].weights.T
		d_readout_oracle_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_readout_oracle_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][:,0]
		xhat_bio_static = np.dot(act_bio, d_readout_static)
		xhat_lif_static = np.dot(act_lif, d_readout_static)
		xhat_bio_oracle = np.dot(act_bio, d_readout_oracle_bio)
		xhat_lif_oracle = np.dot(act_lif, d_readout_oracle_lif)	
		rmse_bio_static = rmse(x_target, xhat_bio_static[:,0])
		rmse_lif_static = rmse(x_target, xhat_lif_static[:,0])
		rmse_bio_oracle = rmse(x_target, xhat_bio_oracle[:,0])
		rmse_lif_oracle = rmse(x_target, xhat_lif_oracle[:,0])

		return [w_pre_bio_new,
			d_readout_static_new,
			d_readout_oracle_bio_new,
			d_readout_oracle_lif_new,
			sim.trange(),
			x_target,
			xhat_bio_static,
			xhat_lif_static,
			rmse_bio_static,
			xhat_lif_static]


	"""
	Run the test
	"""
	orders = [1, 2, 3, 4]
	columns = ('time', 'value', 'population', 'order')
	df_list = []
	for order in orders:
		weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
		weight_filename = 'w_nonlinear_order=%s.npz' % order
		try:
			w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
			to_train = False
		except IOError:
			w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
			to_train = True
		d_readout_init = np.zeros((bio_neurons, dim))

		[w_pre_bio_new,
			d_readout_static_new,
			d_readout_oracle_bio_new,
			d_readout_oracle_lif_new,
			times,
			x_target,
			xhat_bio_static,
			xhat_lif_static,
			rmse_bio_static,
			xhat_lif_static] = sim(
				w_pre_bio=w_pre_bio_init,
				d_readout_static=d_readout_init,
				d_readout_oracle_bio=d_readout_init,
				d_readout_oracle_lif=d_readout_init,
				evo_params=evo_params,
				signal='sinusoids',
				freq=freq_train,
				seeds=seed_train,
				order=order,
				t_final=t_final,
				train=to_train,
				plot=False)
		[w_pre_bio_extra,
			d_readout_static_extra,
			d_readout_oracle_bio_extra,
			d_readout_oracle_lif_extra,
			times,
			x_target,
			xhat_bio_static,
			xhat_lif_static,
			rmse_bio_static,
			xhat_lif_static] = sim(
				w_pre_bio=w_pre_bio_new,
				d_readout_static=d_readout_static_new,
				d_readout_oracle_bio=d_readout_oracle_bio_new,
				d_readout_oracle_lif=d_readout_oracle_lif_new,
				evo_params=evo_params,
				signal='sinusoids',
				freq=freq_test,
				seeds=seed_test,
				order=order,
				t_final=t_final,
				train=False,
				plot='signals')

		np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)

		# assert rmse_bio_static < cutoff
		df = pd.DataFrame(columns=columns, index=range(3*len(times)))
		j=0
		times=np.arange(dt, t_final, dt)
		for t, time in enumerate(times):
			df.loc[j] = [time, xhat_bio_static[t][0], 'bio', order]
			df.loc[j+1] = [time, xhat_lif_static[t][0], 'lif', order]
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
