import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, TrainedSolver, spike_train

def test_representation(Simulator, plt):
	pre_neurons = 100
	bio_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 60
	max_rate = 80
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
	rms = 0.25
	n_steps = 2

	t_transient = 1.0

	signal_train = 'sinusoids'
	freq_train = 1.0
	seed_train = 1
	transform_train = 1.0
	t_train = 1.0

	signal_test = 'custom'
	freq_test = 1.0
	seed_test = 1
	transform_test = 1.0
	t_test = 1.0

	dim = 1
	reg = 0.1
	cutoff = 0.1

	evo_params = {
		'dt': dt,
		'tau_readout': tau_readout,
		'sim_seed': sim_seed,
		'n_processes': 10,
		'popsize': 10,
		'generations' : 20,
		'w_0': 1e-2,
		'delta_w' :1e-2,
		'evo_seed' :9,
		'evo_t_final' :1.0,
	}

	def sim(
		w_pre_bio,
		d_readout_static,
		d_readout_oracle_bio,
		d_readout_oracle_lif,
		evo_params,
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1.0,
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
					np.linspace(-1, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
			elif signal == 'constant':
				stim = nengo.Node(lambda t: freq)
			elif signal == 'custom':
				stim = nengo.Node(lambda t:
					0.0 * (t < t_transient)
					# + -1.0 * (1 < t < 2)
					# + 1.0 * (2 < t < 3))
					+ np.cos(2 * np.pi * freq * (t-t_transient)) * (t > t_transient))

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
		Run the simulation with the new w_pre_bio, then
		Calculate readout decoders by
			- grabbing the ideal decoders from the LIF population
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
		d_readout_static_new = sim.data[conn_lif].weights.T
		d_readout_oracle_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_readout_oracle_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]
		# print 'test bio', np.sum(pre_bio.solver.weights_bio)


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		# xhat_bio_static = np.dot(act_bio, d_readout_static)
		# xhat_lif_static = np.dot(act_lif, d_readout_static)
		# xhat_bio_oracle = np.dot(act_bio, d_readout_oracle_bio)
		# xhat_lif_oracle = np.dot(act_lif, d_readout_oracle_lif)	
		# rmse_bio_static = rmse(sim.data[probe_oracle][:,0], xhat_bio_static[:,0])
		# rmse_lif_static = rmse(sim.data[probe_oracle][:,0], xhat_lif_static[:,0])
		# rmse_bio_oracle = rmse(sim.data[probe_oracle][:,0], xhat_bio_oracle[:,0])
		# rmse_lif_oracle = rmse(sim.data[probe_oracle][:,0], xhat_lif_oracle[:,0])
		# if plot == 'signals':
		# 	plt.plot(sim.trange(), xhat_bio_static[:,0], label='bio, static, rmse=%.5f' % rmse_bio_static)
		# 	plt.plot(sim.trange(), xhat_lif_static[:,0], label='lif, static, rmse=%.5f' % rmse_lif_static)
		# 	plt.plot(sim.trange(), xhat_bio_oracle[:,0], label='bio, oracle, rmse=%.5f' % rmse_bio_oracle)
		# 	plt.plot(sim.trange(), xhat_lif_oracle[:,0], label='lif, oracle, rmse=%.5f' % rmse_lif_oracle)
		# 	plt.plot(sim.trange(), sim.data[probe_oracle][:,0], label='oracle')
		# 	plt.xlabel('time (s)')
		# 	plt.ylabel('$\hat{x}(t)$')
		# 	plt.legend()
		xhat_bio_static = np.dot(act_bio, d_readout_static)[int(t_transient/dt):,0]
		xhat_lif_static = np.dot(act_lif, d_readout_static)[int(t_transient/dt):,0]
		xhat_bio_oracle = np.dot(act_bio, d_readout_oracle_bio)[int(t_transient/dt):,0]
		xhat_lif_oracle = np.dot(act_lif, d_readout_oracle_lif)[int(t_transient/dt):,0]
		rmse_bio_static = rmse(sim.data[probe_oracle][int(t_transient/dt):,0], xhat_bio_static)
		rmse_lif_static = rmse(sim.data[probe_oracle][int(t_transient/dt):,0], xhat_lif_static)
		rmse_bio_oracle = rmse(sim.data[probe_oracle][int(t_transient/dt):,0], xhat_bio_oracle)
		rmse_lif_oracle = rmse(sim.data[probe_oracle][int(t_transient/dt):,0], xhat_lif_oracle)
		if plot == 'signals':
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio_static, label='bio, static, rmse=%.5f' % rmse_bio_static)
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif_static, label='lif, static, rmse=%.5f' % rmse_lif_static)
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio_oracle, label='bio, oracle, rmse=%.5f' % rmse_bio_oracle)
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif_oracle, label='lif, oracle, rmse=%.5f' % rmse_lif_oracle)
			plt.plot(sim.trange()[int(t_transient/dt):], sim.data[probe_oracle][int(t_transient/dt):,0], label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()
		elif plot == 'rates':
			# plt.plot(sim.trange(), 100*np.arange(pre_neurons)[None,:]+act_pre, label='pre')
			plt.plot(sim.trange(), act_bio[:,:10], label='bio')
			# plt.plot(sim.trange(), act_lif[:,:10], label='lif', linestyle='--')
			# plt.plot(sim.trange(), 100*np.arange(bio_neurons)[None,:]+act_bio[:,:10], label='bio')
			# plt.plot(sim.trange(), 100*np.arange(bio_neurons)[None,:]+act_lif[:,:10], label='lif', linestyle='--')
			plt.xlabel('time (s)')
			plt.ylabel('firing rates (Hz)')
			plt.legend()		

		return w_pre_bio_new, d_readout_static_new, d_readout_oracle_bio_new, d_readout_oracle_lif_new


	"""
	Run the test
	"""
	weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
	weight_filename = 'w_representation.npz'
	try:
		w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
		to_train = False
	except IOError:
		w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
		to_train = True
	d_readout_init = np.zeros((bio_neurons, dim))

	w_pre_bio_new, d_readout_static_new, d_readout_oracle_bio_new, d_readout_oracle_lif_new = sim(
		w_pre_bio=w_pre_bio_init,
		d_readout_static=d_readout_init,
		d_readout_oracle_bio=d_readout_init,
		d_readout_oracle_lif=d_readout_init,
		evo_params=evo_params,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		train=to_train,
		plot=False)
	w_pre_bio_extra, d_readout_static_extra, d_readout_oracle_bio_extra, d_readout_oracle_lif_extra = sim(
		w_pre_bio=w_pre_bio_new,
		d_readout_static=d_readout_static_new,
		d_readout_oracle_bio=d_readout_oracle_bio_new,
		d_readout_oracle_lif=d_readout_oracle_lif_new,
		evo_params=evo_params,
		signal=signal_test,
		freq=freq_test,
		seeds=seed_test,
		transform=transform_test,
		t_final=t_test,
		train=False,
		plot='signals')

	np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)