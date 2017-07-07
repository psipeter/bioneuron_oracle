import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, TrainedSolver, spike_train

def test_transform_in_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	bio2_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
	bio_radius = 1
	bio2_radius = 1
	n_syn = 1

	pre_seed = 1
	bio_seed = 2
	conn_seed = 3
	network_seed = 4
	sim_seed = 5
	post_seed = 6
	inter_seed = 7
	bio2_seed = 8
	conn2_seed = 9

	max_freq = 5
	rms = 1.0

	dim = 1
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1

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
		w_bio_bio2,
		d_readout,
		d_readout2,
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
			bio2 = nengo.Ensemble(
				n_neurons=bio2_neurons,
				dimensions=dim,
				seed=bio2_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio2_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio2')
			lif2 = nengo.Ensemble(
				n_neurons=bio2.n_neurons,
				dimensions=bio2.dimensions,
				seed=bio2.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif2')
			oracle = nengo.Node(size_in=dim)
			oracle2 = nengo.Node(size_in=dim)
			temp = nengo.Node(size_in=dim)
			temp2 = nengo.Node(size_in=dim)

			pre_bio_solver = TrainedSolver(weights_bio = w_pre_bio)
			bio_bio2_solver = TrainedSolver(weights_bio = w_bio_bio2)

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
			bio_bio2 = nengo.Connection(bio, bio2,
				seed=conn2_seed,
				synapse=tau,
				transform=transform,
				trained_weights=True,
				solver = bio_bio2_solver,
				n_syn=n_syn)
			nengo.Connection(lif, lif2,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=transform)
			nengo.Connection(oracle, oracle2,
				synapse=tau,
				transform=transform)
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))
			conn_lif2 = nengo.Connection(lif2, temp2,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_lif2 = nengo.Probe(lif2, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
			probe_lif2_spikes = nengo.Probe(lif2.neurons, 'spikes')
			probe_oracle2 = nengo.Probe(oracle2, synapse=tau_readout)


		"""
		Perform spike-match training on the pre-bio weights
		"""
		if train:
			network = spike_train(network, evo_params, plots=False)
			w_pre_bio_new = pre_bio.solver.weights_bio
			w_bio_bio2_new = bio_bio2.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio
			w_bio_bio2_new = w_bio_bio2

		"""
		Run the simulation with the new w_pre_bio
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt)

		"""
		Calculate readout decoders by either
			- grabbing the ideal decoders from the LIF population, OR
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		if readout == 'LIF':
			d_readout_new = sim.data[conn_lif].weights.T
			d_readout2_new = sim.data[conn_lif2].weights.T
		elif readout == 'oracle':
			d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
			d_readout2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio2, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle][:,0], xhat_lif[:,0])
		xhat_bio2 = np.dot(act_bio2, d_readout2)
		xhat_lif2 = sim.data[probe_lif2]
		rmse_bio2 = rmse(sim.data[probe_oracle2][:,0], xhat_bio2[:,0])
		rmse_lif2 = rmse(sim.data[probe_oracle2][:,0], xhat_lif2[:,0])

		error_bio = xhat_bio[:,0]-sim.data[probe_oracle][:,0]
		error_lif = xhat_lif[:,0]-sim.data[probe_oracle][:,0]
		dft_bio = np.fft.fft(error_bio)				
		dft_lif = np.fft.fft(error_lif)

		if plot == 'signals':
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), sim.data[probe_oracle][:,0], label='oracle')
			plt.plot(sim.trange(), xhat_bio2[:,0], label='bio2, rmse=%.5f' % rmse_bio2)
			plt.plot(sim.trange(), xhat_lif2[:,0], label='lif2, rmse=%.5f' % rmse_lif2)
			plt.plot(sim.trange(), sim.data[probe_oracle2][:,0], label='oracle2')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()
		# elif plot == 'errors':
		# 	plt.plot(sim.trange(), error_bio, label='bio')
		# 	plt.plot(sim.trange(), error_lif, label='lif')
		# 	plt.xlabel('time (s)')
		# 	plt.ylabel('Error ($\hat{x}(t)$)')
		# 	plt.legend()
		# elif plot == 'dfts':
		# 	plt.plot(sim.trange(), dft_bio, label='bio')
		# 	plt.plot(sim.trange(), dft_lif, label='lif')
		# 	plt.xlabel('time (s)')
		# 	plt.ylabel('DFT (error ($\hat{x}(t)$))')
		# 	# plt.title('freq=%s' %freq)
		# 	plt.legend()

		return w_pre_bio_new, w_bio_bio2_new, d_readout_new, d_readout2_new, rmse_bio


	"""
	Run the test
	"""
	weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
	weight_filename = 'w_bio_to_bio_pre_bio_freq=%s.npz' % 1
	weight2_filename = 'w_bio_to_bio_bio_bio2_freq=%s.npz' % 1
	try:
		w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
		w_bio_bio2_init = np.load(weight_dir+weight2_filename)['weights_bio']
		to_train = False
	except IOError:
		w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
		w_bio_bio2_init = np.zeros((bio2_neurons, bio_neurons, n_syn))
		to_train = True
	d_readout_init = np.zeros((bio_neurons, dim))
	d_readout2_init = np.zeros((bio_neurons, dim))

	w_pre_bio_new, w_bio_bio2_new, d_readout_new, d_readout2_new, rmse_bio = sim(
		w_pre_bio=w_pre_bio_init,
		w_bio_bio2=w_bio_bio2_init,
		d_readout=d_readout_init,
		d_readout2=d_readout2_init,
		evo_params=evo_params,
		signal='sinusoids',
		freq=1,
		seeds=1,
		transform=1,
		t_final=t_final,
		train=to_train,
		plot=False)
	w_pre_bio_extra, w_bio_bio2_extra, d_readout_extra, d_readout2_extra, rmse_bio = sim(
		w_pre_bio=w_pre_bio_new,
		w_bio_bio2=w_bio_bio2_new,
		d_readout=d_readout_new,
		d_readout2=d_readout2_new,
		evo_params=evo_params,
		signal='sinusoids',
		freq=1,
		seeds=1,
		transform=1,
		t_final=t_final,
		train=False,
		plot='signals')

	np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)
	np.savez(weight_dir+weight2_filename, weights_bio=w_bio_bio2_new)

	assert rmse_bio < cutoff