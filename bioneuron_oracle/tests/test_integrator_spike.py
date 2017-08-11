import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, TrainedSolver, spike_train
from nengolib.signal import s

def test_integrator_1d(Simulator, plt):
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
	bio2_radius = 1
	n_syn = 1

	pre_seed = 1
	bio_seed = 2
	conn_seed = 3
	network_seed = 4
	sim_seed = 5
	post_seed = 6
	inter_seed = 7
	conn2_seed = 9

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
	transform = 10*tau

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
		w_bio_bio,
		d_readout_bio,
		d_readout_lif,
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
			bio_bio_solver = TrainedSolver(weights_bio = w_bio_bio)

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
			bio_bio = nengo.Connection(bio, bio,
				seed=conn2_seed,
				synapse=tau,
				transform=1,
				trained_weights=True,
				solver = bio_bio_solver,
				n_syn=n_syn)
			nengo.Connection(lif, lif,
				synapse=tau,
				transform=1)
			nengo.Connection(stim, oracle,
				# synapse=1/s,
				synapse=nengo.LinearFilter([1.], [1., 0]),
				transform=1)
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
			w_bio_bio_new = bio_bio.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio
			w_bio_bio_new = w_bio_bio

		"""
		Run the simulation with the new w_pre_bio
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
		x_target = sim.data[probe_oracle]

		"""
		Calculate readout decoders by either
			- grabbing the ideal decoders from the LIF population, OR
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		if readout == 'LIF':
			d_readout_bio_new = sim.data[conn_lif].weights.T
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout == 'oracle':
			d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, x_target)[0]
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, x_target)[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][:,0]
		xhat_bio = np.dot(act_bio, d_readout_bio)[:,0]
		xhat_lif = np.dot(act_lif, d_readout_lif)[:,0]
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)

		if plot:
			# plt.plot(sim.trange(), sim.data[probe_stim][:,0], label='stim')
			plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target, label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return w_pre_bio_new, w_bio_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio


	"""
	Run the test
	"""
	weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
	weight_filename = 'w_integrator_1d_pre_bio_10x_transform.npz'
	weight2_filename = 'w_integrator_1d_bio_bio_10x_transform.npz'
	try:
		w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
		w_bio_bio_init = np.load(weight_dir+weight2_filename)['weights_bio']
		to_train = False
	except IOError:
		w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
		w_bio_bio_init = np.zeros((bio_neurons, bio_neurons, n_syn))
		to_train = True
	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	w_pre_bio_new, w_bio_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		w_pre_bio=w_pre_bio_init,
		w_bio_bio=w_bio_bio_init,
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		evo_params=evo_params,
		readout='oracle',
		signal='sinusoids',
		freq=freq_train,
		seeds=seed_train,
		transform=transform,
		t_final=t_final,
		train=to_train,
		plot=False)
	w_pre_bio_extra, w_bio_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		w_pre_bio=w_pre_bio_new,
		w_bio_bio=w_bio_bio_new,
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		evo_params=evo_params,
		readout='oracle',
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		train=False,
		plot=True)

	np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)
	np.savez(weight_dir+weight2_filename, weights_bio=w_bio_bio_new)

	assert rmse_bio < cutoff