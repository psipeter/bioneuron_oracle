import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, get_stim_deriv, train_filters_decoders, build_filter


def test_linear_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	tau_decoders = 0.1
	tau_JL = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
	bio_radius = np.sqrt(2)
	n_syn = 1
	dim = 1
	reg = 0.01
	cutoff = 0.1
	transform = -0.5

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
	t_train = 3.0
	t_test = 3.0
	signal_type = 'white_noise'
	seed_train = 1
	seed_test = 2

	n_processes = 10
	evo_popsize = 10
	evo_gen = 20
	evo_seed = 1
	zeros = [1e2]
	poles = [-1e2, -1e2]
	delta_zeros = [1e1]
	delta_poles = [1e1, 1e1]
	filter_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/deriv_filters/'
	filter_filename = 'test.npz'

	signal_train, _ = get_stim_deriv(
		signal_type, network_seed, sim_seed, freq, seed_train, t_transient, t_train, max_freq, rms, tau, dt)
	signal_test, _ = get_stim_deriv(
		signal_type, network_seed, sim_seed, freq, seed_test, t_transient, t_test, max_freq, rms, tau, dt)

	def make_network():
		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:
			stim = nengo.Node(lambda t: signal_test[int(t/dt)])
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
			nengo.Connection(stim, oracle, synapse=tau, transform=transform)
			nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				n_syn=n_syn,
				transform=transform)
			nengo.Connection(pre, lif, synapse=tau, transform=transform)
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
	# signals = [[network.stim, signal_train]]

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
				# signals,
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
	rmse_bio = rmse(x_target, xhat_bio)
	rmse_lif = rmse(x_target, xhat_lif)

	# plt.plot(sim.trange(), sim.data[network.probe_pre], label='pre')
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
	plt.plot(sim.trange()[int(t_transient/dt):], x_target, label='oracle')
	plt.xlabel('time (s)')
	plt.ylabel('$\hat{x}(t)$')
	plt.legend()

	assert rmse_bio < cutoff