import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, get_stimulus, train_filters_decoders, build_filter


def test_representation_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 10
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

	pre_seed = 1
	bio_seed = 2
	conn_seed = 3
	network_seed = 4
	sim_seed = 5
	post_seed = 6
	inter_seed = 7

	max_freq = 5
	rms = 0.25
	t_transient = 0.1
	t_train = 1.0
	t_test = 1.0
	whitenoise_seed_train = 1
	whitenoise_seed_test = 2

	n_processes = 10
	evo_popsize = 10
	evo_gen = 10
	evo_seed = 1
	b_s_init = [1e0]  # numerator init values, [a, b, c] ==> c + bs + as^2
	a_s_init = [1e-1, 1e0]  # denominator init values
	delta_b_s = [1e-1]  # numerator mutation constants
	delta_a_s = [1e-1, 1e-2]  # denominator mutation constants
	filter_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/filters/'
	filter_filename = 'representation'

	signal_train = get_stimulus(network_seed, sim_seed, whitenoise_seed_train, t_transient, t_train, max_freq, rms, dt)
	signal_test = get_stimulus(network_seed, sim_seed, whitenoise_seed_test, t_transient, t_test, max_freq, rms, dt)

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
			alif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=dim,
				seed=bio.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				radius=bio.radius,
				neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05),
				label='alif')
			oracle = nengo.Node(size_in=dim)
			temp = nengo.Node(size_in=dim)

			nengo.Connection(stim, pre, synapse=None)
			nengo.Connection(stim, oracle, synapse=tau)
			nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				n_syn=n_syn)
			nengo.Connection(pre, lif, synapse=tau)
			nengo.Connection(pre, alif, synapse=tau)
			network.conn_lif = nengo.Connection(lif, temp, synapse=None)

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_pre = nengo.Probe(pre, synapse=tau_readout)
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_alif_spikes = nengo.Probe(alif.neurons, 'spikes')
			probe_lif_activity = nengo.Probe(lif.neurons, 'spikes', synapse=tau_readout)
			probe_bio_activity = nengo.Probe(bio.neurons, 'spikes', synapse=tau_readout)
			probe_alif_activity = nengo.Probe(alif.neurons, 'spikes', synapse=tau_readout)
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)

		network.bio = bio
		network.alif = alif
		network.stim = stim
		network.bio_probe = probe_bio_activity
		network.probe_lif_activity = probe_lif_activity
		network.alif_probe = probe_alif_activity
		network.target_probe = probe_oracle

		return network


	""" 
	Use 1 1+lambda evolutionary algorithm to optimize the readout filters and readout decoders
	for the bioneurons and alifs, then add probes with those filters into the network
	"""
	network = make_network()
	try:
		filter_info = np.load(filter_dir+filter_filename)
		b_s_bio = filter_info['b_s_bio']
		b_s_alif = filter_info['b_s_alif']
		a_s_bio = filter_info['a_s_bio']
		a_s_alif = filter_info['a_s_alif']
		d_bio = filter_info['d_bio']
		d_alif = filter_info['d_alif']
		f_bio = build_filter(b_s_bio, a_s_bio)
		f_alif = build_filter(b_s_alif, a_s_alif)
	except IOError:
		b_s_bio, b_s_alif, a_s_bio, a_s_alif, d_bio, d_alif = train_filters_decoders(
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
			b_s_init,
			a_s_init,
			delta_b_s,
			delta_a_s)
		np.savez(filter_dir+filter_filename,
			b_s_bio=b_s_bio,
			b_s_alif=b_s_alif,
			a_s_bio=a_s_bio,
			a_s_alif=a_s_alif,
			d_bio=d_bio,
			d_alif=d_alif)
		f_bio = build_filter(b_s_bio, a_s_bio)
		f_alif = build_filter(b_s_alif, a_s_alif)

	with network:
		network.stim.output = lambda t: signal_test[int(t/dt)]
		probe_bio_activity = nengo.Probe(network.bio.neurons, 'spikes', synapse=f_bio)
		probe_alif_activity = nengo.Probe(network.alif.neurons, 'spikes', synapse=f_alif)

	"""
	Simulate the network, collect the filtered bioneuron activities and target values,
	and decode the activities to estimate the state 
	"""
	with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
		sim.run(t_transient+t_test)
	act_bio = sim.data[probe_bio_activity]
	act_lif = sim.data[network.probe_lif_activity]
	act_alif = sim.data[probe_alif_activity]
	x_target = sim.data[network.target_probe][int(t_transient/dt):]
	xhat_bio = np.dot(act_bio, d_bio)[int(t_transient/dt):]
	xhat_lif = np.dot(act_lif, sim.data[network.conn_lif].weights.T)[int(t_transient/dt):]
	xhat_alif = np.dot(act_alif, d_alif)[int(t_transient/dt):]
	rmse_bio = rmse(x_target, xhat_bio)
	rmse_lif = rmse(x_target, xhat_lif)
	rmse_alif = rmse(x_target, xhat_alif)

	# plt.plot(sim.trange(), sim.data[probe_pre], label='pre')
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_alif, label='adaptive lif, rmse=%.5f' % rmse_alif)
	plt.plot(sim.trange()[int(t_transient/dt):], x_target, label='oracle')
	plt.xlabel('time (s)')
	plt.ylabel('$\hat{x}(t)$')
	plt.legend()

	assert rmse_bio < cutoff