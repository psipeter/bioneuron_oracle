import numpy as np
import nengo
import collections
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver, get_stim_deriv, train_filters_decoders, build_filter
from nengolib.signal import s, z

def test_integrator_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 200  # 200
	tau = 0.1
	tau_readout = 0.1
	tau_decoders = 0.1
	tau_JL = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
	bio_radius = 1
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
	conn2_seed = 9
	conn3_seed = 10

	freq = 10
	max_freq = 5
	rms = 0.5
	t_transient = 0.5
	t_train = 1.0  # 10
	t_test = 3.0
	signal_type = 'white_noise'
	seed_train = 1
	seed_test = 2

	n_processes = 10
	evo_popsize = 10
	evo_gen = 3  # 30
	evo_seed = 1
	zeros = [1e2]
	poles = [-1e2, -1e2]
	delta_zeros = [1e1]
	delta_poles = [1e1, 1e1]
	filter_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/filters/'
	filter_filename_base = 'integrator_test_bio_3'

	signal_train, _ = get_stim_deriv(
		signal_type, network_seed, sim_seed, freq, seed_train, t_transient, t_train, max_freq, rms, tau, dt)
	signal_test, _ = get_stim_deriv(
		signal_type, network_seed, sim_seed, freq, seed_test, t_transient, t_test, max_freq, rms, tau, dt)

	def make_network(filt_stim_oracle, T_bio_bio, T_inter_bio, d_recurrent_bio, evolved_filt):
		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:
			stim = nengo.Node(lambda t: signal_train[int(t/dt)])
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
			pre_inter = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre_inter')
			inter = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				seed=bio_seed,
				neuron_type=nengo.LIF(),
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=radius,
				label='inter')
			lif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=dim,
				seed=bio.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif')
			oracle = nengo.Node(size_in=dim)
			temp = nengo.Node(size_in=dim)

			recurrent_solver = OracleSolver(decoders_bio = d_recurrent_bio)

			nengo.Connection(stim, pre, synapse=None)
			nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				n_syn=n_syn,
				transform=1*tau)
			bio_bio2 = nengo.Connection(bio, bio,
				weights_bias_conn=False,
				seed=conn2_seed,
				synapse=tau,
				n_syn=n_syn,
				solver=recurrent_solver,
				transform=T_bio_bio)

			nengo.Connection(pre, lif,
				synapse=tau,
				transform=tau)
			nengo.Connection(lif, lif,
				synapse=tau)

			nengo.Connection(stim, oracle, synapse=filt_stim_oracle, transform=1)
			nengo.Connection(oracle, pre_inter, seed=conn3_seed, synapse=None)
			nengo.Connection(pre_inter, inter, synapse=evolved_filt)  # may need another oracle decode on evolved filtered LIF spikes
			nengo.Connection(inter, bio,
				weights_bias_conn=False,
				seed=conn3_seed,
				synapse=tau,
				n_syn=n_syn,
				transform=T_inter_bio)	
			network.conn_lif = nengo.Connection(lif, temp, synapse=tau)

			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_activity = nengo.Probe(lif.neurons, 'spikes', synapse=tau_readout)
			probe_bio_activity = nengo.Probe(bio.neurons, 'spikes', synapse=evolved_filt)
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)

		network.stim = stim
		network.bio_probe = probe_bio_activity
		network.lif_probe = probe_lif_activity
		network.target_probe = probe_oracle

		return network

	def train(network, zeros, poles, delta_zeros, delta_poles, filter_filename):
		""" 
		Use 1 1+lambda evolutionary algorithm to optimize the readout filters and readout decoders
		for the bioneurons and alifs, then add probes with those filters into the network
		"""
		zeros, poles, d_bio = train_filters_decoders(
			network,
			Simulator,
			sim_seed,
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
			network.bio_probe,
			network.target_probe, 
			)
		np.savez(filter_dir+filter_filename,
			zeros=zeros,
			poles=poles,
			d_bio=d_bio)
		f_bio = build_filter(zeros, poles)
		return f_bio, d_bio

	"""
	1. Feedforward pass to evolve the readout filters for bio (bio-bio off, stim-oracle=tau, inter off)
	"""
	fname = filter_filename_base + '_ff.npz'
	try:
		filter_info = np.load(filter_dir+fname)
		zeros = filter_info['zeros']
		poles = filter_info['poles']
		evolved_d_feedforward = filter_info['d_bio']
		evolved_filt_feedforward = build_filter(zeros, poles)
	except IOError:
		network = make_network(
			filt_stim_oracle=nengo.Lowpass(tau),
			T_bio_bio=0.0,
			T_inter_bio=0.0,
			d_recurrent_bio=np.zeros((bio_neurons, dim)),
			evolved_filt=nengo.Lowpass(tau))
		evolved_filt_feedforward, evolved_d_feedforward = train(network, zeros, poles, delta_zeros, delta_poles, fname)
	"""
	2. Integrator pass with bio-bio off, stim-oracle=1/s, inter on, assigning the evolved filter on
		stim - (None)pre-inter - (evo)inter, so that inter's spikes (sent to bio) hopefully better reflect
		the filters applied by the bioneurons' dendrites.
		Evolve new readout decoders for bio
	"""
	fname = filter_filename_base + '_inter_fb.npz'
	try:
		filter_info = np.load(filter_dir+fname)
		zeros = filter_info['zeros']
		poles = filter_info['poles']
		evolved_d_inter_feedback = filter_info['d_bio']
		evolved_filt_inter_feedback = build_filter(zeros, poles)
	except IOError:
		network = make_network(
			filt_stim_oracle=1/s, # 1/s
			T_bio_bio=0.0,
			T_inter_bio=1.0,  # 1.0
			d_recurrent_bio=np.zeros((bio_neurons, dim)),
			evolved_filt=evolved_filt_feedforward)
		evolved_filt_inter_feedback, evolved_d_inter_feedback = train(network, zeros, poles, delta_zeros, delta_poles, fname)
	"""
	3. Integrator pass with bio-bio on, stim-oracle=1/s, and inter off, using last evolved decoders to readout.
		Evolve final readout filters and decoders 
	"""
	network = make_network(
		filt_stim_oracle=1/s, # 1/s
		T_bio_bio=1.0,  # 1.0
		T_inter_bio=0.0,
		d_recurrent_bio=evolved_d_inter_feedback,
		evolved_filt=evolved_filt_inter_feedback)
	f_bio = evolved_filt_inter_feedback
	d_bio = evolved_d_inter_feedback
	"""
	4. (optional) Integrator pass with bio-bio on, stim-oracle=1/s, and inter off,
		but performing a new evolution from the starting point of the evolved_d_inter_feedback decoders.
		This may help if inter spikes are still different from bio spikes
	"""
	# fname = filter_filename_base + '_bio_fb.npz'
	# try:
	# 	filter_info = np.load(filter_dir+fname)
	# 	zeros = filter_info['zeros']
	# 	poles = filter_info['poles']
	# 	evolved_d_bio_feedback = filter_info['d_bio']
	# 	evolved_filt_bio_feedback = build_filter(zeros, poles)
	# except IOError:
	# 	evolved_filt_bio_feedback, evolved_d_bio_feedback = train(network, zeros, poles, delta_zeros, delta_poles, fname)
	# 	network = make_network(
	# 		filt_stim_oracle=1/s,
	# 		T_bio_bio=1.0,
	# 		T_inter_bio=0.0,
	# 		d_recurrent_bio=evolved_d_inter_feedback,  # evolved_d_bio_feedback
	# 		evolved_filt=evolved_filt_inter_feedback)  # evolved_filt_bio_feedback
	# f_bio = evolved_filt_bio_feedback
	# d_bio = evolved_d_bio_feedback


	"""
	Simulate the network, collect the filtered bioneuron activities and target values,
	and decode the activities to estimate the state 
	"""
	with network:
		network.stim.output = lambda t: signal_test[int(t/dt)]
		network.bio_probe.synapse = f_bio
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