import numpy as np
import nengo
import collections
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver, get_signals, train_feedforward, train_feedback, build_filter
from nengolib.signal import s, z

def test_integrator_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100  # 300
	tau = 0.1
	tau_readout = 0.1
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
	t_train = 20.0  # 10
	t_test = 10.0
	signal_type = 'white_noise'
	seed_train = 3
	seed_test = 2
	bio_type = nengo.LIF() # nengo.AdaptiveLIF(tau_n=.09, inc_n=.0095) # BahlNeuron() # 

	n_processes = 10
	evo_popsize = 20
	evo_gen_feedforward = 20
	evo_gen_feedback = 200
	evo_seed = 1
	zeros_init = [1e2]
	poles_init = [-1e2, -1e2]
	zeros_delta = [1e1]
	poles_delta = [1e1, 1e1]
	decoders_delta = 1e-5
	mutation_rate = 0.1
	training_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/filters/'
	training_file_base = 'integrator_evolve_%sneurons_%ss_%spop_%sgenff_%sgenfb_%stype'\
		%(bio_neurons, t_train, evo_popsize, evo_gen_feedforward, evo_gen_feedback, bio_type)

	def make_network(
		signal,
		integral,
		T_stim_oracle,
		T_integ_oracle,
		T_feedforward,
		T_feedback,
		d_feedback,
		d_feedforward,
		f_feedforward):

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:
			stim = nengo.Node(lambda t: signal[int(t/dt)])
			integ = nengo.Node(lambda t: integral[int(t/dt)])
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
				neuron_type=bio_type,
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
			out = nengo.Node(size_in=dim)

			solver_feedforward = nengo.solvers.NoSolver(d_feedforward)
			solver_feedback = nengo.solvers.NoSolver(d_feedback)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				n_syn=n_syn,
				transform=T_feedforward)
			bio_bio = nengo.Connection(bio, bio,
				weights_bias_conn=False,
				seed=conn2_seed,
				synapse=tau,
				n_syn=n_syn,
				solver=solver_feedback,
				transform=T_feedback)

			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				transform=T_feedforward)
			nengo.Connection(lif, lif,
				synapse=tau,
				transform=T_feedback)
			lif_out = nengo.Connection(lif, out,
				synapse=None)
			pre_alif = nengo.Connection(pre, alif,
				synapse=tau,
				transform=T_feedforward)
			nengo.Connection(alif, alif,
				synapse=tau,
				transform=T_feedback)
			alif_out = nengo.Connection(alif, out,
				synapse=None)

			stim_oracle = nengo.Connection(stim, oracle,
				synapse=tau,  
				transform=T_stim_oracle)
			integ_oracle = nengo.Connection(integ, oracle,
				synapse=None,  
				transform=T_integ_oracle)

			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_bio_activity = nengo.Probe(bio.neurons, 'spikes', synapse=f_feedforward)
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_pre = nengo.Probe(pre, synapse=tau_readout)
			probe_bio = nengo.Probe(bio, synapse=f_feedforward, solver=solver_feedforward)
			probe_lif = nengo.Probe(lif, synapse=tau_readout)
			probe_alif = nengo.Probe(alif, synapse=tau_readout)

		network.stim = stim
		network.integ = integ
		network.pre_bio = pre_bio
		network.pre_lif = pre_lif
		network.lif_out = lif_out
		network.alif_out = alif_out
		network.stim_oracle = stim_oracle
		network.integ_oracle = integ_oracle
		network.bio_bio = bio_bio

		network.probe_pre = probe_pre
		network.probe_bio_spikes = probe_bio_spikes
		network.probe_bio_activity = probe_bio_activity
		network.probe_oracle = probe_oracle
		network.probe_bio = probe_bio
		network.probe_lif = probe_lif
		network.probe_alif = probe_alif

		return network

	"""
	0. Create normalized signal and integral
	"""
	signal_train, integral_train, transform_train = get_signals(
		signal_type, network_seed, sim_seed, freq, seed_train, t_transient, t_train, max_freq, rms, tau, dt)
	signal_test, integral_test, transform_test = get_signals(
		signal_type, network_seed, sim_seed, freq, seed_test, t_transient, t_test, max_freq, rms, tau, dt)
	scale_pre_train = tau * transform_train
	scale_pre_test = tau *transform_test

	"""
	1. Feedforward pass to evolve the readout filters for bio (bio-bio=0, stim-oracle=1, integ-oracle=0)
	"""
	training_file = training_file_base + '_ff.npz'
	try:
		training_info = np.load(training_dir+training_file)
		zeros_feedforward = training_info['zeros_feedforward']
		poles_feedforward = training_info['poles_feedforward']
		d_feedforward = training_info['d_feedforward']
		f_feedforward = build_filter(zeros_feedforward, poles_feedforward)
	except IOError:
		network = make_network(
			signal=signal_train,
			integral=integral_train,
			T_stim_oracle=1.0,
			T_integ_oracle=0.0,
			T_feedforward=1.0,
			T_feedback=0.0,
			d_feedback=np.zeros((bio_neurons, dim)),
			d_feedforward=np.zeros((bio_neurons, dim)),
			f_feedforward=nengo.Lowpass(tau))
		zeros_feedforward, poles_feedforward, d_feedforward, _ = train_feedforward(
			network,
			Simulator,
			sim_seed,
			t_transient,
			t_train,
			dt,
			reg,
			n_processes,
			evo_popsize,
			evo_gen_feedforward,
			evo_seed,
			zeros_init,
			poles_init,
			zeros_delta,
			poles_delta,
			network.probe_bio_activity,
			network.probe_oracle,
			training_dir,
			training_file)
		f_feedforward = build_filter(zeros_feedforward, poles_feedforward)


	"""
	2. Feedback pass to evolve the recurrent decoders for bio (bio-bio=1, stim-oracle=0, integ-oracle=1),
		assuming a decode with the feedforward filters and decoders.
		Initial feedback decoders calculated from
			- standard LIFRate eval_points and activities
			- oracle method with lowpass filter
			- oracle method with evolved filter (=d_feedforward from (1.))
	"""
	network = make_network(
		signal=signal_train,
		integral=integral_train,
		T_stim_oracle=1.0,
		T_integ_oracle=0.0,
		T_feedforward=1.0,
		T_feedback=0.0,
		d_feedback=np.zeros((bio_neurons, dim)),
		d_feedforward=np.zeros((bio_neurons, dim)),
		f_feedforward=nengo.Lowpass(tau))
	with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
		sim.run(t_transient+t_train)
	''' eval_points '''
	# d_feedforward = sim.data[network.lif_out].weights.T
	# f_feedforward = nengo.Lowpass(tau_readout)
	# zeros_feedforward = []
	# poles_feedforward = [-1.0/tau_readout]
	# d_feedback_init = sim.data[network.lif_out].weights.T
	''' lowpass '''
	act_bio = nengo.Lowpass(tau_readout).filt(sim.data[network.probe_bio_spikes])[int(t_transient/dt):]
	x_target = sim.data[network.probe_oracle][int(t_transient/dt):]
	d_feedback_init = nengo.solvers.LstsqL2(reg=reg)(act_bio, x_target)[0]
	''' evolved '''
	# d_feedback_init = d_feedforward

	training_file = training_file_base + '_fb.npz'
	try:
		training_info = np.load(training_dir+training_file)
		d_feedback = training_info['d_feedback']
	except IOError:
		network = make_network(
			signal=signal_train,
			integral=integral_train,
			T_stim_oracle=0.0,
			T_integ_oracle=1.0,
			T_feedforward=scale_pre_train,
			T_feedback=1.0,
			d_feedback=np.zeros((bio_neurons, dim)),
			d_feedforward=d_feedforward,
			f_feedforward=f_feedforward)
		d_feedback = train_feedback(
			network,
			Simulator,
			sim_seed,
			t_transient,
			t_train,
			dt,
			reg,
			n_processes,
			evo_popsize,
			evo_gen_feedback,
			evo_seed,
			zeros_feedforward,
			poles_feedforward,
			d_feedforward,
			d_feedback_init,
			network.bio_bio,
			decoders_delta,
			mutation_rate,
			network.probe_bio_activity,
			network.probe_oracle,
			training_dir,
			training_file)


	# """
	# 2.5. Feedback pass to further evolve the readout filters for bio, given an active feedback connection
	# """
	# training_file2 = training_file_base + '_fb2.npz'
	# try:
	# 	training_info2 = np.load(training_dir+training_file2)
	# 	zeros_feedforward2 = training_info2['zeros_feedforward']
	# 	poles_feedforward2 = training_info2['poles_feedforward']
	# 	d_feedforward2 = training_info2['d_feedforward']
	# 	f_feedforward2 = build_filter(zeros_feedforward, poles_feedforward)
	# except IOError:
	# 	network = make_network(
	# 		signal=signal_train,
	# 		integral=integral_train,
	# 		T_stim_oracle=0.0,
	# 		T_integ_oracle=1.0,
	# 		T_feedforward=scale_pre_train,
	# 		T_feedback=1.0,
	# 		d_feedback=d_feedback,
	# 		d_feedforward=d_feedforward,
	# 		f_feedforward=f_feedforward)
	# 	zeros_feedforward2, poles_feedforward2, d_feedforward2, _ = train_feedforward(
	# 		network,
	# 		Simulator,
	# 		sim_seed,
	# 		t_transient,
	# 		t_train,
	# 		dt,
	# 		reg,
	# 		n_processes,
	# 		evo_popsize,
	# 		evo_gen_feedforward,
	# 		evo_seed,
	# 		zeros_init,
	# 		poles_init,
	# 		zeros_delta,
	# 		poles_delta,
	# 		network.probe_bio_activity,
	# 		network.probe_oracle,
	# 		training_dir,
	# 		training_file2)
	# 	f_feedforward2 = build_filter(zeros_feedforward, poles_feedforward)


	"""
	3. Build the network with the trained filters and decoders in the appropriate places,
		then simulate the network, collect the filtered bioneuron activities and target values,
		and decode the activities to estimate the state.
	"""
	network = make_network(
		# signal=signal_train,
		signal=signal_test,
		# integral=integral_train,
		integral=integral_test,
		T_stim_oracle=0.0,
		T_integ_oracle=1.0,
		# T_feedforward=scale_pre_train,
		T_feedforward=scale_pre_test,
		T_feedback=1.0,
		d_feedback=d_feedback,
		d_feedforward=d_feedforward,
		# d_feedforward=d_feedforward2,
		f_feedforward=f_feedforward)
		# f_feedforward=f_feedforward2)
	with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
		sim.run(t_transient+t_test)
	act_bio2 = nengo.Lowpass(tau_readout).filt(sim.data[network.probe_bio_spikes][int(t_transient/dt):])
	x_target = sim.data[network.probe_oracle][int(t_transient/dt):]
	xhat_bio = sim.data[network.probe_bio][int(t_transient/dt):]
	xhat_lif = sim.data[network.probe_lif][int(t_transient/dt):]
	xhat_alif = sim.data[network.probe_alif][int(t_transient/dt):]
	xhat_bio2 = np.dot(act_bio2, sim.data[network.lif_out].weights.T)
	rmse_bio = rmse(x_target, xhat_bio)
	rmse_lif = rmse(x_target, xhat_lif)
	rmse_alif = rmse(x_target, xhat_alif)
	rmse_bio2 = rmse(x_target, xhat_bio2)

	"""
	4. Plot decoded estimates and target value
	"""
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio2, label='bio w/ static decoders, rmse=%.5f' % rmse_bio2)
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
	# plt.plot(sim.trange()[int(t_transient/dt):], xhat_alif, label='alif, rmse=%.5f' % rmse_alif)
	plt.plot(sim.trange()[int(t_transient/dt):], x_target, label='oracle')
	plt.xlabel('time (s)')
	plt.ylabel('$\hat{x}(t)$')
	plt.legend()
	# plt.saveas = plt.get_filename(ext='.png')
	plt.saveas = training_file_base + '.png'

	assert rmse_bio < cutoff