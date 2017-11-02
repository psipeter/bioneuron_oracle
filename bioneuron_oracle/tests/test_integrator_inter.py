import numpy as np
import nengo
import collections
from nengo.utils.numpy import rmse
from bioneuron_oracle import (BahlNeuron, OracleSolver, get_signals, filt_fwd_bwd,
	train_filters_decoders, train_feedforward, train_feedback, build_filter)
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

	freq = 1
	max_freq = 5
	rms = 0.5
	t_transient = 0.5
	t_train = 10.0  # 10
	t_test = 10.0
	signal_type_train = 'sinusoids'
	signal_type_test = 'white_noise'
	seed_train = 3
	seed_test = 2
	bio_type =  nengo.LIF() # BahlNeuron() # nengo.AdaptiveLIF(tau_n=.01, inc_n=.05)
	inter_type = nengo.LIF() # BahlNeuron() # nengo.AdaptiveLIF(tau_n=.01, inc_n=.05)

	n_processes = 10
	evo_popsize = 10
	evo_gen_feedforward = 10
	evo_gen_feedback = 11
	evo_seed = 1
	zeros_init = [1e2]
	poles_init = [-1e2, -1e2]
	zeros_delta = [1e1]
	poles_delta = [1e1, 1e1]
	training_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/filters/'
	training_file_base = 'integrator_inter_%sneurons_%ss_%spop_%sgenff_%sgenfb_%stype'\
		%(bio_neurons, t_train, evo_popsize, evo_gen_feedforward, evo_gen_feedback, bio_type)

	def make_network(
		signal,
		integral,
		T_stim_oracle,
		T_integ_oracle,
		T_bio_bio,
		T_inter_bio,
		T_pre_bio,
		T_pre_lif,
		d_feedback,
		d_inter,
		f_feedforward):

		""" apply readout filter to the integrated signal """
		forward, backward = filt_fwd_bwd(integral, f_feedforward)
		forward /= max(abs(forward))
		backward /= max(abs(backward))

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:
			stim = nengo.Node(lambda t: signal[int(t/dt)])
			integ = nengo.Node(lambda t: integral[int(t/dt)])
			fwd = nengo.Node(lambda t: forward[int(t/dt)])
			bwd = nengo.Node(lambda t: backward[int(t/dt)])
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
				# neuron_type=BahlNeuron(),
				# neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05),
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
				neuron_type=inter_type,
				# neuron_type=BahlNeuron(),
				# neuron_type=nengo.Direct(),
				# neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05),
				# neuron_type=nengo.AdaptiveLIF(tau_n=0.09, inc_n=0.0095),
				# neuron_type=nengo.LIF(),
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				radius=bio_radius,
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
			temp2 = nengo.Node(size_in=dim)

			recurrent_solver = OracleSolver(decoders_bio = d_feedback)
			# inter_solver = OracleSolver(decoders_bio = d_inter)
			inter_solver = nengo.solvers.NoSolver(d_inter)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				n_syn=n_syn,
				transform=T_pre_bio)
			bio_bio = nengo.Connection(bio, bio,
				weights_bias_conn=False,
				seed=conn2_seed,
				synapse=tau,
				n_syn=n_syn,
				solver=recurrent_solver,
				transform=T_bio_bio)

			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				transform=T_pre_lif)
			nengo.Connection(lif, lif,
				synapse=tau)

			stim_oracle = nengo.Connection(stim, oracle,
				synapse=tau,  
				transform=T_stim_oracle) # connection for training H(s) w/ feedforward pass
			integ_oracle = nengo.Connection(integ, oracle,
				synapse=None,  
				transform=T_integ_oracle) # connection for training d_feedback w/ feedback pass 

			# nengo.Connection(integ, pre_inter, synapse=None)
			# integ_inter = nengo.Connection(pre_inter, inter,
			# 	weights_bias_conn=True,
			# 	seed=conn_seed,
			# 	synapse=tau,
			# 	n_syn=n_syn)
			# nengo.Connection(inter, bio,
			# 	weights_bias_conn=False,
			# 	seed=conn2_seed, 
			# 	synapse=tau,  # only for ExpSyn on bio?
			# 	n_syn=n_syn,
			# 	transform=T_inter_bio,
			# 	solver=inter_solver) 

			integ_inter = nengo.Connection(integ, inter,  # no H(s)
				seed=conn3_seed,
				synapse=None)
			nengo.Connection(inter, bio,
				weights_bias_conn=False,
				seed=conn3_seed,
				synapse=tau,  # only for ExpSyn on bio?
				n_syn=n_syn,
				# need to increase this transform for alif to account for fewer spikes they produce
				# OR precompute readout decoders for inter with oracle method and save them to a NoSolver
				transform=T_inter_bio,
				solver=inter_solver)  

			network.conn_lif = nengo.Connection(lif, temp, synapse=tau)
			network.conn_alif = nengo.Connection(inter, temp2, synapse=tau, solver=inter_solver)

			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_activity = nengo.Probe(lif.neurons, 'spikes', synapse=tau_readout)
			probe_bio_activity = nengo.Probe(bio.neurons, 'spikes', synapse=f_feedforward)
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_inter = nengo.Probe(inter, synapse=tau_readout, solver=inter_solver)
			probe_inter_activity = nengo.Probe(inter.neurons, 'spikes', synapse=tau_readout)
			probe_inter_spikes = nengo.Probe(inter.neurons, 'spikes', synapse=None)
			probe_pre = nengo.Probe(pre, synapse=tau_readout)
			probe_integ = nengo.Probe(integ, synapse=None)

		network.stim = stim
		network.integ = integ
		network.pre_bio = pre_bio
		network.pre_lif = pre_lif
		network.stim_oracle = stim_oracle
		network.integ_oracle = integ_oracle
		network.integ_inter = integ_inter
		network.probe_bio_spikes = probe_bio_spikes
		network.probe_bio_activity = probe_bio_activity
		network.lif_probe = probe_lif_activity
		network.probe_oracle = probe_oracle
		network.probe_inter = probe_inter
		network.probe_inter_activity = probe_inter_activity
		network.probe_inter_spikes = probe_inter_spikes
		network.probe_pre = probe_pre
		network.probe_integ = probe_integ

		return network

	"""
	0. Create normalized signal and integral
	"""
	signal_train, integral_train, transform_train = get_signals(
		signal_type_train, network_seed, sim_seed, freq, seed_train, t_transient, t_train, max_freq, rms, tau, dt)
	signal_test, integral_test, transform_test = get_signals(
		signal_type_test, network_seed, sim_seed, freq, seed_test, t_transient, t_test, max_freq, rms, tau, dt)
	scale_pre_train = tau * transform_train
	scale_pre_test = tau *transform_test

	"""
	1. Feedforward pass to evolve the readout filters for bio (bio-bio=0, stim-oracle=1, integ-oracle=0)
		and the alif decoders for inter
	"""
	training_file = training_file_base + '_ff.npz'
	try:
		training_info = np.load(training_dir+training_file)
		zeros_feedforward = training_info['zeros_feedforward']
		poles_feedforward = training_info['poles_feedforward']
		d_feedforward = training_info['d_feedforward']
		d_inter = training_info['d_inter']
		f_feedforward = build_filter(zeros_feedforward, poles_feedforward)
	except IOError:
		''' we want only a lowpass filter for bio-bio feedback during testing, 
		so make sure to compute d_feedback using that H(s) during this training '''
		network = make_network(
			signal=signal_train,
			integral=integral_train,
			T_stim_oracle=1.0,
			T_integ_oracle=0.0,
			T_bio_bio=0.0,
			T_inter_bio=0.0,
			T_pre_bio=1.0,
			T_pre_lif=1.0,
			d_feedback=np.zeros((bio_neurons, dim)),
			d_inter=np.zeros((bio_neurons, dim)),
			f_feedforward=nengo.Lowpass(tau))
		zeros_feedforward, poles_feedforward, d_feedforward, d_inter = train_feedforward(
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
	2. Integrator pass with bio-bio=0, stim-oracle=0, integ-oracle=1,
		attach some filtered version of integral to the oracle
		(synapse=none, H(s) forward, or H(s) backward)
		so that inter's spikes (sent to bio) hopefully better reflect
		the filters applied by the bioneurons' dendrites.
		Evolve new readout decoders for bio
	"""
	zeros_init = []
	poles_init = [-1.0/tau]
	zeros_delta = []
	poles_delta = [1e-10]
	training_file = training_file_base + '_fb.npz'
	try:
		training_info = np.load(training_dir+training_file)
		zeros_feedback = training_info['zeros_feedforward']
		poles_feedback = training_info['poles_feedforward']
		d_feedback = training_info['d_feedforward']
		f_feedback = build_filter(zeros_feedback, poles_feedback)
	except IOError:
		network = make_network(
			signal=signal_train,
			integral=integral_train,
			T_stim_oracle=0.0,
			T_integ_oracle=1.0,
			T_bio_bio=0.0,
			T_inter_bio=1.0,
			T_pre_bio=scale_pre_train,
			T_pre_lif=scale_pre_train,
			d_feedback=np.zeros((bio_neurons, dim)),
			d_inter=d_inter,
			f_feedforward=f_feedforward)
		zeros_feedback, poles_feedback, d_feedback, _ = train_feedforward(
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
		f_feedback = build_filter(zeros_feedback, poles_feedback)

	"""
	3. Integrator pass with bio-bio on, stim-oracle=0, integ-oracle=1, and inter off,
		using last evolved decoders and H(s) to readout.
		Evolve final readout filters and decoders 
	"""
	# network = make_network(
	# 	signal=signal_train,
	# 	integral=integral_train,
	# 	T_stim_oracle=0.0,
	# 	T_integ_oracle=1.0,
	# 	T_bio_bio=0.0,
	# 	T_inter_bio=1.0,
	# 	T_pre_bio=scale_pre_train,
	# 	T_pre_lif=scale_pre_train,
	# 	d_feedback=d_feedback,
	# 	d_inter=d_inter,
	# 	f_feedforward=f_feedforward)

	# network = make_network(
	# 	signal=signal_train,
	# 	integral=integral_train,
	# 	T_stim_oracle=0.0,
	# 	T_integ_oracle=1.0,
	# 	T_bio_bio=1.0,
	# 	T_inter_bio=0.0,
	# 	T_pre_bio=scale_pre_train,
	# 	T_pre_lif=scale_pre_train,
	# 	d_feedback=d_feedback,
	# 	d_inter=d_inter,
	# 	f_feedforward=f_feedforward)
	# with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
	# 	sim.run(dt)
	# d_feedback = sim.data[network.conn_lif].weights.T
	# d_feedforward = sim.data[network.conn_lif].weights.T

	network = make_network(
		signal=signal_train,
		integral=integral_train,
		T_stim_oracle=0.0,
		T_integ_oracle=1.0,
		T_bio_bio=1.0,
		T_inter_bio=0.0,
		T_pre_bio=scale_pre_train,
		T_pre_lif=scale_pre_train,
		d_feedback=d_feedback,
		d_inter=d_inter,
		f_feedforward=f_feedforward)

	"""
	Simulate the network, collect the filtered bioneuron activities and target values,
	and decode the activities to estimate the state 
	"""
	with network:
		network.stim.output = lambda t: signal_test[int(t/dt)]
		network.integ.output = lambda t: integral_test[int(t/dt)]
		network.pre_bio.transform = scale_pre_test
		network.pre_lif.transform = scale_pre_test
		network.probe_bio_activity.synapse = f_feedforward
	with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
		sim.run(t_transient+t_test)
	act_bio = sim.data[network.probe_bio_activity]
	act_bio2 = nengo.Lowpass(tau_readout).filt(sim.data[network.probe_bio_spikes][int(t_transient/dt):])
	act_lif = sim.data[network.lif_probe]
	x_target = sim.data[network.probe_oracle][int(t_transient/dt):]
	xhat_bio = np.dot(act_bio, d_feedforward)[int(t_transient/dt):]
	xhat_bio2 = np.dot(act_bio2, sim.data[network.conn_lif].weights.T)
	xhat_lif = np.dot(act_lif, sim.data[network.conn_lif].weights.T)[int(t_transient/dt):]
	rmse_bio = rmse(x_target, xhat_bio)
	rmse_lif = rmse(x_target, xhat_lif)
	rmse_bio2 = rmse(x_target, xhat_bio2)

	# from nengo.utils.matplotlib import rasterplot
	# figure, ax1 = plt.subplots(1,1)
	# rasterplot(sim.trange(), sim.data[network.probe_bio_spikes])
	# ax1.set(xlabel='time (s)', ylabel='spikes')
	# figure.savefig('plots/%s_bio_spike_raster.png' %training_file_base)

	# plt.plot(sim.trange(), sim.data[network.probe_pre], label='pre')
	# plt.plot(sim.trange()[int(t_transient/dt):], integral_test[int(t_transient/dt):-1], label='integ')
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio2, label='bio w/ lowpass, conn_lif, rmse=%.5f' % rmse_bio2)
	plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
	# plt.plot(sim.trange()[int(t_transient/dt):], xhat_alif, label='alif, rmse=%.5f' % rmse_alif)
	plt.plot(sim.trange()[int(t_transient/dt):], x_target, label='oracle')
	# plt.plot(sim.trange()[int(t_transient/dt):], sim.data[network.probe_inter][int(t_transient/dt):], label='inter')
	plt.xlabel('time (s)')
	plt.ylabel('$\hat{x}(t)$')
	plt.legend()
	plt.saveas = training_file_base + '.png'

	# np.savez(filter_dir+filter_filename_base+'_alif_data.npz',
	# 	times=sim.trange()[int(t_transient/dt):],
	# 	act=sim.data[network.bio_probe],
	# 	xhat=xhat_bio,
	# 	rmse=rmse_bio)

	assert rmse_bio < cutoff