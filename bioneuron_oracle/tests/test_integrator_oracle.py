
import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver, get_signal, get_stim_deriv
from nengolib.signal import s, z

# def test_integrator_1d(Simulator, plt):
# 	# Nengo Parameters
# 	pre_neurons = 100
# 	bio_neurons = 100
# 	tau = 0.1
# 	tau_readout = 0.1
# 	dt = 0.001
# 	min_rate = 150
# 	max_rate = 200
# 	radius = 1
# 	bio_radius = 1
# 	n_syn = 1

# 	pre_seed = 1
# 	bio_seed = 2
# 	conn_seed = 3
# 	network_seed = 4
# 	sim_seed = 5
# 	post_seed = 6
# 	inter_seed = 7
# 	conn2_seed = 8
# 	conn3_seed = 8

# 	max_freq = 2*0.5*np.pi  # between f=0 and f=2*f_sinusoid 
# 	rms = 0.25
# 	n_steps = 10

# 	signal_train = 'white_noise'
# 	freq_train = 1
# 	seed_train = 3
# 	transform_train = 20.0
# 	t_train = 10.0

# 	signal_test = 'white_noise'
# 	freq_test = 1
# 	seed_test = 3
# 	transform_test = 20.0
# 	t_test = 1.0

# 	dim = 1
# 	reg = 0.1
# 	t_final = 1.0
# 	cutoff = 0.1
# 	jl_dims = 3
# 	jl_dim_mag = 2.5e-4

# 	def sim(
# 		d_recurrent_bio,
# 		d_readout_bio,
# 		d_readout_lif,
# 		d_JL,
# 		w_train,
# 		readout_LIF = 'LIF',
# 		signal='sinusoids',
# 		t_final=1.0,
# 		freq=1,
# 		seeds=1,
# 		transform=1,
# 		plot=False):

# 		"""
# 		Load the recurrent decoders, with the non-JL dimensions,
# 		scaled by the training factor, w_train. w_train==1 means only oracle
# 		spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
# 		and intermediate values are a weighted mix.
# 		"""
# 		d_recurrent_bio[:dim] *= (1.0 - w_train)

# 		"""
# 		Define the network
# 		"""
# 		with nengo.Network(seed=network_seed) as network:

# 			if signal == 'sinusoids':
# 				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq * t),
# 					label='stim')
# 			elif signal == 'white_noise':
# 				stim = nengo.Node(nengo.processes.WhiteSignal(
# 					period=t_final, high=max_freq, rms=rms, seed=seeds),
# 					label='stim')
# 			elif signal == 'step':
# 				stim = nengo.Node(lambda t:
# 					np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
# 			elif signal == 'constant':
# 				stim = nengo.Node(lambda t: freq)

# 			pre = nengo.Ensemble(
# 				n_neurons=pre_neurons,
# 				dimensions=dim,
# 				seed=pre_seed,
# 				neuron_type=nengo.LIF(),
# 				radius=radius,
# 				label='pre')
# 			bio = nengo.Ensemble(
# 				n_neurons=bio_neurons,
# 				dimensions=dim+jl_dims,
# 				seed=bio_seed,
# 				neuron_type=BahlNeuron(),
# 				# neuron_type=nengo.LIF(),
# 				radius=bio_radius,
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				label='bio')
# 			inter = nengo.Ensemble(
# 				n_neurons=bio_neurons,
# 				dimensions=dim,
# 				seed=bio_seed,
# 				neuron_type=nengo.LIF(),
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				# radius=radius,
# 				label='inter')
# 			lif = nengo.Ensemble(
# 				n_neurons=bio.n_neurons,
# 				dimensions=dim,
# 				seed=bio.seed,
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				# radius=bio.radius,
# 				neuron_type=nengo.LIF(),
# 				label='lif')
# 			oracle = nengo.Node(size_in=dim, label='oracle')
# 			temp = nengo.Node(size_in=dim, label='temp')

# 			recurrent_solver = OracleSolver(decoders_bio = d_recurrent_bio)

# 			nengo.Connection(stim, pre,
# 				synapse=None)
# 			''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
# 			nengo.Connection(pre, bio[:dim],
# 				weights_bias_conn=True,
# 				seed=conn_seed,
# 				synapse=tau,
# 				transform=transform*tau)
# 			nengo.Connection(pre, lif,
# 				synapse=tau,
# 				transform=transform*tau)
# 			''' Connect recurrent (spikes) feedback to all dims of bio '''
# 			nengo.Connection(bio, bio,
# 				seed=conn2_seed,
# 				synapse=tau,
# 				solver=recurrent_solver)
# 			nengo.Connection(lif, lif,
# 				synapse=tau)
# 			nengo.Connection(stim, oracle,
# 				synapse=1/s,
# 				transform=transform)
# 			nengo.Connection(oracle, inter,
# 				seed=conn3_seed,
# 				synapse=None,
# 				transform=1)
# 			nengo.Connection(inter, bio[:dim],
# 				seed=conn3_seed,
# 				synapse=tau,
# 				transform=w_train)
# 			conn_lif = nengo.Connection(lif, temp,
# 				synapse=tau,
# 				solver=nengo.solvers.LstsqL2(reg=reg))

# 			probe_stim = nengo.Probe(stim, synapse=None)
# 			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
# 			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
# 			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
# 			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)


# 		"""
# 		Simulate the network, collect bioneuron activities and target values,
# 		and apply the oracle method to calculate readout decoders
# 		"""
# 		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
# 			sim.run(t_final)
# 		lpf = nengo.Lowpass(tau_readout)
# 		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
# 		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
# 		# bio readout is always "oracle" for the oracle method training
# 		if readout_LIF == 'LIF':
# 			d_readout_lif_new = sim.data[conn_lif].weights.T
# 		elif readout_LIF == 'oracle':
# 			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]
# 		d_recurrent_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
# 		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
# 		if jl_dims > 0:
# 			d_recurrent_bio_new = np.hstack((d_recurrent_bio_new, d_JL))
# 			d_readout_bio_new = np.hstack((d_readout_bio_new, d_JL))

# 		"""
# 		Use the old readout decoders to estimate the bioneurons' outputs for plotting
# 		"""
# 		x_target = sim.data[probe_oracle]
# 		xhat_bio = np.dot(act_bio, d_readout_bio)
# 		xhat_lif = np.dot(act_lif, d_readout_lif)
# 		rmse_bio = rmse(x_target[:,0], xhat_bio[:,0])
# 		rmse_lif = rmse(x_target[:,0], xhat_lif[:,0])

# 		if plot:
# 			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
# 			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
# 			plt.plot(sim.trange(), x_target[:,0], label='oracle')
# 			if jl_dims > 0:
# 				plt.plot(sim.trange(), xhat_bio[:,1:], label='jm_dims')
# 			plt.xlabel('time (s)')
# 			plt.ylabel('$\hat{x}(t)$')
# 			plt.legend()

# 		return d_recurrent_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio


# 	"""
# 	Run the test
# 	"""
# 	jl_rng = np.random.RandomState(seed=conn_seed)
# 	d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
# 	d_recurrent_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
# 	d_readout_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
# 	d_readout_lif_init = np.zeros((bio_neurons, dim))

# 	d_recurrent_bio_new, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
# 		d_recurrent_bio=d_recurrent_bio_init,
# 		d_readout_bio=d_readout_bio_init,
# 		d_readout_lif=d_readout_lif_init,
# 		d_JL=d_JL,
# 		w_train=1.0,
# 		signal=signal_train,
# 		freq=freq_train,
# 		seeds=seed_train,
# 		transform=transform_train,
# 		t_final=t_train,
# 		plot=False)
# 	d_recurrent_bio_extra, d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
# 		d_recurrent_bio=d_recurrent_bio_new,
# 		d_readout_bio=d_readout_bio_extra,
# 		d_readout_lif=d_readout_lif_extra,
# 		d_JL=d_JL,
# 		w_train=0.0,
# 		signal=signal_train,
# 		freq=freq_train,
# 		seeds=seed_train,
# 		transform=transform_train,
# 		t_final=t_train,
# 		plot=False)
# 	d_recurrent_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
# 		d_recurrent_bio=d_recurrent_bio_new,
# 		d_readout_bio=d_readout_bio_new,
# 		d_readout_lif=d_readout_lif_new,
# 		d_JL=d_JL,
# 		w_train=0.0,
# 		signal=signal_test,
# 		freq=freq_test,
# 		seeds=seed_test,
# 		transform=transform_test,
# 		t_final=t_test,
# 		plot=True)



def test_integrator_deriv_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	tau_decoders = 0.1
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
	conn2_seed = 8
	conn3_seed = 8

	max_freq = 5
	rms = 0.25
	t_transient = 0.0

	signal_train = 'white_noise'
	freq_train = 1
	seed_train = 3
	t_train = 10.0

	signal_test = 'white_noise'
	freq_test = 1
	seed_test = 1
	t_test = 1.0

	dim = 1
	reg = 0.01
	t_final = 1.0
	cutoff = 0.1
	# jl_dim_mag = 1e-1

	def sim(
		d_recurrent_bio,
		d_readout_bio,
		d_readout_lif,
		tau_decoders,
		w_train,
		readout_LIF = 'LIF',
		signal_type='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		plot=False):

		"""
		Load the recurrent decoders,
		scaled by the training factor, w_train. w_train==1 means only oracle
		spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
		and intermediate values are a weighted mix.
		"""
		d_recurrent_bio[:,:dim] *= (1.0 - w_train)

		stimulus, derivative = get_signal(
			signal_type, network_seed, sim_seed, freq, seeds, t_transient, t_final, max_freq, rms, tau, dt)
		lpf_signals = nengo.Lowpass(tau)
		stim_trans = 1.0 / max(abs(stimulus))
		deriv_trans = 1.0 / max(abs(lpf_signals.filt(derivative, dt=dt)))
		# stim_trans2 = 1.0 / max(abs(lpf_signals.filt(stim_trans*stimulus, dt=dt)))
		# deriv_trans2 = 1.0 / max(abs(lpf_signals.filt(deriv_trans*derivative, dt=dt)))

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			stim = nengo.Node(lambda t: stimulus[int(t/dt)])
			deriv = nengo.Node(lambda t: derivative[int(t/dt)])
			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre')
			pre_deriv = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre_deriv')
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim+1,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio')
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
			oracle = nengo.Node(size_in=dim, label='oracle')
			temp = nengo.Node(size_in=dim, label='temp')

			recurrent_solver = OracleSolver(decoders_bio = d_recurrent_bio)

			# feedforward connections
			nengo.Connection(stim, pre,
				synapse=None,
				transform=stim_trans)
			nengo.Connection(deriv, pre_deriv,
				synapse=None,
				transform=deriv_trans)
			nengo.Connection(stim, oracle,
				synapse=1/s,
				transform=stim_trans)
			''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
			nengo.Connection(pre, bio[0],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=tau)
			nengo.Connection(pre_deriv, bio[1],
				weights_bias_conn=False,
				seed=2*conn_seed,
				synapse=tau,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=tau)
			''' Connect recurrent (spikes) feedback to all dims of bio '''
			nengo.Connection(bio[0], bio[0],  # full recurrence?
				seed=conn2_seed,
				synapse=tau,
				solver=recurrent_solver)
			nengo.Connection(lif, lif,
				synapse=tau)
			# nengo.Connection(stim, oracle,
			# 	synapse=1/s,
			# 	transform=transform)
			nengo.Connection(oracle, inter,
				seed=conn3_seed,
				synapse=None,
				transform=1)
			nengo.Connection(inter, bio[0],
				seed=conn3_seed,
				synapse=tau,
				transform=w_train)
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_oracle_decoders = nengo.Probe(oracle, synapse=tau_decoders)


		"""
		Simulate the network, collect bioneuron activities and target values,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)
		lpf = nengo.Lowpass(tau_readout)
		lpf_decoders = nengo.Lowpass(tau_decoders)
		act_bio = lpf.filt(sim.data[probe_bio_spikes][int(t_transient/dt):], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes][int(t_transient/dt):], dt=dt)
		act_bio_decoders = lpf_decoders.filt(sim.data[probe_bio_spikes][int(t_transient/dt):], dt=dt)
		act_lif_decoders = lpf_decoders.filt(sim.data[probe_lif_spikes][int(t_transient/dt):], dt=dt)
		# bio readout is always "oracle" for the oracle method training
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle][int(t_transient/dt):])[0]
			# d_readout_lif_new = nengo.solvers.Lstsq()(act_lif_decoders, sim.data[probe_oracle_decoders][int(t_transient/dt):])[0]
		d_recurrent_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle][int(t_transient/dt):])[0]
		# d_recurrent_bio_new = nengo.solvers.Lstsq()(act_bio_decoders, sim.data[probe_oracle_decoders][int(t_transient/dt):])[0]
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle][int(t_transient/dt):])[0]
		# d_readout_bio_new = nengo.solvers.Lstsq()(act_bio_decoders, sim.data[probe_oracle_decoders][int(t_transient/dt):])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][int(t_transient/dt):]
		xhat_bio = np.dot(act_bio, d_readout_bio)
		xhat_lif = np.dot(act_lif, d_readout_lif)
		rmse_bio = rmse(x_target[:,0], xhat_bio[:,0])
		rmse_lif = rmse(x_target[:,0], xhat_lif[:,0])

		if plot:
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange()[int(t_transient/dt):], x_target[:,0], label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_recurrent_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio


	"""
	Run the test
	"""
	d_recurrent_bio_init = np.zeros((bio_neurons, dim))
	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_recurrent_bio_new, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_init,
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		tau_decoders=tau_decoders,
		w_train=1.0,
		signal_type=signal_train,
		freq=freq_train,
		seeds=seed_train,
		t_final=t_train,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_extra,
		d_readout_lif=d_readout_lif_extra,
		tau_decoders=tau_decoders,
		w_train=0.0,
		signal_type=signal_train,
		freq=freq_train,
		seeds=seed_train,
		t_final=t_train,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		tau_decoders=tau_decoders,
		w_train=0.0,
		signal_type=signal_test,
		freq=freq_test,
		seeds=seed_test,
		t_final=t_test,
		plot=True)


def test_integrator_extra_1d(Simulator, plt):
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
	conn2_seed = 8
	conn3_seed = 8

	max_freq = 5
	rms = 0.25
	n_steps = 10

	signal_train = 'white_noise'
	freq_train = 1
	seed_train = 3
	transform_train = 5.0
	t_train = 1.0

	signal_test = 'white_noise'
	freq_test = 1
	seed_test = 1
	transform_test = 5.0
	t_test = 1.0

	dim = 1
	reg = 0.01
	t_final = 1.0
	cutoff = 0.1

	def sim(
		d_recurrent_bio,
		d_readout_bio,
		d_readout_lif,
		w_train,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1,
		plot=False):

		"""
		Load the recurrent decoders,
		scaled by the training factor, w_train. w_train==1 means only oracle
		spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
		and intermediate values are a weighted mix.
		"""
		d_recurrent_bio[:,:dim] *= (1.0 - w_train)
		# extra_trans = get_stim_extra(
		# 	signal, network_seed, sim_seed, freq, seeds, t_final, max_freq, rms, tau, dt)
		deriv_trans = get_stim_deriv(
			signal, 1, network_seed, sim_seed, freq, seeds, t_final, max_freq, rms, tau, dt)
		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq * t),
					label='stim')
			elif signal == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds),
					label='stim')
			elif signal == 'step':
				stim = nengo.Node(lambda t:
					np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
			elif signal == 'constant':
				stim = nengo.Node(lambda t: freq)

			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre')
			pre_extra = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre_extra')
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim+1,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio')
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
			oracle = nengo.Node(size_in=dim, label='oracle')
			temp = nengo.Node(size_in=dim, label='temp')

			recurrent_solver = OracleSolver(decoders_bio = d_recurrent_bio)

			nengo.Connection(stim, pre,
				synapse=None)
			# nengo.Connection(stim, pre_extra, synapse=(tau*(1.0 - ~z) / dt) + 1)
			nengo.Connection(stim, pre_extra, synapse=((1.0 - ~z) / dt))
			''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
			nengo.Connection(pre, bio[0],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform*tau)
			nengo.Connection(pre_extra, bio[1],
				weights_bias_conn=False,
				seed=2*conn_seed,
				synapse=tau,
				transform=deriv_trans*transform,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform*tau)
			''' Connect recurrent (spikes) feedback to all dims of bio '''
			nengo.Connection(bio[0], bio[0],  # full recurrence?
				seed=conn2_seed,
				synapse=tau,
				solver=recurrent_solver)
			nengo.Connection(lif, lif,
				synapse=tau)
			nengo.Connection(stim, oracle,
				synapse=1/s,
				transform=transform)
			nengo.Connection(oracle, inter,
				seed=conn3_seed,
				synapse=None,
				transform=1)
			nengo.Connection(inter, bio[0],
				seed=conn3_seed,
				synapse=tau,
				transform=w_train)
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
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]
		d_recurrent_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle]
		xhat_bio = np.dot(act_bio, d_readout_bio)
		xhat_lif = np.dot(act_lif, d_readout_lif)
		rmse_bio = rmse(x_target[:,0], xhat_bio[:,0])
		rmse_lif = rmse(x_target[:,0], xhat_lif[:,0])

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target[:,0], label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_recurrent_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio


	"""
	Run the test
	"""
	d_recurrent_bio_init = np.zeros((bio_neurons, dim))
	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_recurrent_bio_new, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_init,
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		w_train=1.0,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_extra,
		d_readout_lif=d_readout_lif_extra,
		w_train=0.0,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		w_train=0.0,
		signal=signal_test,
		freq=freq_test,
		seeds=seed_test,
		transform=transform_test,
		t_final=t_test,
		plot=True)


def test_integrator_deriv_recurrent_1d(Simulator, plt):
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
	conn2_seed = 8
	conn3_seed = 8

	max_freq = 5
	rms = 0.25
	n_steps = 10

	signal_train = 'white_noise'
	freq_train = 1
	seed_train = 3
	transform_train = 5.0
	t_train = 1.0

	signal_test = 'white_noise'
	freq_test = 1
	seed_test = 1
	transform_test = 5.0
	t_test = 1.0

	dim = 1
	reg = 0.01
	t_final = 1.0
	cutoff = 0.1

	def sim(
		d_recurrent_bio,
		d_derivative_bio,
		d_readout_bio,
		d_readout_lif,
		w_train,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1,
		plot=False):

		"""
		Load the recurrent decoders,
		scaled by the training factor, w_train. w_train==1 means only oracle
		spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
		and intermediate values are a weighted mix.
		"""
		d_recurrent_bio[:,:dim] *= (1.0 - w_train)
		d_derivative_bio[:,:dim] *= (1.0 - w_train)
		deriv_trans = get_stim_deriv(
			signal, 1, network_seed, sim_seed, freq, seeds, t_final, max_freq, rms, tau, dt)

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq * t),
					label='stim')
			elif signal == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds),
					label='stim')
			elif signal == 'step':
				stim = nengo.Node(lambda t:
					np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
			elif signal == 'constant':
				stim = nengo.Node(lambda t: freq)

			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre')
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim+1,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio')
			inter = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim+1,
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
			oracle = nengo.Node(size_in=dim, label='oracle')
			oracle_deriv = nengo.Node(size_in=1, label='oracle_deriv')
			temp = nengo.Node(size_in=dim, label='temp')

			recurrent_solver = OracleSolver(decoders_bio = d_recurrent_bio)
			derivative_solver = OracleSolver(decoders_bio = d_derivative_bio)

			''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
			nengo.Connection(stim, pre,
				synapse=None)
			nengo.Connection(pre, bio[0],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform*tau)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform*tau)

			''' Connect recurrent (spikes) feedback to all dims of bio '''
			nengo.Connection(bio[0], bio[0],
				seed=conn2_seed,
				synapse=tau,
				solver=recurrent_solver)
			nengo.Connection(bio[0], bio[1],
				seed=2*conn2_seed,
				synapse=tau,
				solver=derivative_solver)
			nengo.Connection(lif, lif,
				synapse=tau)
			nengo.Connection(stim, oracle,
				synapse=1/s,
				transform=transform)
			nengo.Connection(oracle, oracle_deriv,
				synapse=(1.0 - ~z) / dt,
				transform=deriv_trans)	# transform doesn't account for 2nd filter?	

			''' Connect training spikes back to bio '''
			nengo.Connection(oracle, inter[0],
				seed=conn3_seed,
				synapse=None,
				transform=1)
			nengo.Connection(inter[0], bio[0],
				seed=conn3_seed,
				synapse=tau,
				transform=w_train)
			nengo.Connection(oracle_deriv, inter[1],
				seed=2*conn3_seed,
				synapse=None,
				transform=1)
			nengo.Connection(inter[1], bio[1],
				seed=2*conn3_seed,
				synapse=tau,
				transform=w_train)
	
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_oracle_deriv = nengo.Probe(oracle_deriv, synapse=tau_readout)


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
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]
		d_recurrent_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_derivative_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle_deriv])[0]
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle]
		xhat_bio = np.dot(act_bio, d_readout_bio)
		xhat_lif = np.dot(act_lif, d_readout_lif)
		rmse_bio = rmse(x_target[:,0], xhat_bio[:,0])
		rmse_lif = rmse(x_target[:,0], xhat_lif[:,0])

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target[:,0], label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_recurrent_bio_new, d_derivative_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio


	"""
	Run the test
	"""
	d_recurrent_bio_init = np.zeros((bio_neurons, dim))
	d_derivative_bio_init = np.zeros((bio_neurons, 1))
	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_recurrent_bio_new, d_derivative_bio_new, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_init,
		d_derivative_bio=d_derivative_bio_init,
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		w_train=1.0,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		plot=False)
	d_recurrent_bio_extra, d_derivative_bio_extra, d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_derivative_bio=d_derivative_bio_new,
		d_readout_bio=d_readout_bio_extra,
		d_readout_lif=d_readout_lif_extra,
		w_train=0.0,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		plot=False)
	d_recurrent_bio_extra, d_derivative_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_derivative_bio=d_derivative_bio_new,
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		w_train=0.0,
		signal=signal_test,
		freq=freq_test,
		seeds=seed_test,
		transform=transform_test,
		t_final=t_test,
		plot=True)



# def test_integrator_2d(Simulator, plt):
# 	# Nengo Parameters
# 	pre_neurons = 100
# 	bio_neurons = 100
# 	tau = 0.1
# 	tau_readout = 0.1
# 	dt = 0.001
# 	min_rate = 150
# 	max_rate = 200
# 	radius = 1
# 	bio_radius = 0.2
# 	n_syn = 1

# 	pre_seed = 1
# 	bio_seed = 2
# 	conn_seed = 3
# 	network_seed = 4
# 	sim_seed = 5
# 	post_seed = 6
# 	inter_seed = 7
# 	conn2_seed = 9

# 	max_freq = 5
# 	rms = 0.25
# 	n_steps = 10

# 	signal_train = 'white_noise'
# 	freq_train = [1,2]
# 	seed_train = [1,2]
# 	transform_train = 10.0
# 	t_train = 1.0

# 	signal_test = 'white_noise'
# 	freq_test = [2,1]
# 	seed_test = [3,4]
# 	transform_test = 10.0
# 	t_test = 1.0

# 	dim = 2
# 	reg = 0.1
# 	cutoff = 0.1
# 	jl_dims = 3
# 	jl_dim_mag = 1e-4

# 	def sim(
# 		d_recurrent_bio,
# 		d_readout_bio,
# 		d_readout_lif,
# 		d_JL,
# 		w_train,
# 		readout_LIF = 'LIF',
# 		signal='sinusoids',
# 		t_final=1.0,
# 		freq=1,
# 		seeds=1,
# 		transform=1,
# 		plot=False):

# 		"""
# 		Load the recurrent decoders, with the non-JL dimensions,
# 		scaled by the training factor, w_train. w_train==1 means only oracle
# 		spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
# 		and intermediate values are a weighted mix.
# 		"""
# 		d_recurrent_bio[:dim] *= (1.0 - w_train)

# 		"""
# 		Define the network
# 		"""
# 		with nengo.Network(seed=network_seed) as network:

# 			if signal == 'sinusoids':
# 				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq[0] * t),
# 					label='stim')
# 				stim2 = nengo.Node(lambda t: np.cos(2 * np.pi * freq[1] * t),
# 					label='stim')
# 			elif signal == 'white_noise':
# 				stim = nengo.Node(nengo.processes.WhiteSignal(
# 					period=t_final, high=max_freq, rms=rms, seed=seeds[0]),
# 					label='stim')
# 				stim2 = nengo.Node(nengo.processes.WhiteSignal(
# 					period=t_final, high=max_freq, rms=rms, seed=seeds[1]),
# 					label='stim')
# 			elif signal == 'step':
# 				stim = nengo.Node(lambda t:
# 					np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
# 				stim2 = nengo.Node(lambda t:
# 					np.linspace(freq, -freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
# 			elif signal == 'constant':
# 				stim = nengo.Node(lambda t: freq[0])
# 				stim2 = nengo.Node(lambda t: freq[1])

# 			pre = nengo.Ensemble(
# 				n_neurons=pre_neurons,
# 				dimensions=dim,
# 				seed=pre_seed,
# 				neuron_type=nengo.LIF(),
# 				radius=radius,
# 				label='pre')
# 			bio = nengo.Ensemble(
# 				n_neurons=bio_neurons,
# 				dimensions=dim+jl_dims,
# 				seed=bio_seed,
# 				neuron_type=BahlNeuron(),
# 				# neuron_type=nengo.LIF(),
# 				radius=bio_radius,
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				label='bio')
# 			inter = nengo.Ensemble(
# 				n_neurons=bio_neurons,
# 				dimensions=dim,
# 				seed=bio_seed,
# 				neuron_type=nengo.LIF(),
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				# radius=radius,
# 				label='inter')
# 			lif = nengo.Ensemble(
# 				n_neurons=bio.n_neurons,
# 				dimensions=dim,
# 				seed=bio.seed,
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				radius=bio.radius,
# 				neuron_type=nengo.LIF(),
# 				label='lif')
# 			oracle = nengo.Node(size_in=dim, label='oracle')
# 			temp = nengo.Node(size_in=dim, label='temp')

# 			recurrent_solver = OracleSolver(decoders_bio = d_recurrent_bio)

# 			nengo.Connection(stim, pre[0],
# 				synapse=None)
# 			nengo.Connection(stim2, pre[1],
# 				synapse=None)
# 			''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
# 			nengo.Connection(pre, bio[:dim],
# 				weights_bias_conn=True,
# 				seed=conn_seed,
# 				synapse=tau,
# 				transform=transform*tau)
# 			nengo.Connection(pre, lif,
# 				synapse=tau,
# 				transform=transform*tau)
# 			''' Connect recurrent (spikes) feedback to all dims of bio '''
# 			nengo.Connection(bio, bio,
# 				seed=conn2_seed,
# 				synapse=tau,
# 				solver=recurrent_solver)
# 			nengo.Connection(lif, lif,
# 				synapse=tau)
# 			nengo.Connection(stim, oracle[0],
# 				synapse=1/s,
# 				transform=transform)
# 			nengo.Connection(stim2, oracle[1],
# 				synapse=1/s,
# 				transform=transform)
# 			nengo.Connection(oracle, inter,
# 				synapse=None,
# 				transform=1)
# 			nengo.Connection(inter, bio[:dim],
# 				synapse=tau,
# 				transform=w_train)
# 			conn_lif = nengo.Connection(lif, temp,
# 				synapse=tau,
# 				solver=nengo.solvers.LstsqL2(reg=reg))

# 			probe_stim = nengo.Probe(stim, synapse=None)
# 			probe_stim2 = nengo.Probe(stim2, synapse=None)
# 			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
# 			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
# 			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
# 			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)


# 		"""
# 		Simulate the network, collect bioneuron activities and target values,
# 		and apply the oracle method to calculate readout decoders
# 		"""
# 		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
# 			sim.run(t_final)
# 		lpf = nengo.Lowpass(tau_readout)
# 		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
# 		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
# 		# bio readout is always "oracle" for the oracle method training
# 		if readout_LIF == 'LIF':
# 			d_readout_lif_new = sim.data[conn_lif].weights.T
# 		elif readout_LIF == 'oracle':
# 			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]
# 		d_recurrent_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
# 		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
# 		if jl_dims > 0:
# 			d_recurrent_bio_new = np.hstack((d_recurrent_bio_new, d_JL))
# 			d_readout_bio_new = np.hstack((d_readout_bio_new, d_JL))

# 		"""
# 		Use the old readout decoders to estimate the bioneurons' outputs for plotting
# 		"""
# 		x_target = sim.data[probe_oracle]
# 		xhat_bio = np.dot(act_bio, d_readout_bio)
# 		xhat_lif = np.dot(act_lif, d_readout_lif)
# 		rmse_bio_dim_1 = rmse(x_target[:,0], xhat_bio[:,0])
# 		rmse_lif_dim_1 = rmse(x_target[:,0], xhat_lif[:,0])
# 		rmse_bio_dim_2 = rmse(x_target[:,1], xhat_bio[:,1])
# 		rmse_lif_dim_2 = rmse(x_target[:,1], xhat_lif[:,1])

# 		if plot:
# 			plt.plot(sim.trange(), xhat_bio[:,0], label='bio dim 1, rmse=%.5f' % rmse_bio_dim_1)
# 			plt.plot(sim.trange(), xhat_lif[:,0], label='lif dim 1, rmse=%.5f' % rmse_lif_dim_1)
# 			plt.plot(sim.trange(), x_target[:,0], label='oracle dim 1')
# 			plt.plot(sim.trange(), xhat_bio[:,1], label='bio dim 2, rmse=%.5f' % rmse_bio_dim_2)
# 			plt.plot(sim.trange(), xhat_lif[:,1], label='lif dim 2, rmse=%.5f' % rmse_lif_dim_2)
# 			plt.plot(sim.trange(), x_target[:,1], label='oracle dim 2')
# 			if jl_dims > 0:
# 				plt.plot(sim.trange(), xhat_bio[:,2:], label='jm_dims')
# 			plt.xlabel('time (s)')
# 			plt.ylabel('$\hat{x}(t)$')
# 			plt.legend()

# 		return d_recurrent_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio_dim_1, rmse_bio_dim_2


# 	"""
# 	Run the test
# 	"""
# 	jl_rng = np.random.RandomState(seed=conn_seed)
# 	d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
# 	d_recurrent_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
# 	d_readout_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
# 	d_readout_lif_init = np.zeros((bio_neurons, dim))

# 	d_recurrent_bio_new, d_readout_bio_extra, d_readout_lif_extra, rmse_bio_dim_1, rmse_bio_dim_2 = sim(
# 		d_recurrent_bio=d_recurrent_bio_init,
# 		d_readout_bio=d_readout_bio_init,
# 		d_readout_lif=d_readout_lif_init,
# 		d_JL=d_JL,
# 		w_train=1.0,
# 		signal=signal_train,
# 		freq=freq_train,
# 		seeds=seed_train,
# 		transform=transform_train,
# 		t_final=t_train,
# 		plot=False)
# 	d_recurrent_bio_extra, d_readout_bio_new, d_readout_lif_new, rmse_bio_dim_1, rmse_bio_dim_2 = sim(
# 		d_recurrent_bio=d_recurrent_bio_new,
# 		d_readout_bio=d_readout_bio_extra,
# 		d_readout_lif=d_readout_lif_extra,
# 		d_JL=d_JL,
# 		w_train=0.0,
# 		signal=signal_train,
# 		freq=freq_train,
# 		seeds=seed_train,
# 		transform=transform_train,
# 		t_final=t_train,
# 		plot=False)
# 	d_recurrent_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio_dim_1, rmse_bio_dim_2 = sim(
# 		d_recurrent_bio=d_recurrent_bio_new,
# 		d_readout_bio=d_readout_bio_new,
# 		d_readout_lif=d_readout_lif_new,
# 		d_JL=d_JL,
# 		w_train=0.0,
# 		signal=signal_test,
# 		freq=freq_test,
# 		seeds=seed_test,
# 		transform=transform_test,
# 		t_final=t_test,
# 		plot=True)

# 	assert rmse_bio_dim_1 < cutoff
# 	assert rmse_bio_dim_2 < cutoff