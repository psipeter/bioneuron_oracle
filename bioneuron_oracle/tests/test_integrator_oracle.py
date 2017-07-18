
import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver
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
	rms = 0.25
	n_steps = 10
	freq_train = 1
	freq_test = 2
	seed_train = 1
	seed_test = 3

	dim = 1
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1
	transform = 1
	jl_dims = 0
	jl_dim_mag = 3e-5

	def sim(
		d_recurrent_bio,
		d_readout_bio,
		d_readout_lif,
		d_JL,
		w_train,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1,
		plot=False):

		"""
		Load the recurrent decoders, with the non-JL dimensions,
		scaled by the training factor, w_train. w_train==1 means only oracle
		spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
		and intermediate values are a weighted mix.
		"""
		d_recurrent_bio[:dim] *= (1.0 - w_train)

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
				dimensions=dim+jl_dims,
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
			''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
			nengo.Connection(pre, bio[:dim],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform*tau)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform*tau)
			''' Connect recurrent (spikes) feedback to all dims of bio '''
			nengo.Connection(bio, bio,
				seed=conn2_seed,
				synapse=tau,
				solver=recurrent_solver)
			nengo.Connection(lif, lif,
				synapse=tau)
			nengo.Connection(stim, oracle,
				synapse=1/s,
				transform=transform)
			nengo.Connection(oracle, inter,
				synapse=None,
				transform=1)
			nengo.Connection(inter, bio[:dim],
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
		if jl_dims > 0:
			d_recurrent_bio_new = np.hstack((d_recurrent_bio_new, d_JL))
			d_readout_bio_new = np.hstack((d_readout_bio_new, d_JL))

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle]
		xhat_bio = np.dot(act_bio, d_readout_bio)
		xhat_lif = np.dot(act_lif, d_readout_lif)
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target[:,0], label='oracle')
			if jl_dims > 0:
				plt.plot(sim.trange(), xhat_bio[:,1:], label='jm_dims')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_recurrent_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio


	"""
	Run the test
	"""
	jl_rng = np.random.RandomState(seed=conn_seed)
	d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
	d_recurrent_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
	d_readout_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_recurrent_bio_new, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_init,
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		d_JL=d_JL,
		w_train=1.0,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_extra,
		d_readout_lif=d_readout_lif_extra,
		d_JL=d_JL,
		w_train=1.0,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		d_JL=d_JL,
		w_train=1.0,
		signal='white_noise',
		freq=freq_test,
		seeds=seed_test,
		transform=5*transform,
		t_final=t_final,
		plot=True)


def test_integrator_2d(Simulator, plt):
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
	conn2_seed = 9

	max_freq = 5
	rms = 0.25
	n_steps = 10
	freq_train = [1, 2]
	freq_test = [2, 1]
	seed_train = [1, 3]
	seed_test = [3, 1]

	dim = 2
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1
	transform = 1
	jl_dims = 0
	jl_dim_mag = 3e-5

	def sim(
		d_recurrent_bio,
		d_readout_bio,
		d_readout_lif,
		d_JL,
		w_train,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1,
		plot=False):

		"""
		Load the recurrent decoders, with the non-JL dimensions,
		scaled by the training factor, w_train. w_train==1 means only oracle
		spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
		and intermediate values are a weighted mix.
		"""
		d_recurrent_bio[:dim] *= (1.0 - w_train)

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq[0] * t),
					label='stim')
				stim2 = nengo.Node(lambda t: np.cos(2 * np.pi * freq[1] * t),
					label='stim')
			elif signal == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds[0]),
					label='stim')
				stim2 = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds[1]),
					label='stim')
			elif signal == 'step':
				stim = nengo.Node(lambda t:
					np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
				stim2 = nengo.Node(lambda t:
					np.linspace(freq, -freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
			elif signal == 'constant':
				stim = nengo.Node(lambda t: freq[0])
				stim2 = nengo.Node(lambda t: freq[1])

			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre')
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim+jl_dims,
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
				# radius=0.2,
				neuron_type=nengo.LIF(),
				label='lif')
			oracle = nengo.Node(size_in=dim, label='oracle')
			temp = nengo.Node(size_in=dim, label='temp')

			recurrent_solver = OracleSolver(decoders_bio = d_recurrent_bio)

			nengo.Connection(stim, pre[0],
				synapse=None)
			nengo.Connection(stim2, pre[1],
				synapse=None)
			''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
			nengo.Connection(pre, bio[:dim],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform*tau)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform*tau)
			''' Connect recurrent (spikes) feedback to all dims of bio '''
			nengo.Connection(bio, bio,
				seed=conn2_seed,
				synapse=tau,
				solver=recurrent_solver)
			nengo.Connection(lif, lif,
				synapse=tau)
			nengo.Connection(stim, oracle[0],
				synapse=1/s,
				transform=transform)
			nengo.Connection(stim2, oracle[1],
				synapse=1/s,
				transform=transform)
			nengo.Connection(oracle, inter,
				synapse=None,
				transform=1)
			nengo.Connection(inter, bio[:dim],
				synapse=tau,
				transform=w_train)
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
		if jl_dims > 0:
			d_recurrent_bio_new = np.hstack((d_recurrent_bio_new, d_JL))
			d_readout_bio_new = np.hstack((d_readout_bio_new, d_JL))

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle]
		xhat_bio = np.dot(act_bio, d_readout_bio)
		xhat_lif = np.dot(act_lif, d_readout_lif)
		rmse_bio_dim_1 = rmse(x_target[:,0], xhat_bio[:,0])
		rmse_lif_dim_1 = rmse(x_target[:,0], xhat_lif[:,0])
		rmse_bio_dim_2 = rmse(x_target[:,1], xhat_bio[:,1])
		rmse_lif_dim_2 = rmse(x_target[:,1], xhat_lif[:,1])

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio dim 1, rmse=%.5f' % rmse_bio_dim_1)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif dim 1, rmse=%.5f' % rmse_lif_dim_1)
			plt.plot(sim.trange(), x_target[:,0], label='oracle dim 1')
			plt.plot(sim.trange(), xhat_bio[:,1], label='bio dim 2, rmse=%.5f' % rmse_bio_dim_2)
			plt.plot(sim.trange(), xhat_lif[:,1], label='lif dim 2, rmse=%.5f' % rmse_lif_dim_2)
			plt.plot(sim.trange(), x_target[:,1], label='oracle dim 2')
			if jl_dims > 0:
				plt.plot(sim.trange(), xhat_bio[:,2:], label='jm_dims')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_recurrent_bio_new, d_readout_bio_new, d_readout_lif_new, rmse_bio_dim_1, rmse_bio_dim_2


	"""
	Run the test
	"""
	jl_rng = np.random.RandomState(seed=conn_seed)
	d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
	d_recurrent_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
	d_readout_bio_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_recurrent_bio_new, d_readout_bio_extra, d_readout_lif_extra, rmse_bio_dim_1, rmse_bio_dim_2 = sim(
		d_recurrent_bio=d_recurrent_bio_init,
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		d_JL=d_JL,
		w_train=1.0,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_new, d_readout_lif_new, rmse_bio_dim_1, rmse_bio_dim_2 = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_extra,
		d_readout_lif=d_readout_lif_extra,
		d_JL=d_JL,
		w_train=1.0,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		plot=False)
	d_recurrent_bio_extra, d_readout_bio_extra, d_readout_lif_extra, rmse_bio_dim_1, rmse_bio_dim_2 = sim(
		d_recurrent_bio=d_recurrent_bio_new,
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		d_JL=d_JL,
		w_train=1.0,
		signal='white_noise',
		freq=freq_test,
		seeds=seed_test,
		transform=5*transform,
		t_final=t_final,
		plot=True)

	assert rmse_bio_dim_1 < cutoff
	assert rmse_bio_dim_2 < cutoff