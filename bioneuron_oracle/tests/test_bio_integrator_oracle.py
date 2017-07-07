import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver
from nengolib.signal import s

def test_bio_integrator_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
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
	rms = 0.5

	dim = 1
	jl_dims = 3
	jl_dim_mag = 3e-4
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1

	def sim(
		d_recurrent,
		d_readout,
		d_JL,
		w_train,
		t_final=1.0,
		signal='sinusoids',
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
		d_recurrent[:dim] *= (1.0 - w_train)

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2*np.pi*freq*t))
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
				dimensions=dim+jl_dims,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				# max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio')
			inter = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				seed=bio_seed,
				neuron_type=nengo.LIF(),
				max_rates=bio.max_rates,
				# intercepts=nengo.dists.Uniform(-intercept, intercept),
				# radius=radius,
				label='inter')
			lif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=dim,
				seed=bio.seed,
				neuron_type=nengo.LIF(),
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# intercepts=nengo.dists.Uniform(-intercept, intercept),
				radius=radius,
				label='lif')
			oracle = nengo.Node(size_in=dim)

			recurrent_solver = OracleSolver(decoders_bio = d_recurrent)

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

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)


		"""
		Simulate the network, collect bioneuron activities and target values,
		and apply the oracle method to calculate recurrent and readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		d_recurrent_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		if jl_dims > 0:
			d_recurrent_new = np.hstack((d_recurrent_new, d_JL))
			d_readout_new = np.hstack((d_readout_new, d_JL))


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		# xhat_bio = np.dot(act_bio, d_readout_new)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle][:,0], xhat_lif[:,0])

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			if jl_dims > 0:
				plt.plot(sim.trange(), xhat_bio[:,1:], label='jm_dims')
			plt.plot(sim.trange(), sim.data[probe_oracle][:,0], label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_recurrent_new, d_readout_new, rmse_bio


	jl_rng = np.random.RandomState(seed=conn_seed)
	d_JL = jl_rng.randn(bio_neurons, jl_dims) * jl_dim_mag
	d_recurrent_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))
	d_readout_init = np.hstack((np.zeros((bio_neurons, dim)), d_JL))

	d_recurrent_new, d_readout_extra, rmse_bio = sim(
		d_recurrent=d_recurrent_init,
		d_readout=d_readout_init,
		d_JL=d_JL,
		w_train=1.0,
		signal='sinusoids',
		freq=2,
		seeds=2,
		transform=2*np.pi,
		t_final=t_final)
	d_recurrent_extra, d_readout_new, rmse_bio = sim(
		d_recurrent=d_recurrent_new,
		d_readout=d_readout_init,
		d_JL=d_JL,
		w_train=0.0,
		signal='sinusoids',
		freq=2,
		seeds=2,
		transform=2*np.pi,
		t_final=t_final)
	d_recurrent_extra, d_readout_extra, rmse_bio = sim(
		d_recurrent=d_recurrent_new,
		d_readout=d_readout_new,
		d_JL=d_JL,
		w_train=0.0,
		signal='sinusoids',
		freq=1,
		seeds=1,
		transform=2*np.pi,
		t_final=t_final,
		plot=True)

	assert rmse_bio < cutoff