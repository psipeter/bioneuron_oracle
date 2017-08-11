import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, get_stim_deriv
from nengolib.signal import s, z

def test_two_inputs_1d(Simulator, plt):
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
	pre2_seed = 8
	conn2_seed = 9

	max_freq = 5
	rms = 0.25
	signal_train = 'white_noise'
	freq_train = 10.0
	seed_train = [1, 3]
	transform_train = 5.0
	t_train = 1

	signal_test = 'white_noise'
	freq_test = 10.0
	seed_test = [1, 3]
	transform_test = 5.0
	t_test = 1

	dim = 1
	reg = 0.01
	t_final = 1.0
	cutoff = 0.1
	transform = 1.0

	def sim(
		d_readout_bio,
		d_readout_lif,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=[1,1],
		seeds=[1,1],
		transform=1.0,
		plot=False):

		deriv_trans = get_stim_deriv(
			signal, 1, network_seed, sim_seed, freq, seeds[0], t_final, max_freq, rms, tau, dt)

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq[0] * t))
				stim2 = nengo.Node(lambda t: np.cos(2 * np.pi * freq[1] * t))
			elif signal == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds[0]))
				stim2 = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds[1]))

			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre')
			pre2 = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=pre2_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre2')
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

			nengo.Connection(stim, pre, synapse=None)
			nengo.Connection(stim2, pre2, synapse=None)
			nengo.Connection(stim, pre_deriv, synapse=(1.0 - ~z) / dt)
			nengo.Connection(stim2, pre_deriv, synapse=(1.0 - ~z) / dt)
			pre_bio = nengo.Connection(pre, bio[0],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				n_syn=n_syn)
			pre2_bio = nengo.Connection(pre2, bio[0],
				weights_bias_conn=False,
				seed=conn2_seed,
				synapse=tau,
				transform=transform,
				n_syn=n_syn)
			nengo.Connection(pre_deriv, bio[1],
				weights_bias_conn=False,
				seed=2*conn_seed,
				synapse=tau,
				transform=0.5*deriv_trans*transform,  # approximate
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform)
			nengo.Connection(pre2, lif,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim2, oracle,
				synapse=tau,
				transform=transform)
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
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][:,0]
		xhat_bio = np.dot(act_bio, d_readout_bio)[:,0]
		xhat_lif = np.dot(act_lif, d_readout_lif)[:,0]
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)

		if plot:
			plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target, label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_readout_bio_new, d_readout_lif_new, rmse_bio


	"""
	Run the test
	"""
	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		plot=False)
	d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		signal=signal_test,
		freq=freq_test,
		seeds=seed_test,
		transform=transform_test,
		t_final=t_test,
		plot=True)

	assert rmse_bio < cutoff