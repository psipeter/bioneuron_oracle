import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, get_signal
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
	t_transient = 1.0

	signal_train = 'white_noise'
	freq_train = [1, 1]
	seed_train = [1, 3]
	t_train = 10.0

	signal_test = 'white_noise'
	freq_test = [1, 1]
	seed_test = [1, 3]
	t_test = 1.0

	dim = 1
	reg = 0.01
	t_final = 1.0
	cutoff = 0.1

	def sim(
		d_readout_bio,
		d_readout_lif,
		readout_LIF = 'LIF',
		signal_type='sinusoids',
		t_final=1.0,
		freq=[1,1],
		seeds=[1,1],
		plot=False):

		stimulus, derivative = get_signal(
			signal_type, network_seed, sim_seed, freq[0], seeds[0], t_transient, t_final, max_freq, rms, tau, dt)
		lpf_signals = nengo.Lowpass(tau)
		stim_trans = 1.0 / max(abs(stimulus))
		deriv_trans = 1.0 / max(abs(lpf_signals.filt(derivative, dt=dt)))
		stim_trans2 = 1.0 / max(abs(lpf_signals.filt(stim_trans*stimulus, dt=dt)))
		deriv_trans2 = 1.0 / max(abs(lpf_signals.filt(deriv_trans*derivative, dt=dt)))

		stimulus2, derivative2 = get_signal(
			signal_type, network_seed, sim_seed, freq[1], seeds[1], t_transient, t_final, max_freq, rms, tau, dt)
		lpf_signals = nengo.Lowpass(tau)
		stim_trans_2 = 1.0 / max(abs(stimulus2))
		deriv_trans_2 = 1.0 / max(abs(lpf_signals.filt(derivative2, dt=dt)))
		stim_trans2_2 = 1.0 / max(abs(lpf_signals.filt(stim_trans_2*stimulus2, dt=dt)))
		deriv_trans2_2 = 1.0 / max(abs(lpf_signals.filt(deriv_trans_2*derivative2, dt=dt)))

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			stim = nengo.Node(lambda t: stimulus[int(t/dt)])
			deriv = nengo.Node(lambda t: derivative[int(t/dt)])
			stim2 = nengo.Node(lambda t: stimulus2[int(t/dt)])
			deriv2 = nengo.Node(lambda t: derivative2[int(t/dt)])
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
			pre_deriv2 = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				seed=2*pre_seed,
				neuron_type=nengo.LIF(),
				radius=radius,
				label='pre_deriv2')
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

			nengo.Connection(stim, pre,
				synapse=None,
				transform=stim_trans)
			nengo.Connection(stim2, pre2,
				synapse=None,
				transform=stim_trans_2)
			nengo.Connection(deriv, pre_deriv,
				synapse=tau,
				transform=deriv_trans)
			nengo.Connection(deriv2, pre_deriv2,
				synapse=tau,
				transform=deriv_trans_2)
			pre_bio = nengo.Connection(pre, bio[0],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=stim_trans2,
				n_syn=n_syn)
			pre2_bio = nengo.Connection(pre2, bio[0],
				weights_bias_conn=False,
				seed=conn2_seed,
				synapse=tau,
				transform=stim_trans2_2,
				n_syn=n_syn)
			nengo.Connection(pre_deriv, bio[1],
				weights_bias_conn=False,
				seed=2*conn_seed,
				synapse=tau,
				transform=deriv_trans2,
				n_syn=n_syn)
			nengo.Connection(pre_deriv2, bio[1],
				weights_bias_conn=False,
				seed=2*conn_seed,
				synapse=tau,
				transform=deriv_trans2_2,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=stim_trans2)
			nengo.Connection(pre2, lif,
				synapse=tau,
				transform=stim_trans2_2)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=stim_trans*stim_trans2)
			nengo.Connection(stim2, oracle,
				synapse=tau,
				transform=stim_trans_2*stim_trans2_2)
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
		act_bio = lpf.filt(sim.data[probe_bio_spikes][int(t_transient/dt):], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes][int(t_transient/dt):], dt=dt)
		# bio readout is always "oracle" for the oracle method training
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle][int(t_transient/dt):])[0]
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle][int(t_transient/dt):])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][int(t_transient/dt):,0]
		xhat_bio = np.dot(act_bio, d_readout_bio)[:,0]
		xhat_lif = np.dot(act_lif, d_readout_lif)[:,0]
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)

		if plot:
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange()[int(t_transient/dt):], x_target, label='oracle')
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
		signal_type=signal_train,
		freq=freq_train,
		seeds=seed_train,
		t_final=t_train,
		plot=False)
	d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		signal_type=signal_test,
		freq=freq_test,
		seeds=seed_test,
		t_final=t_test,
		plot=True)

	assert rmse_bio < cutoff