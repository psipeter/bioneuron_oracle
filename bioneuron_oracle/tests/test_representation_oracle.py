import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver, get_signal
from nengolib.signal import z, s


def test_representation_1d(Simulator, plt):
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

	pre_seed = 1
	bio_seed = 2
	conn_seed = 3
	network_seed = 4
	sim_seed = 5
	post_seed = 6
	inter_seed = 7
	to_plot = 'signals'

	max_freq = 5
	rms = 0.25
	t_transient = 0.1

	signal_train = 'white_noise'
	freq_train = 10.0
	seed_train = 3
	t_train = 10.0

	signal_test = 'white_noise'
	freq_test = 10.0
	seed_test = 1
	t_test = 1.0


	def sim(
		d_readout_bio,
		d_readout_lif,
		d_readout_alif,
		t_final=1.0,
		tau_decoders=0.1,
		readout_LIF='LIF',
		signal_type='sinusoids',
		freq=1,
		seeds=1,
		plot=False):

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
				dimensions=dim+1,
				seed=bio.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				radius=bio.radius,
				neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05),
				label='alif')
			oracle = nengo.Node(size_in=dim)
			temp = nengo.Node(size_in=dim)

			# feedforward connections
			nengo.Connection(stim, pre,
				synapse=None,
				transform=stim_trans)
			nengo.Connection(deriv, pre_deriv,
				synapse=None,
				transform=deriv_trans)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=stim_trans)
				# transform=stim_trans*stim_trans2)
			nengo.Connection(pre, bio[0],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				# transform=stim_trans2,
				n_syn=n_syn)
			nengo.Connection(pre_deriv, bio[1],
				weights_bias_conn=False,
				seed=2*conn_seed,
				synapse=tau,
				# transform=deriv_trans2,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau)
				# transform=stim_trans2)
			nengo.Connection(pre, alif[0],
				synapse=tau)
				# transform=stim_trans2)
			nengo.Connection(pre_deriv, alif[1],
				synapse=tau)
				# transform=deriv_trans2)
			# temp connections to grab decoders
			conn_lif = nengo.Connection(lif, temp,
				synapse=None,
				solver=nengo.solvers.LstsqL2(reg=reg))
			conn_alif = nengo.Connection(alif[0], temp,
				synapse=None,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_deriv = nengo.Probe(deriv, synapse=None)
			probe_pre = nengo.Probe(pre, synapse=tau_readout)
			probe_pre_deriv = nengo.Probe(pre_deriv, synapse=tau_readout)
			# probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			# probe_alif = nengo.Probe(alif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_pre_spikes = nengo.Probe(pre.neurons, 'spikes')
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_alif_spikes = nengo.Probe(alif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_oracle_decoders = nengo.Probe(oracle, synapse=tau_decoders)


		"""
		Simulate the network, collect bioneuron activities and target values,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)

		# sim.data[probe_stim][:] = sim.data[probe_stim][int(t_transient/dt):]  # read-only

		lpf = nengo.Lowpass(tau_readout)
		lpf_decoders = nengo.Lowpass(tau_decoders)
		act_bio = lpf.filt(sim.data[probe_bio_spikes][int(t_transient/dt):], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes][int(t_transient/dt):], dt=dt)
		act_alif = lpf.filt(sim.data[probe_alif_spikes][int(t_transient/dt):], dt=dt)
		act_bio_decoders = lpf_decoders.filt(sim.data[probe_bio_spikes][int(t_transient/dt):], dt=dt)
		act_lif_decoders = lpf_decoders.filt(sim.data[probe_lif_spikes][int(t_transient/dt):], dt=dt)
		act_alif_decoders = lpf_decoders.filt(sim.data[probe_alif_spikes][int(t_transient/dt):], dt=dt)
		# bio readout is always "oracle" for the oracle method training
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio_decoders, sim.data[probe_oracle][int(t_transient/dt):])[0]
		# d_readout_bio_new = nengo.solvers.Lstsq()(act_bio_decoders, sim.data[probe_oracle_decoders][int(t_transient/dt):])[0]
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
			d_readout_alif_new = sim.data[conn_alif].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif_decoders, sim.data[probe_oracle][int(t_transient/dt):])[0]
			d_readout_alif_new = nengo.solvers.LstsqL2(reg=reg)(act_alif_decoders, sim.data[probe_oracle][int(t_transient/dt):])[0]
			# d_readout_lif_new = nengo.solvers.Lstsq()(act_lif_decoders, sim.data[probe_oracle_decoders][int(t_transient/dt):])[0]
			# d_readout_alif_new = nengo.solvers.Lstsq()(act_alif_decoders, sim.data[probe_oracle_decoders][int(t_transient/dt):])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][int(t_transient/dt):]
		xhat_bio = np.dot(act_bio, d_readout_bio)
		xhat_lif = np.dot(act_lif, d_readout_lif)
		xhat_alif = np.dot(act_alif, d_readout_alif)
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)
		rmse_alif = rmse(x_target, xhat_alif)

		if plot == 'signals':
			plt.plot(sim.trange(), sim.data[probe_pre], label='pre')
			plt.plot(sim.trange(), sim.data[probe_pre_deriv], label='pre_deriv')
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange()[int(t_transient/dt):], xhat_alif, label='adaptive lif, rmse=%.5f' % rmse_alif)
			plt.plot(sim.trange()[int(t_transient/dt):], x_target, label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			# plt.xlim((0.15, t_transient+t_final))
			plt.legend()
		elif plot == 'rates':
			# plt.plot(sim.trange(), 100*np.arange(pre_neurons)[None,:]+act_pre, label='pre')
			# plt.plot(sim.trange(), act_bio[:,:10], label='bio')
			# plt.plot(sim.trange(), act_lif[:,:10], label='lif', linestyle='--')
			plt.plot(sim.trange()[int(t_transient/dt):], 100*np.arange(bio_neurons)[None,:]+act_bio[:,:10], label='bio')
			# plt.plot(sim.trange(), 100*np.arange(bio_neurons)[None,:]+act_lif[:,:10], label='lif', linestyle='--')
			plt.xlabel('time (s)')
			plt.ylabel('firing rates (Hz)')
			plt.legend()

		return d_readout_bio_new, d_readout_lif_new, d_readout_alif_new, rmse_bio


	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))
	d_readout_alif_init = np.zeros((bio_neurons, dim))

	d_readout_bio_new, d_readout_lif_new, d_readout_alif_new, rmse_bio = sim(
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		d_readout_alif=d_readout_alif_init,
		tau_decoders=tau_decoders,
		signal_type=signal_train,
		freq=freq_train,
		seeds=seed_train,
		t_final=t_train,
		readout_LIF='oracle',
		plot=False)
	d_readout_bio_extra, d_readout_lif_extra, d_readout_alif_extra, rmse_bio = sim(
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		d_readout_alif=d_readout_alif_new,
		tau_decoders=tau_decoders,
		signal_type=signal_test,
		freq=freq_test,
		seeds=seed_test,
		t_final=t_test,
		readout_LIF='oracle',
		plot=to_plot)
	assert rmse_bio < cutoff