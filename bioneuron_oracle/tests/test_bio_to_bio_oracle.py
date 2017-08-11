import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver, get_stim_deriv
from nengolib.signal import s, z

def test_bio_to_bio_1d(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	bio2_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
	bio_radius = 1
	bio2_radius = 1
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
	rms = 1.0

	signal_train = 'white_noise'
	freq_train = 1
	seed_train = 1
	transform_train = 3.0
	t_train = 1.0

	signal_test = 'white_noise'
	freq_test = 1
	seed_test = 1
	transform_test = 3.0
	t_test = 1.0

	dim = 1
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1


	def sim(
		d_bio_bio2,
		d_bio_bio2_deriv,
		d_readout_bio,
		d_readout_bio2,
		d_readout_lif,
		d_readout_lif2,
		readout_LIF = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1,
		plot=False):


		deriv_trans = get_stim_deriv(
			signal, network_seed, sim_seed, freq, seeds, t_final, max_freq, rms, tau, dt)

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
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif')
			bio2 = nengo.Ensemble(
				n_neurons=bio2_neurons,
				dimensions=dim+1,
				seed=bio2_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio2_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio2')
			lif2 = nengo.Ensemble(
				n_neurons=bio2.n_neurons,
				dimensions=dim,
				seed=bio2.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif2')
			oracle = nengo.Node(size_in=dim, label='oracle')
			oracle_deriv = nengo.Node(size_in=dim, label='oracle_deriv')
			oracle2 = nengo.Node(size_in=dim, label='oracle2')
			temp = nengo.Node(size_in=dim, label='temp')
			temp2 = nengo.Node(size_in=dim, label='temp2')

			bio_bio2_solver = OracleSolver(decoders_bio = d_bio_bio2)
			bio_bio2_deriv_solver = OracleSolver(decoders_bio = d_bio_bio2_deriv)

			nengo.Connection(stim, pre, synapse=None)
			nengo.Connection(stim, pre_deriv, synapse=(1.0 - ~z) / dt)
			pre_bio = nengo.Connection(pre, bio[0],
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				n_syn=n_syn)
			nengo.Connection(pre_deriv, bio[1],
				weights_bias_conn=False,
				seed=2*conn_seed,
				synapse=tau,
				transform=deriv_trans*transform,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=1)
			bio_bio2 = nengo.Connection(bio[0], bio2[0],  # 1d or 2d?
				weights_bias_conn=True,
				seed=conn2_seed,
				synapse=tau,
				transform=transform,
				solver = bio_bio2_solver,
				n_syn=n_syn)
			# bio_bio2_deriv = nengo.Connection(bio[1], bio2[1],  # 1d or 2d?
			# 	weights_bias_conn=False,
			# 	seed=2*conn2_seed,
			# 	synapse=tau,
			# 	transform=transform,
			# 	solver = bio_bio2_deriv_solver,
			# 	n_syn=n_syn)
			nengo.Connection(lif, lif2,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=transform)
			nengo.Connection(oracle, oracle_deriv,
				synapse=(1.0 - ~z) / dt,
				transform=transform)
			nengo.Connection(oracle, oracle2,
				synapse=tau,
				transform=transform)
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))
			conn_lif2 = nengo.Connection(lif2, temp2,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_pre = nengo.Probe(pre, synapse=tau_readout)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_oracle_deriv = nengo.Probe(oracle_deriv, synapse=tau)  # synapse?
			probe_lif2 = nengo.Probe(lif2, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
			probe_lif2_spikes = nengo.Probe(lif2.neurons, 'spikes')
			probe_oracle2 = nengo.Probe(oracle2, synapse=tau_readout)


		"""
		Simulate the network, collect bioneuron activities and target values,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
		act_lif2 = lpf.filt(sim.data[probe_lif2_spikes], dt=dt)
		# bio readout is always "oracle" for the oracle method training
		d_readout_bio_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_readout_bio2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio2, sim.data[probe_oracle2])[0]
		if readout_LIF == 'LIF':
			d_readout_lif_new = sim.data[conn_lif].weights.T
			d_readout_lif2_new = sim.data[conn_lif2].weights.T
		elif readout_LIF == 'oracle':
			d_readout_lif_new = nengo.solvers.LstsqL2(reg=reg)(act_lif, sim.data[probe_oracle])[0]
			d_readout_lif2_new = nengo.solvers.LstsqL2(reg=reg)(act_lif2, sim.data[probe_oracle2])[0]
		d_bio_bio2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_bio_bio2_deriv_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle_deriv])[0]

		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		x_target = sim.data[probe_oracle][:,0]
		x_target2 = sim.data[probe_oracle2][:,0]
		xhat_bio = np.dot(act_bio, d_readout_bio)[:,0]
		xhat_bio2 = np.dot(act_bio2, d_readout_bio2)[:,0]
		xhat_lif = np.dot(act_lif, d_readout_lif)[:,0]
		xhat_lif2 = np.dot(act_lif2, d_readout_lif2)[:,0]
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_bio2 = rmse(x_target2, xhat_bio2)
		rmse_lif = rmse(x_target, xhat_lif)
		rmse_lif2 = rmse(x_target2, xhat_lif2)

		if plot:
			plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target, label='oracle')
			plt.plot(sim.trange(), xhat_bio2, label='bio2, rmse=%.5f' % rmse_bio2)
			plt.plot(sim.trange(), xhat_lif2, label='lif2, rmse=%.5f' % rmse_lif2)
			plt.plot(sim.trange(), x_target2, label='oracle2')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_bio_bio2_new, d_bio_bio2_new, d_readout_bio_new, d_readout_lif_new, \
				d_readout_bio2_new, d_readout_lif2_new, rmse_bio, rmse_bio2


	"""
	Run the test
	"""
	d_bio_bio2_init = np.zeros((bio_neurons, dim))
	d_bio_bio2_deriv_init = np.zeros((bio_neurons, dim))
	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_bio2_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))
	d_readout_lif2_init = np.zeros((bio_neurons, dim))

	d_bio_bio2_new, d_bio_bio2_deriv_new, d_readout_bio_extra, d_readout_lif_extra, \
			d_readout_bio2_extra, d_readout_lif2_extra, rmse_bio, rmse_bio2 = sim(
		d_bio_bio2=d_bio_bio2_init,
		d_bio_bio2_deriv=d_bio_bio2_deriv_init,
		d_readout_bio=d_readout_bio_init,
		d_readout_bio2=d_readout_bio2_init,
		d_readout_lif=d_readout_lif_init,
		d_readout_lif2=d_readout_lif2_init,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		plot=False)
	d_bio_bio2_extra, d_bio_bio2_deriv_extra, d_readout_bio_new, d_readout_lif_new, \
			d_readout_bio2_new, d_readout_lif2_new, rmse_bio, rmse_bio2 = sim(
		d_bio_bio2=d_bio_bio2_new,
		d_bio_bio2_deriv=d_bio_bio2_deriv_new,
		d_readout_bio=d_readout_bio_extra,
		d_readout_bio2=d_readout_bio2_extra,
		d_readout_lif=d_readout_lif_extra,
		d_readout_lif2=d_readout_lif2_extra,
		signal=signal_train,
		freq=freq_train,
		seeds=seed_train,
		transform=transform_train,
		t_final=t_train,
		plot=False)
	d_bio_bio2_extra, d_bio_bio2_deriv_extra, d_readout_bio_extra, d_readout_lif_extra, \
			d_readout_bio2_extra, d_readout_lif2_extra, rmse_bio, rmse_bio2 = sim(
		d_bio_bio2=d_bio_bio2_new,
		d_bio_bio2_deriv=d_bio_bio2_deriv_new,
		d_readout_bio=d_readout_bio_new,
		d_readout_bio2=d_readout_bio2_new,
		d_readout_lif=d_readout_lif_new,
		d_readout_lif2=d_readout_lif2_new,
		signal=signal_test,
		freq=freq_test,
		seeds=seed_test,
		transform=transform_test,
		t_final=t_test,
		plot=True)
	assert rmse_bio2 < cutoff
