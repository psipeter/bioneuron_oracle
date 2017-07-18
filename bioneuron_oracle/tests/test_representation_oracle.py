import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver


def test_representation_1d(Simulator, plt):
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

	max_freq = 5
	rms = 1.0
	n_steps = 10
	freq_train = 1
	freq_test = 2
	seed_train = 1
	seed_test = 2

	dim = 1
	reg = 0.1
	transform = 1
	t_final = 1.0
	cutoff = 0.1

	def sim(
		d_readout_bio,
		d_readout_lif,
		t_final=1.0,
		readout_LIF='LIF',
		signal='sinusoids',
		freq=1,
		seeds=1,
		transform=1,
		plot=False):

		"""
		Define the network
		"""
		with nengo.Network(seed=network_seed) as network:

			if signal == 'sinusoids':
				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq * t))
			elif signal == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=t_final, high=max_freq, rms=rms, seed=seeds))
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
				dimensions=dim,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio')
			lif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=bio.dimensions,
				seed=bio.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif')
			oracle = nengo.Node(size_in=dim)
			temp = nengo.Node(size_in=dim)

			nengo.Connection(stim, pre, synapse=None)
			nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle,
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
		error_bio = x_target - xhat_bio
		error_lif = x_target - xhat_lif
		f_plot = np.fft.fftfreq(sim.trange().shape[-1])
		dft_bio = np.fft.fft(error_bio)				
		dft_lif = np.fft.fft(error_lif)				

		if plot == 'signals':
			plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target, label='oracle')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()
		elif plot == 'errors':
			plt.plot(sim.trange(), error_bio, label='bio')
			plt.plot(sim.trange(), error_lif, label='lif')
			plt.xlabel('time (s)')
			plt.ylabel('Error ($\hat{x}(t)$)')
			plt.legend()
		elif plot == 'dfts':
			plt.plot(sim.trange(), dft_bio, label='bio')
			plt.plot(sim.trange(), dft_lif, label='lif')
			plt.xlabel('time (s)')
			plt.ylabel('DFT (error ($\hat{x}(t)$))')
			plt.legend()

		return d_readout_bio_new, d_readout_lif_new, rmse_bio


	d_readout_bio_init = np.zeros((bio_neurons, dim))
	d_readout_lif_init = np.zeros((bio_neurons, dim))

	d_readout_bio_new, d_readout_lif_new, rmse_bio = sim(
		d_readout_bio=d_readout_bio_init,
		d_readout_lif=d_readout_lif_init,
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		plot=False)
	d_readout_bio_extra, d_readout_lif_extra, rmse_bio = sim(
		d_readout_bio=d_readout_bio_new,
		d_readout_lif=d_readout_lif_new,
		signal='sinusoids',
		freq=freq_train,
		seeds=seed_train,
		transform=transform,
		t_final=t_final,
		plot='signals')

	assert rmse_bio < cutoff

# def test_representation_2d(Simulator, plt):
# 	# Nengo Parameters
# 	pre_neurons = 100
# 	bio_neurons = 10
# 	dt = 0.001
# 	min_rate = 150
# 	max_rate = 200
# 	radius = 1
# 	n_syn = 1

# 	pre_seed = 1
# 	bio_seed = 2
# 	conn_seed = 3
# 	network_seed = 4
# 	sim_seed = 5
# 	post_seed = 6
# 	inter_seed = 7

# 	max_freq = 5
# 	rms = 0.5

# 	dim = 2
# 	reg = 0.1
# 	t_final = 1.0
# 	cutoff = 0.1

# 	def sim(
# 		d_readout=None,
# 		tau=0.01,
# 		tau_readout=0.01,
# 		t_final=1.0,
# 		signal='sinusoids',
# 		freq=[1,1],
# 		seeds=[1,1],
# 		transform=-0.5,
# 		plot=False):

# 		"""
# 		Define the network
# 		"""
# 		with nengo.Network(seed=network_seed) as network:

# 			if signal == 'sinusoids':
# 				stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq[0] * t))
# 				stim2 = nengo.Node(lambda t: np.cos(2 * np.pi * freq[1] * t))
# 			elif signal == 'white_noise':
# 				stim = nengo.Node(nengo.processes.WhiteSignal(
# 					period=t_final, high=max_freq, rms=rms, seed=seeds[0]))
# 				stim2 = nengo.Node(nengo.processes.WhiteSignal(
# 					period=t_final, high=max_freq, rms=rms, seed=seeds[1]))
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
# 				dimensions=dim,
# 				seed=bio_seed,
# 				neuron_type=BahlNeuron(),
# 				# neuron_type=nengo.LIF(),
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				label='bio')
# 			lif = nengo.Ensemble(
# 				n_neurons=bio.n_neurons,
# 				dimensions=dim,
# 				seed=bio.seed,
# 				neuron_type=nengo.LIF(),
# 				max_rates=nengo.dists.Uniform(min_rate, max_rate),
# 				label='lif')
# 			oracle = nengo.Node(size_in=dim)

# 			nengo.Connection(stim, pre[0], synapse=None)
# 			nengo.Connection(stim2, pre[1], synapse=None)
# 			nengo.Connection(pre, bio,
# 				weights_bias_conn=True,
# 				seed=conn_seed,
# 				synapse=tau,
# 				transform=transform)
# 			nengo.Connection(pre, lif,
# 				synapse=tau,
# 				transform=transform)
# 			nengo.Connection(stim, oracle[0],
# 				synapse=tau,
# 				transform=transform)
# 			nengo.Connection(stim2, oracle[1],
# 				synapse=tau,
# 				transform=transform)

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
# 		d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]


# 		"""
# 		Use the old readout decoders to estimate the bioneurons' outputs for plotting
# 		"""
# 		xhat_bio = np.dot(act_bio, d_readout)
# 		xhat_lif = sim.data[probe_lif]
# 		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
# 		rmse_lif = rmse(sim.data[probe_oracle][:,0], xhat_lif[:,0])
# 		rmse_bio2 = rmse(sim.data[probe_oracle][:,1], xhat_bio[:,1])
# 		rmse_lif2 = rmse(sim.data[probe_oracle][:,1], xhat_lif[:,1])

# 		if plot:
# 			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
# 			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
# 			plt.plot(sim.trange(), xhat_bio[:,1], label='bio2, rmse=%.5f' % rmse_bio2)
# 			plt.plot(sim.trange(), xhat_lif[:,1], label='lif2, rmse=%.5f' % rmse_lif2)
# 			plt.plot(sim.trange(), sim.data[probe_oracle][:,0], label='oracle')
# 			plt.plot(sim.trange(), sim.data[probe_oracle][:,1], label='oracle2')
# 			plt.xlabel('time (s)')
# 			plt.ylabel('$\hat{x}(t)$')
# 			plt.legend()

# 		return d_readout_new, rmse_bio


# 	d_readout_init = np.zeros((bio_neurons, dim))

# 	d_readout_new, rmse_bio = sim(
# 		d_readout=d_readout_init,
# 		signal='sinusoids',
# 		tau=0.1,
# 		tau_readout=0.1,
# 		freq = [2,3],
# 		seeds = [2,3],
# 		transform=-0.5,
# 		t_final=t_final,
# 		plot=False)
# 	d_readout_extra, rmse_bio = sim(
# 		d_readout=d_readout_new,
# 		signal='sinusoids',
# 		tau=0.1,
# 		tau_readout=0.1,
# 		freq = [3,2],
# 		seeds = [3,2],
# 		transform=-0.5,
# 		t_final=t_final,
# 		plot=True)

# 	assert rmse_bio < cutoff