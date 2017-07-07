import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver

def test_bio_to_bio(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 100
	bio2_neurons = 100
	tau = 0.1
	tau_readout = 0.01
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
	reg = 0.01
	t_final = 0.5
	cutoff = 0.1

	def sim(
		d_bio_bio2,
		d_readout=None,
		d_readout2=None,
		t_final=1.0,
		signal='sinusoids',
		freq=1,
		seeds=1,
		transform = -0.5,
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
				label='bio')
			lif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=dim,
				seed=bio.seed,
				neuron_type=nengo.LIF(),
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='lif')
			bio2 = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				seed=bio2_seed,
				neuron_type=BahlNeuron(),
				label='bio2')
			lif2 = nengo.Ensemble(
				n_neurons=bio2.n_neurons,
				dimensions=dim,
				seed=bio2.seed,
				neuron_type=nengo.LIF(),
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='lif2')
			oracle = nengo.Node(size_in=dim)
			oracle2 = nengo.Node(size_in=dim)

			bio_bio2_solver = OracleSolver(decoders_bio = d_bio_bio2)

			nengo.Connection(stim, pre, synapse=None)
			nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform)
			nengo.Connection(bio, bio2,
				weights_bias_conn=True,
				seed=conn2_seed,
				synapse=tau,
				transform=transform,
				solver=bio_bio2_solver)
			nengo.Connection(lif, lif2,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=transform)
			nengo.Connection(oracle, oracle2,
				synapse=tau,
				transform=transform)

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
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
		d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_readout2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio2, sim.data[probe_oracle2])[0]


		"""
		Compute decoders for the bio-bio2 connection
		"""
		d_bio_bio2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle], xhat_lif)
		xhat_bio2 = np.dot(act_bio2, d_readout2)
		xhat_lif2 = sim.data[probe_lif2]
		rmse_bio2 = rmse(sim.data[probe_oracle2][:,0], xhat_bio2[:,0])
		rmse_lif2 = rmse(sim.data[probe_oracle2], xhat_lif2)

		if plot:
			# plt.plot(sim.trange(), sim.data[probe_stim], label='stim')			
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), sim.data[probe_oracle], label='oracle')
			plt.plot(sim.trange(), xhat_bio2[:,0], label='bio2, rmse=%.5f' % rmse_bio2)
			plt.plot(sim.trange(), xhat_lif2, label='lif2, rmse=%.5f' % rmse_lif2)
			plt.plot(sim.trange(), sim.data[probe_oracle2], label='oracle2')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_bio_bio2_new, d_readout_new, d_readout2_new, rmse_bio


	d_bio_bio2_init = np.zeros((bio_neurons, dim))
	d_readout_init = np.zeros((bio_neurons, dim))
	d_readout2_init = np.zeros((bio_neurons, dim))

	d_bio_bio2_new, d_readout_extra, d_readout2_extra, rmse_bio = sim(
		d_bio_bio2=d_bio_bio2_init,
		d_readout=d_readout_init,
		d_readout2=d_readout2_init,
		signal='sinusoids',
		freq = 2,
		seeds = 2,
		transform=-0.5,
		t_final=t_final)
	d_bio_bio2_extra, d_readout_new, d_readout2_new, rmse_bio = sim(
		d_bio_bio2=d_bio_bio2_new,
		d_readout=d_readout_extra,
		d_readout2=d_readout2_extra,
		signal='sinusoids',
		freq = 2,
		seeds = 2,
		transform=-0.5,
		t_final=t_final,
		plot=False)
	d_bio_bio2_extra, d_readout_extra, d_readout2_extra, rmse_bio = sim(
		d_bio_bio2=d_bio_bio2_new,
		d_readout=d_readout_new,
		d_readout2=d_readout2_new,
		signal='sinusoids',
		freq = 3,
		seeds = 3,
		transform=-0.5,
		t_final=t_final,
		plot=True)

	assert rmse_bio < cutoff



def test_bio_to_bio_2d(Simulator, plt):
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

	dim = 2
	reg = 0.01
	t_final = 0.5
	cutoff = 0.1

	def sim(
		d_bio_bio2,
		d_readout=None,
		d_readout2=None,
		t_final=1.0,
		signal='sinusoids',
		freq=[1,1],
		seeds=[1,1],
		transform = -0.5,
		plot=False):


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
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				seed=bio_seed,
				neuron_type=BahlNeuron(),
				label='bio')
			lif = nengo.Ensemble(
				n_neurons=bio.n_neurons,
				dimensions=dim,
				seed=bio.seed,
				neuron_type=nengo.LIF(),
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				radius=radius,
				label='lif')
			bio2 = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				seed=bio2_seed,
				neuron_type=BahlNeuron(),
				label='bio2')
			lif2 = nengo.Ensemble(
				n_neurons=bio2.n_neurons,
				dimensions=dim,
				seed=bio2.seed,
				neuron_type=nengo.LIF(),
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				radius=radius,
				label='lif2')
			oracle = nengo.Node(size_in=dim)
			oracle2 = nengo.Node(size_in=dim)

			bio_bio2_solver = OracleSolver(decoders_bio = d_bio_bio2)

			nengo.Connection(stim, pre[0], synapse=None)
			nengo.Connection(stim2, pre[1], synapse=None)
			nengo.Connection(pre, bio,
				weights_bias_conn=True,
				seed=conn_seed,
				synapse=tau,
				transform=transform)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=transform)
			nengo.Connection(bio, bio2,
				weights_bias_conn=True,
				seed=conn2_seed,
				synapse=tau,
				transform=transform,
				solver=bio_bio2_solver)
			nengo.Connection(lif, lif2,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle[0],
				synapse=tau,
				transform=transform)
			nengo.Connection(stim2, oracle[1],
				synapse=tau,
				transform=transform)
			nengo.Connection(oracle, oracle2,
				synapse=tau,
				transform=transform)

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_stim2 = nengo.Probe(stim2, synapse=None)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
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
		d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
		d_readout2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio2, sim.data[probe_oracle2])[0]


		"""
		Compute decoders for the bio-bio2 connection
		"""
		d_bio_bio2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle][:,0], xhat_lif[:,0])
		rmse_bio_2 = rmse(sim.data[probe_oracle][:,1], xhat_bio[:,1])
		rmse_lif_2 = rmse(sim.data[probe_oracle][:,1], xhat_lif[:,1])

		xhat_bio2 = np.dot(act_bio2, d_readout2)
		xhat_lif2 = sim.data[probe_lif2]
		rmse_bio2 = rmse(sim.data[probe_oracle2][:,0], xhat_bio2[:,0])
		rmse_lif2 = rmse(sim.data[probe_oracle2][:,0], xhat_lif2[:,0])
		rmse_bio2_2 = rmse(sim.data[probe_oracle2][:,1], xhat_bio2[:,1])
		rmse_lif2_2 = rmse(sim.data[probe_oracle2][:,1], xhat_lif2[:,1])

		if plot:
			plt.plot(sim.trange(), xhat_bio[:,0], label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif[:,0], label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), xhat_bio[:,1], label='bio_2, rmse=%.5f' % rmse_bio_2)
			plt.plot(sim.trange(), xhat_lif[:,1], label='lif_2, rmse=%.5f' % rmse_lif_2)
			plt.plot(sim.trange(), xhat_bio2[:,0], label='bio2, rmse=%.5f' % rmse_bio2)
			plt.plot(sim.trange(), xhat_lif2[:,0], label='lif2, rmse=%.5f' % rmse_lif2)
			plt.plot(sim.trange(), xhat_bio2[:,1], label='bio2_2, rmse=%.5f' % rmse_bio2_2)
			plt.plot(sim.trange(), xhat_lif2[:,1], label='lif2_2, rmse=%.5f' % rmse_lif2_2)
			plt.plot(sim.trange(), sim.data[probe_oracle][:,0], label='oracle')
			plt.plot(sim.trange(), sim.data[probe_oracle][:,1], label='oracle_2')
			plt.plot(sim.trange(), sim.data[probe_oracle2][:,0], label='oracle2')
			plt.plot(sim.trange(), sim.data[probe_oracle2][:,1], label='oracle2_2')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		return d_bio_bio2_new, d_readout_new, d_readout2_new, rmse_bio


	d_bio_bio2_init = np.zeros((bio_neurons, dim))
	d_readout_init = np.zeros((bio_neurons, dim))
	d_readout2_init = np.zeros((bio_neurons, dim))

	d_bio_bio2_new, d_readout_extra, d_readout2_extra, rmse_bio = sim(
		d_bio_bio2=d_bio_bio2_init,
		d_readout=d_readout_init,
		d_readout2=d_readout2_init,
		signal='sinusoids',
		freq=[2,3],
		seeds=[2,3],
		transform=-0.5,
		t_final=t_final)
	d_bio_bio2_extra, d_readout_new, d_readout2_new, rmse_bio = sim(
		d_bio_bio2=d_bio_bio2_new,
		d_readout=d_readout_extra,
		d_readout2=d_readout2_extra,
		signal='sinusoids',
		freq=[2,3],
		seeds=[2,3],
		transform=-0.5,
		t_final=t_final,
		plot=False)
	d_bio_bio2_extra, d_readout_extra, d_readout2_extra, rmse_bio = sim(
		d_bio_bio2=d_bio_bio2_new,
		d_readout=d_readout_new,
		d_readout2=d_readout2_new,
		signal='sinusoids',
		freq=[3,2],
		seeds=[3,2],
		transform=-0.5,
		t_final=t_final,
		plot=True)

	assert rmse_bio < cutoff