import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, TrainedSolver, spike_train
import pandas as pd
import seaborn as sns

def test_n_neurons(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	tau = 0.1
	tau_readout = 0.1
	dt = 0.001
	min_rate = 150
	max_rate = 200
	radius = 1
	bio_radius = 1
	n_syn = 1

	network_seed = 4
	sim_seed = 5
	post_seed = 6
	inter_seed = 7

	max_freq = 5
	rms = 1.0
	freq_train = 1
	freq_test = 2
	seed_train = 1
	seed_test = 2

	dim = 1
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1
	transform = 1.0

	n_avg = 10
	rng = np.random.RandomState(seed=1)
	bio_neurons = np.array([2, 4, 6, 8, 10, 15, 30, 50])
	seeds = rng.randint(0, 9009, size=n_avg)

	def sim(
		d_readout_bio,
		d_readout_lif,
		readout_LIF='LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		pre_seed=1,
		bio_seed=1,
		conn_seed=1,
		transform=1.0,
		bio_neurons=3,
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
			pre_bio = nengo.Connection(pre, bio,
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

		return d_readout_bio_new, d_readout_lif_new, rmse_bio, rmse_lif


	"""
	Run the test
	"""
	columns = ('n_neurons', 'seed', 'pop', 'rmse')
	df = pd.DataFrame(columns=columns)
	k=0
	for bio_neuron in bio_neurons:
		for seed in seeds:
			d_readout_bio_init = np.zeros((bio_neuron, dim))
			d_readout_lif_init = np.zeros((bio_neuron, dim))

			d_readout_bio_new, d_readout_lif_new, rmse_bio, rmse_lif = sim(
				d_readout_bio=d_readout_bio_init,
				d_readout_lif=d_readout_lif_init,
				signal='sinusoids',
				bio_neurons=bio_neuron,
				freq=freq_test,
				seeds=seed_test,
				transform=transform,
				t_final=t_final,
				pre_seed=seed,
				bio_seed=2*seed,
				conn_seed=3*seed,
				plot=False)
			d_readout_bio_extra, d_readout_lif_extra, rmse_bio, rmse_lif = sim(
				d_readout_bio=d_readout_bio_new,
				d_readout_lif=d_readout_lif_new,
				signal='sinusoids',
				bio_neurons=bio_neuron,
				freq=freq_train,
				seeds=seed_train,
				transform=transform,
				t_final=t_final,
				pre_seed=seed,
				bio_seed=2*seed,
				conn_seed=3*seed,
				plot='signals')
			df.loc[k] = [bio_neuron, seed, 'bio', rmse_bio]
			df.loc[k+1] = [bio_neuron, seed, 'lif', rmse_lif]
			k+=2

	sns.tsplot(time='n_neurons', value='rmse', unit='seed', condition='pop', data=df)
	plt.xlim(0,max(bio_neurons))

	assert rmse_bio < cutoff