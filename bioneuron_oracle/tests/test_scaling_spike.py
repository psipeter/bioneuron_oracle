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
	seeds = rng.randint(0,9009,size=n_avg)

	evo_params = {
		'dt': dt,
		'tau_readout': tau_readout,
		'sim_seed': sim_seed,
		'n_processes': 10,
		'popsize': 10,
		'generations' : 10,
		'w_0': 1e-1,
		'delta_w' :1e-1,
		'evo_seed' :9,
		'evo_t_final' :1.0,
	}

	def sim(
		w_pre_bio,
		d_readout,
		evo_params,
		readout = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1.0,
		bio_neurons=3,
		pre_seed=1,
		bio_seed=1,
		conn_seed=1,
		train=False,
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

			pre_bio_solver = TrainedSolver(weights_bio = w_pre_bio)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				trained_weights=True,
				solver = pre_bio_solver,
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
		Perform spike-match training on the pre-bio weights
		"""
		if train:
			network = spike_train(network, evo_params, plots=False)
			w_pre_bio_new = pre_bio.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio

		"""
		Run the simulation with the new w_pre_bio
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)

		"""
		Calculate readout decoders by either
			- grabbing the ideal decoders from the LIF population, OR
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		if readout == 'LIF':
			d_readout_new = sim.data[conn_lif].weights.T
		elif readout == 'oracle':
			d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle], xhat_lif)

		return w_pre_bio_new, d_readout_new, rmse_bio, rmse_lif


	"""
	Run the test
	"""
	columns = ('n_neurons', 'seed', 'pop', 'rmse')
	df = pd.DataFrame(columns=columns)
	k=0
	for bio_neuron in bio_neurons:
		for seed in seeds:	
			weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
			weight_filename = 'w_scaling_n_neurons=%s_seed=%s.npz' %(bio_neuron, seed) 
			try:
				w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
				to_train = False
			except IOError:
				w_pre_bio_init = np.zeros((bio_neuron, pre_neurons, n_syn))
				to_train = True
			d_readout_init = np.zeros((bio_neuron, dim))

			w_pre_bio_new, d_readout_new, rmse_bio, rmse_lif = sim(
				w_pre_bio=w_pre_bio_init,
				d_readout=d_readout_init,
				evo_params=evo_params,
				signal='sinusoids',
				freq=freq_train,
				seeds=seed_train,
				transform=transform,
				bio_neurons=bio_neuron,
				pre_seed=seed,
				bio_seed=seed,
				conn_seed=seed,
				t_final=t_final,
				train=to_train,
				plot=False)
			w_pre_bio_extra, d_readout_extra, rmse_bio, rmse_lif = sim(
				w_pre_bio=w_pre_bio_new,
				d_readout=d_readout_new,
				evo_params=evo_params,
				signal='sinusoids',
				freq=freq_test,
				seeds=seed_test,
				transform=transform,
				bio_neurons=bio_neuron,
				pre_seed=seed,
				bio_seed=seed,
				conn_seed=seed,
				t_final=t_final,
				train=False,
				plot=True)
			np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)
			df.loc[k] = [bio_neuron, seed, 'bio', rmse_bio]
			df.loc[k+1] = [bio_neuron, seed, 'lif', rmse_lif]
			k+=2

	sns.tsplot(time='n_neurons', value='rmse', unit='seed', condition='pop', data=df)
	plt.xlim(0,max(bio_neurons))

	assert rmse_bio < cutoff


def test_generations(Simulator, plt):
	# Nengo Parameters
	pre_neurons = 100
	bio_neurons = 10
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
	generations = [1, 5, 10, 20, 30]
	rng = np.random.RandomState(seed=1)
	seeds = rng.randint(0,9009,size=n_avg)

	def sim(
		w_pre_bio,
		d_readout,
		readout = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1.0,
		generations=10,
		evo_seed=9,
		train=False,
		plot=False):

		evo_params = {
			'dt': dt,
			'tau_readout': tau_readout,
			'sim_seed': sim_seed,
			'n_processes': 10,
			'popsize': 10,
			'generations' : generations,
			'w_0': 1e-1,
			'delta_w' :1e-1,
			'evo_seed' : evo_seed,
			'evo_t_final' :1.0,
		}
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

			pre_bio_solver = TrainedSolver(weights_bio = w_pre_bio)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				seed=conn_seed,
				synapse=tau,
				transform=transform,
				trained_weights=True,
				solver = pre_bio_solver,
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
		Perform spike-match training on the pre-bio weights
		"""
		if train:
			network = spike_train(network, evo_params, plots=False)
			w_pre_bio_new = pre_bio.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio

		"""
		Run the simulation with the new w_pre_bio
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)

		"""
		Calculate readout decoders by either
			- grabbing the ideal decoders from the LIF population, OR
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		if readout == 'LIF':
			d_readout_new = sim.data[conn_lif].weights.T
		elif readout == 'oracle':
			d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]


		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)
		xhat_lif = sim.data[probe_lif]
		rmse_bio = rmse(sim.data[probe_oracle][:,0], xhat_bio[:,0])
		rmse_lif = rmse(sim.data[probe_oracle], xhat_lif)

		return w_pre_bio_new, d_readout_new, rmse_bio, rmse_lif


	"""
	Run the test
	"""
	columns = ('generations', 'seed', 'pop', 'rmse')
	df = pd.DataFrame(columns=columns)
	k=0
	for generation in generations:
		for seed in seeds:
			weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
			weight_filename = 'w_scaling_generation=%s_seed=%s.npz' %(generation, seed) 
			try:
				w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
				to_train = False
			except IOError:
				w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
				to_train = True
			d_readout_init = np.zeros((bio_neurons, dim))

			w_pre_bio_new, d_readout_new, rmse_bio, rmse_lif = sim(
				w_pre_bio=w_pre_bio_init,
				d_readout=d_readout_init,
				signal='sinusoids',
				freq=freq_train,
				seeds=seed_train,
				transform=transform,
				generations=generation,
				evo_seed = seed,
				t_final=t_final,
				train=to_train,
				plot=False)
			w_pre_bio_extra, d_readout_extra, rmse_bio, rmse_lif = sim(
				w_pre_bio=w_pre_bio_new,
				d_readout=d_readout_new,
				signal='sinusoids',
				freq=freq_test,
				seeds=seed_test,
				transform=transform,
				generations=generation,
				evo_seed = seed,
				t_final=t_final,
				train=False,
				plot=True)
			np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)
			df.loc[k] = [generation, seed, 'bio', rmse_bio]
			df.loc[k+1] = [generation, seed, 'lif', rmse_lif]
			k+=2

	sns.tsplot(time='generations', value='rmse', unit='seed', condition='pop', data=df)
	plt.xlim(0,max(generations))

	assert rmse_bio < cutoff