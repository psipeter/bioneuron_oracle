import numpy as np
import nengo
from pathos import multiprocessing as mp
from nengo.utils.numpy import rmse
from nengolib.signal import s, LinearSystem
import copy

__all__ = ['train_filters_decoders', 'build_filter']

def build_filter(zeros, poles):
	"""
	create the transfer function from the passed constants to serve as the filter
	"""
	built_filter = LinearSystem((zeros, poles, 1.0))
	built_filter /= built_filter.dcgain
	return built_filter

def train_filters_decoders(
	network,
	Simulator,
	sim_seed,
	signal_train,
	t_transient,
	t_final,
	dt,
	reg,
	n_processes,
	evo_popsize,
	evo_gen,
	evo_seed,
	zeros_init,
	poles_init,
	delta_zeros,
	delta_poles,
	bio_probe,
	target_probe):

	def evaluate(inputs):
		network=inputs[0]
		Simulator=inputs[1]
		signal_train=inputs[2]
		zeros = inputs[3][0]
		poles = inputs[3][1]
		bio_probe = inputs[4][0]
		target_probe = inputs[4][1]
		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters
		"""
		filt = build_filter(zeros, poles)
		with network:
			network.stim.output = lambda t: signal_train[int(t/dt)]
			bio_probe.synapse = filt
		"""
		run the simulation, collect filtered activites,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)
		act_bio = sim.data[bio_probe][int(t_transient/dt):]
		target = sim.data[target_probe][int(t_transient/dt):]
		if np.sum(act_bio) > 0:
			d_bio = nengo.solvers.LstsqL2(reg=reg)(act_bio, target)[0]
		else:
			d_bio = np.zeros((act_bio.shape[1], target.shape[1]))
		xhat_bio = np.dot(act_bio, d_bio)
		rmse_bio = rmse(target, xhat_bio)
		return rmse_bio

	def get_decoders(inputs, plot=False):
		network=inputs[0]
		Simulator=inputs[1]
		signal_train=inputs[2]
		zeros = inputs[3][0]
		poles = inputs[3][1]
		bio_probe = inputs[4][0]
		target_probe = inputs[4][1]
		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters
		"""
		filt = build_filter(zeros, poles)
		with network:
			network.stim.output = lambda t: signal_train[int(t/dt)]
			bio_probe.synapse = filt

		"""
		run the simulation, collect filtered activites,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)
		act_bio = sim.data[bio_probe][int(t_transient/dt):]
		target = sim.data[target_probe][int(t_transient/dt):]
		if np.sum(act_bio) > 0:
			d_bio = nengo.solvers.LstsqL2(reg=reg)(act_bio, target)[0]
		else:
			d_bio = np.zeros((act_bio.shape[1], target.shape[1]))
		xhat_bio = np.dot(act_bio, d_bio)
		rmse_bio = rmse(target, xhat_bio)

		if plot:
			import matplotlib.pyplot as plt
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' %rmse_bio)
			ax1.plot(sim.trange()[int(t_transient/dt):], target, label='oracle')
			ax1.set(xlabel='time (s)', ylabel='activity', title='zeros: %s \npoles: %s'
				%(zeros, poles))
			ax1.legend()
			figure.savefig('plots/deriv_filters/decodes_%s.png' %id(bio_probe))
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(sim.trange()[int(t_transient/dt):], act_bio, label='bio')
			ax1.set(xlabel='time (s)', ylabel='activity', title=str(filt))
			ax1.legend()
			figure.savefig('plots/deriv_filters/activities_%s.png' %id(bio_probe))

		return d_bio

	pool = mp.ProcessingPool(nodes=n_processes)
	rng = np.random.RandomState(seed=evo_seed)

	""" Initialize evolutionary population """
	filter_pop = []
	for p in range(evo_popsize):
		my_zeros= []
		my_poles = []
		for z in zeros_init:
			my_zeros.append(rng.uniform(-z, z))
		for p in poles_init:
			my_poles.append(rng.uniform(0, p))  # poles must be negative
		filter_pop.append([my_zeros, my_poles])

	""" Run evolutionary strategy """
	probes = [bio_probe, target_probe]
	fit_vs_gen = []
	for g in range(evo_gen):
		inputs = [[network, Simulator, signal_train, filter_pop[p], probes] for p in range(evo_popsize)]
		# fitnesses = np.array([evaluate(inputs[0]), evaluate(inputs[1]), evaluate(inputs[2])])  # debugging
		fitnesses = np.array(pool.map(evaluate, inputs))
		best_filter = filter_pop[np.argmin(fitnesses)]
		best_fitness = fitnesses[np.argmin(fitnesses)]
		fit_vs_gen.append([best_fitness])
		decay = np.exp(-g / 5.0)
		# decay = 1.0  # off
		""" repopulate filter pops with mutated copies of the best individual """
		filter_pop_new = []
		for p in range(evo_popsize):
			my_zeros = []
			my_poles = []
			for term in range(len(best_filter[0])):
				my_zeros.append(best_filter[0][term] + rng.normal(0, delta_zeros[term]) * decay)  # mutate
			for term in range(len(best_filter[1])):
				my_poles.append(best_filter[1][term] + rng.normal(0, delta_poles[term]) * decay)  # mutate	
			filter_pop_new.append([my_zeros, my_poles])
		filter_pop = filter_pop_new

	""" Grab the best filters and decoders and plot fitness vs generation """
	best_zeros = best_filter[0]
	best_poles = best_filter[1]
	decoders = np.array([get_decoders(inp) for inp in inputs])
	d_bio = decoders[np.argmin(fitnesses)]

	get_decoders([network, Simulator, signal_train, filter_pop[np.argmin(fitnesses)], probes], plot=True)  # plot training error
	fit_vs_gen = np.array(fit_vs_gen)
	import matplotlib.pyplot as plt
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(np.arange(0, evo_gen), fit_vs_gen)
	ax1.set(xlabel='Generation', ylabel='Fitness ($\hat{x}$ RMSE)')
	ax1.legend()
	figure.savefig('plots/deriv_filters/fitness_vs_generation_%s.png' % id(bio_probe))

	return best_zeros, best_poles, d_bio