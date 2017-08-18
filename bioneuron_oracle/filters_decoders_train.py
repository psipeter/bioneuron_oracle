import numpy as np
import nengo
from pathos import multiprocessing as mp
from nengo.utils.numpy import rmse
from nengolib.signal import s, LinearSystem
import copy

__all__ = ['train_filters_decoders', 'build_filter']

# def build_filter(b_s, a_s):
def build_filter(zeros, poles, gain):
	"""
	create the transfer function from the passed constants to serve as the filter
	"""
	# Manual way
	# numerator = 0
	# denominator = 0
	# for term in range(len(b_s)):
	# 	numerator += b_s[term] * (s**term) #add b0 +b1*s +b2*s^2 + ...
	# for term in range(len(a_s)):
	# 	denominator += a_s[term] * (s**term) #add a0 +a1*s +a2*s^2 + ...
	# assert denominator_bio != 0
	# built_filter = numerator / denominator

	# Nengolib way
	# https://arvoelke.github.io/nengolib-docs/types.html#linear_system_like
	# built_filter = LinearSystem((b_s, a_s))
	built_filter = LinearSystem((zeros, poles, gain))
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
	# b_s_init,
	# a_s_init,
	# delta_b_s,
	# delta_a_s
	zeros_init,
	poles_init,
	gain_init,
	delta_zeros,
	delta_poles,
	delta_gain):

	def evaluate(inputs):
		network=inputs[0]
		Simulator=inputs[1]
		signal_train=inputs[2]
		# b_s_bio=inputs[3][0]
		# b_s_alif=inputs[4][0]
		# a_s_bio=inputs[3][1]
		# a_s_alif=inputs[4][1]
		zeros_bio = inputs[3][0]
		zeros_alif = inputs[4][0]
		poles_bio = inputs[3][1]
		poles_alif = inputs[4][1]
		gain_bio = inputs[3][2]
		gain_alif = inputs[4][2]

		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters
		"""
		f_bio = build_filter(zeros_bio, poles_bio, gain_bio)
		f_alif = build_filter(zeros_alif, poles_alif, gain_alif)
		# bio_filter = build_filter(b_s_bio, a_s_bio)
		# alif_filter = build_filter(b_s_alif, a_s_alif)
		with network:
			network.stim.output = lambda t: signal_train[int(t/dt)]
			network.bio_probe.synapse = f_bio
			network.alif_probe.synapse = f_alif

		"""
		run the simulation, collect filtered activites,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)
		act_bio = sim.data[network.bio_probe][int(t_transient/dt):]
		act_alif = sim.data[network.alif_probe][int(t_transient/dt):]
		target = sim.data[network.target_probe][int(t_transient/dt):]
		if np.sum(act_bio) > 0:
			d_bio = nengo.solvers.LstsqL2(reg=reg)(act_bio, target)[0]
		else:
			d_bio = np.zeros((act_bio.shape[1], target.shape[1]))
		if np.sum(act_alif) > 0:
			d_alif = nengo.solvers.LstsqL2(reg=reg)(act_alif, target)[0]
		else:
			d_alif = np.zeros((act_alif.shape[1], target.shape[1]))
		xhat_bio = np.dot(act_bio, d_bio)
		xhat_alif = np.dot(act_alif, d_alif)
		rmse_bio = rmse(target, xhat_bio)
		rmse_alif = rmse(target, xhat_alif)

		return rmse_bio, rmse_alif

	def get_decoders(inputs, plot=False):
		network=inputs[0]
		Simulator=inputs[1]
		signal_train=inputs[2]
		# b_s_bio=inputs[3][0]
		# b_s_alif=inputs[4][0]
		# a_s_bio=inputs[3][1]
		# a_s_alif=inputs[4][1]
		zeros_bio = inputs[3][0]
		zeros_alif = inputs[4][0]
		poles_bio = inputs[3][1]
		poles_alif = inputs[4][1]
		gain_bio = inputs[3][2]
		gain_alif = inputs[4][2]

		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters
		"""
		# bio_filter = build_filter(b_s_bio, a_s_bio)
		# alif_filter = build_filter(b_s_alif, a_s_alif)
		f_bio = build_filter(zeros_bio, poles_bio, gain_bio)
		f_alif = build_filter(zeros_alif, poles_alif, gain_alif)
		with network:
			network.stim.output = lambda t: signal_train[int(t/dt)]
			network.bio_probe.synapse = f_bio
			network.alif_probe.synapse = f_alif

		"""
		run the simulation, collect filtered activites,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)
		act_bio = sim.data[network.bio_probe][int(t_transient/dt):]
		act_alif = sim.data[network.alif_probe][int(t_transient/dt):]
		target = sim.data[network.target_probe][int(t_transient/dt):]
		if np.sum(act_bio) > 0:
			d_bio = nengo.solvers.LstsqL2(reg=reg)(act_bio, target)[0]
		else:
			d_bio = np.zeros((act_bio.shape[1], target.shape[1]))
		if np.sum(act_alif) > 0:
			d_alif = nengo.solvers.LstsqL2(reg=reg)(act_alif, target)[0]
		else:
			d_alif = np.zeros((act_alif.shape[1], target.shape[1]))
		xhat_bio = np.dot(act_bio, d_bio)
		xhat_alif = np.dot(act_alif, d_alif)
		rmse_bio = rmse(target, xhat_bio)
		rmse_alif = rmse(target, xhat_alif)

		if plot:
			import matplotlib.pyplot as plt
			ID = np.random.randint(9999)
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' %rmse_bio)
			ax1.plot(sim.trange()[int(t_transient/dt):], xhat_alif, label='alif, rmse=%.5f' %rmse_alif)
			ax1.plot(sim.trange()[int(t_transient/dt):], target, label='oracle')
			ax1.set(xlabel='time (s)', ylabel='activity', title='zeros: %s \npoles: %s \ngain: %s'
				%(zeros_bio, poles_bio, gain_bio))
			ax1.legend()
			figure.savefig('plots/filters/decodes_%s.png' %ID)

			figure, ax1 = plt.subplots(1,1)
			ax1.plot(sim.trange()[int(t_transient/dt):], act_bio, label='bio')
			ax1.plot(sim.trange()[int(t_transient/dt):], act_alif, label='alif')
			ax1.set(xlabel='time (s)', ylabel='activity', title=str(f_bio))
			ax1.legend()
			figure.savefig('plots/filters/activities_%s.png' %ID)

		return d_bio, d_alif

	pool = mp.ProcessingPool(nodes=n_processes)
	rng = np.random.RandomState(seed=evo_seed)

	# for ens in bio_ens.iterkeys():  # For each readout filter to be trained:
	"""
	Initialize evolutionary population
	"""
	bio_filter_pop = []
	alif_filter_pop = []
	for p in range(evo_popsize):
		my_zeros_bio = []
		my_zeros_alif = []
		my_poles_bio = []
		my_poles_alif = []
		my_gain_bio = gain_init
		my_gain_alif = gain_init
		# for term in range(len(b_s_init)):
		# 	b_s_init[term] += rng.normal(0, delta_b_s[term])
		# for term in range(len(a_s_init)):
		# 	a_s_init[term] += rng.normal(0, delta_a_s[term])

		''' always initialize with given poles/zeros '''
		for z in zeros_init:
			my_zeros_bio.append(rng.uniform(z, z))
			my_zeros_alif.append(rng.uniform(z, z))
		for p in poles_init:
			my_poles_bio.append(rng.uniform(p, p))
			my_poles_alif.append(rng.uniform(p, p))
		# my_gain_bio += rng.normal(0, delta_gain)
		# my_gain_alif += rng.normal(0, delta_gain)

		bio_filter_pop.append([my_zeros_bio, my_poles_bio, my_gain_bio])
		alif_filter_pop.append([my_zeros_alif, my_poles_alif, my_gain_alif])
		# bio_filter_pop.append([b_s_init, a_s_init])  # [[b_s], [a_s], transfer_function]
		# alif_filter_pop.append([b_s_init, a_s_init])

	fit_vs_gen = []  # [[bio_fit_g=1, alif_fit_g=1], [bio_fit_g=2, alif_fit_g=2]]
	for g in range(evo_gen):
		inputs = [[network, Simulator, signal_train, bio_filter_pop[p], alif_filter_pop[p]]
			for p in range(evo_popsize)]
		# fitnesses = np.array([evaluate(inputs[0]), evaluate(inputs[1]), evaluate(inputs[2])])  # debugging
		# assert False
		fitnesses = np.array(pool.map(evaluate, inputs))
		# print fitnesses, np.argmin(fitnesses[:,0]), np.argmin(fitnesses[:,1])
		# assert False
		best_bio_filter = bio_filter_pop[np.argmin(fitnesses[:,0])]
		best_alif_filter = alif_filter_pop[np.argmin(fitnesses[:,1])]
		best_bio_fitness = fitnesses[np.argmin(fitnesses[:,0])][0]
		best_alif_fitness = fitnesses[np.argmin(fitnesses[:,1])][1]
		# print bio_filter_pop
		# print alif_filter_pop
		# print fitnesses, np.argmin(fitnesses[:,0]), best_bio_fitness
		# print fitnesses, np.argmin(fitnesses[:,1]), best_alif_fitness
		# if g==1: assert False
		fit_vs_gen.append([best_bio_fitness, best_alif_fitness])
		decay = np.exp(-g / 10.0)
		# decay = 1.0  # off
		#repopulate filter pops with mutated copies of the best individual
		bio_filter_pop_new = []
		alif_filter_pop_new = []
		for p in range(evo_popsize):
			my_zeros_bio = []
			my_zeros_alif = []
			my_poles_bio = []
			my_poles_alif = []
			# b_s_bio = best_bio_filter[0]
			# b_s_alif = best_alif_filter[0]
			# a_s_bio = best_bio_filter[1]
			# a_s_alif = best_alif_filter[1]
			# my_zeros_bio = best_bio_filter[0]
			# my_zeros_alif = best_alif_filter[0]
			# my_poles_bio = best_bio_filter[1]
			# my_poles_alif = best_alif_filter[1]
			for term in range(len(best_bio_filter[0])):
				my_zeros_bio.append(best_bio_filter[0][term] + rng.normal(0, delta_zeros[term]) * decay)  # mutate
				my_zeros_alif.append(best_alif_filter[0][term] + rng.normal(0, delta_zeros[term]) * decay)  # mutate
			for term in range(len(best_bio_filter[1])):
				my_poles_bio.append(best_bio_filter[1][term] + rng.normal(0, delta_poles[term]) * decay)  # mutate
				my_poles_alif.append(best_alif_filter[1][term] + rng.normal(0, delta_poles[term]) * decay)  # mutate
			my_gain_bio = best_bio_filter[2]
			my_gain_alif = best_alif_filter[2]
			# my_gain_bio += rng.normal(0, delta_gain) 
			# my_gain_alif += rng.normal(0, delta_gain) 	
			bio_filter_pop_new.append([my_zeros_bio, my_poles_bio, my_gain_bio])
			alif_filter_pop_new.append([my_zeros_alif, my_poles_alif, my_gain_alif])
		bio_filter_pop = bio_filter_pop_new
		alif_filter_pop = alif_filter_pop_new

			# for term in range(len(b_s_bio)):
			# 	b_s_bio[term] += rng.normal(0, delta_b_s[term])  # mutate, TODO generation scaling
			# 	b_s_alif[term] += rng.normal(0, delta_b_s[term])  # mutate
			# for term in range(len(a_s_bio)):
			# 	a_s_bio[term] += rng.normal(0, delta_a_s[term])  # mutate
			# 	a_s_alif[term] += rng.normal(0, delta_a_s[term])  # mutate
			# bio_filter_pop[p] = [b_s_bio, a_s_bio]
			# alif_filter_pop[p] = [b_s_alif, a_s_alif]

	"""
	Grab the best filters and decoders
	"""
	# f_bio = build_filter(best_bio_filter[0], best_bio_filter[1])  # transfer function
	# f_alif = build_filter(best_alif_filter[0], best_alif_filter[1])  # transfer function
	# b_s_bio = best_bio_filter[0]
	# b_s_alif = best_alif_filter[0]
	# a_s_bio = best_bio_filter[1]
	# a_s_alif = best_alif_filter[1]
	zeros_bio = best_bio_filter[0]
	zeros_alif = best_alif_filter[0]
	poles_bio = best_bio_filter[1]
	poles_alif = best_alif_filter[1]
	gain_bio = best_bio_filter[2]
	gain_alif = best_alif_filter[2]
	decoders = np.array([get_decoders(inp) for inp in inputs])
	d_bio = decoders[np.argmin(fitnesses[:,0])][0]
	d_alif = decoders[np.argmin(fitnesses[:,1])][1]
	get_decoders([network, Simulator, signal_train,
		bio_filter_pop[np.argmin(fitnesses[:,0])],
		alif_filter_pop[np.argmin(fitnesses[:,1])]], plot=True)  # plot training error

	fit_vs_gen = np.array(fit_vs_gen)
	import matplotlib.pyplot as plt
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(np.arange(0, evo_gen), fit_vs_gen[:,0], label='bio')
	ax1.plot(np.arange(0, evo_gen), fit_vs_gen[:,1], label='alif')
	ax1.set(xlabel='Generation', ylabel='Fitness ($\hat{x}$ RMSE)')
	ax1.legend()
	figure.savefig('plots/filters/fitness_vs_generation.png')

	return zeros_bio, zeros_alif, poles_bio, poles_alif, gain_bio, gain_alif, d_bio, d_alif
	# return b_s_bio, b_s_alif, a_s_bio, a_s_alif, d_bio, d_alif

# numerator_bio = 0
# numerator_alif = 0
# denominator_bio = 0
# denominator_alif = 0
# for term in range(len(b_s)):
# 	numerator_bio += b_s[term] * (s**term) #add b0 +b1*s +b2*s^2 + ...
# 	numerator_alif += b_s[term] * (s**term) #add b0 +b1*s +b2*s^2 + ...
# for term in range(len(a_s)):
# 	denominator_bio += a_s[term] * (s**term) #add a0 +a1*s +a2*s^2 + ...
# 	denominator_alif += a_s[term] * (s**term) #add a0 +a1*s +a2*s^2 + ...
# assert denominator_bio != 0
# assert denominator_alif != 0
# transfer_function_bio = numerator_bio / denominator_bio
# transfer_function_alif = numerator_alif / denominator_alif