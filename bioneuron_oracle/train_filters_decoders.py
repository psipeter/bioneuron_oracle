import numpy as np
import nengo
from pathos import multiprocessing as mp
from nengo.utils.numpy import rmse
from nengolib.signal import s, LinearSystem
import nengolib
import copy

__all__ = ['train_feedforward', 'train_feedback', 'build_filter']

def build_filter(zeros, poles):
	"""
	create the transfer function from the passed constants to serve as the filter
	"""
	built_filter = LinearSystem((zeros, poles, 1.0))
	built_filter /= built_filter.dcgain
	return built_filter

def train_feedforward(
	network,
	Simulator,
	sim_seed,
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
	zeros_delta,
	poles_delta,
	bio_probe,  # probe of bioneuron activity (filtered spikes)
	target_probe,
	training_dir,
	training_file):

	def evaluate(inputs):
		network=inputs[0]
		Simulator=inputs[1]
		zeros = inputs[2][0]
		poles = inputs[2][1]
		bio_probe = inputs[3][0]
		target_probe = inputs[3][1]
		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters, as well as other filters that have been
		trained already (these can't be fed into pool.evaluate() without _paramdict errors)
		"""
		filt = build_filter(zeros, poles)
		with network:
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
		zeros = inputs[2]
		poles = inputs[3]
		bio_probe = inputs[4]
		target_probe = inputs[5]
		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters
		"""
		filt = build_filter(zeros, poles)
		with network:
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
		if hasattr(network, 'probe_inter_activity'):
			integ = sim.data[network.probe_integ][int(t_transient/dt):]
			act_inter = sim.data[network.probe_inter_activity][int(t_transient/dt):]
			d_inter = nengo.solvers.LstsqL2(reg=reg)(act_inter, integ)[0]
			d_inter = d_inter.reshape((d_inter.shape[0],1))
			# d_inter = sim.data[network.conn_lif].weights.T
		else:
			d_inter = None

		if plot:
			import matplotlib.pyplot as plt
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' %rmse_bio)
			ax1.plot(sim.trange()[int(t_transient/dt):], target, label='oracle')
			ax1.set(xlabel='time (s)', ylabel='activity', title='zeros: %s \npoles: %s'
				%(zeros, poles))
			ax1.legend()
			figure.savefig('plots/filters/ff_decodes_%s.png' %id(bio_probe))
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(sim.trange()[int(t_transient/dt):], act_bio, label='bio')
			ax1.set(xlabel='time (s)', ylabel='activity', title=str(filt))
			ax1.legend()
			figure.savefig('plots/filters/ff_activities_%s.png' %id(bio_probe))

		return d_bio, d_inter

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
	fit_vs_gen = []
	for g in range(evo_gen):
		probes = [bio_probe, target_probe]
		# reconfigure nengolib synapses to have propper attributes to be passed to pool.map()
		for probe in network.probes:
			if isinstance(probe.synapse, LinearSystem):
				try:
					probe.synapse._paramdict = nengo.Lowpass(0.1)._paramdict
					probe.synapse.tau = 0.1
					probe.synapse.default_size_in = 1
					probe.synapse.default_size_out = 1
				except:
					continue
		for conn in network.connections:
			if isinstance(conn.synapse, LinearSystem):
				try:
					conn.synapse._paramdict = nengo.Lowpass(0.1)._paramdict
					conn.synapse.tau = 0.1
					conn.synapse.default_size_in = 1
					conn.synapse.default_size_out = 1
				except:
					continue
		# for probe in probes:
		# 	if isinstance(probe.synapse, LinearSystem):
		# 		probe.synapse._paramdict = nengo.Lowpass(0.1)._paramdict
		# 		probe.synapse.tau = 0.1
		# 		probe.synapse.default_size_in = 1
		# 		probe.synapse.default_size_out = 1
		inputs = [[network, Simulator, filter_pop[p], probes] for p in range(evo_popsize)]
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
				my_zeros.append(best_filter[0][term] + rng.normal(0, zeros_delta[term]) * decay)  # mutate
			for term in range(len(best_filter[1])):
				my_poles.append(best_filter[1][term] + rng.normal(0, poles_delta[term]) * decay)  # mutate	
			filter_pop_new.append([my_zeros, my_poles])
		filter_pop = filter_pop_new

	""" Grab the best filters and decoders and plot fitness vs generation """
	best_zeros = best_filter[0]
	best_poles = best_filter[1]
	best_d_bio, d_inter = get_decoders([network, Simulator, best_zeros, best_poles, bio_probe, target_probe], plot=True)

	fit_vs_gen = np.array(fit_vs_gen)
	import matplotlib.pyplot as plt
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(np.arange(0, evo_gen), fit_vs_gen)
	ax1.set(xlabel='Generation', ylabel='Fitness ($\hat{x}$ RMSE)')
	ax1.legend()
	figure.savefig('plots/evolution/feedforward_%s.png' % id(bio_probe))
	figure, ax1 = plt.subplots(1,1)
	times = np.arange(0, 1e0, 1e-3)
	ax1.plot(times, build_filter(best_zeros, best_poles).impulse(len(times)), label='evolved')
	ax1.plot(times, nengolib.Lowpass(0.1).impulse(len(times)), label='lowpass')
	ax1.set(xlabel='time', ylabel='amplitude')
	ax1.legend()
	figure.savefig('plots/evolution/f_feedforward_%s.png' % id(bio_probe))

	np.savez(training_dir+training_file,
		zeros_feedforward=best_zeros,
		poles_feedforward=best_poles,
		d_feedforward=best_d_bio,
		d_inter=d_inter)

	return best_zeros, best_poles, best_d_bio, d_inter



def train_feedback(
	network,
	Simulator,
	sim_seed,
	t_transient,
	t_final,
	dt,
	reg,
	n_processes,
	evo_popsize,
	evo_gen,
	evo_seed,
	zeros_feedforward,
	poles_feedforward,
	d_feedforward,
	d_feedback_init,
	conn_feedback,
	decoders_delta,
	mutation_rate,
	bio_probe,
	target_probe,
	training_dir,
	training_file):

	def evaluate(inputs):
		network=inputs[0]
		Simulator=inputs[1]
		d_feedback = inputs[2]
		conn_feedback = inputs[3]
		bio_probe = inputs[4][0]
		zeros_feedforward = inputs[4][1]
		poles_feedforward = inputs[4][2]
		d_feedforward = inputs[4][3]
		target_probe = inputs[5]
		plot = inputs[6]

		""" set the readout filter and the recurrent bioneuron decoders"""
		filt = build_filter(zeros_feedforward, poles_feedforward)
		with network:
			bio_probe.synapse = filt
			conn_feedback.solver = nengo.solvers.NoSolver(d_feedback)
			# conn_feedback.solver.decoders_bio = d_feedback
		"""
		run the simulation, collect filtered activites,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_transient+t_final)
		act_bio = sim.data[bio_probe][int(t_transient/dt):]
		target = sim.data[target_probe][int(t_transient/dt):]
		xhat_bio = np.dot(act_bio, d_feedforward)
		rmse_bio = rmse(target, xhat_bio)

		if plot:
			import matplotlib.pyplot as plt
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(sim.trange()[int(t_transient/dt):], xhat_bio, label='bio, rmse=%.5f' %rmse_bio)
			ax1.plot(sim.trange()[int(t_transient/dt):], target, label='oracle')
			ax1.set(xlabel='time (s)', ylabel='activity')
			ax1.legend()
			figure.savefig('plots/filters/fb_decodes_%s.png' %id(bio_probe))
			# figure, ax1 = plt.subplots(1,1)
			# ax1.plot(sim.trange()[int(t_transient/dt):], act_bio, label='bio')
			# ax1.set(xlabel='time (s)', ylabel='activity')
			# ax1.legend()
			# figure.savefig('plots/filters/fb_activities_%s.png' %id(bio_probe))

		return rmse_bio

	pool = mp.ProcessingPool(nodes=n_processes)
	rng = np.random.RandomState(seed=evo_seed)

	""" Initialize evolutionary population """
	decoder_pop = []
	for p in range(evo_popsize):
		# my_decoders = rng.uniform(-10*decoders_delta, 10*decoders_delta, size=d_feedback_init.shape)
		# my_decoders = d_feedback_init + rng.uniform(-decoders_delta, decoders_delta, size=d_feedback_init.shape)
		d_feedback_delta = rng.normal(0, decoders_delta, size=d_feedback_init.shape)
		for dec in range(d_feedback_delta.shape[0]):
			for dim in range(d_feedback_delta.shape[1]):
				if rng.uniform(0.0, 1.0) > mutation_rate: # set to zero with prob = 1-mutation_rate
					d_feedback_delta[dec][dim] = 0.0
		my_decoders = d_feedback_init + d_feedback_delta
		decoder_pop.append(my_decoders)

	""" Run evolutionary strategy """
	fit_vs_gen = []
	for g in range(evo_gen):
		# reconfigure nengolib synapses to have propper attributes to be passed to pool.map()
		for probe in network.probes:
			if isinstance(probe.synapse, LinearSystem):
				try:
					probe.synapse._paramdict = nengo.Lowpass(0.1)._paramdict
					probe.synapse.tau = 0.1
					probe.synapse.default_size_in = 1
					probe.synapse.default_size_out = 1
				except:
					continue
		for conn in network.connections:
			if isinstance(conn.synapse, LinearSystem):
				try:
					conn.synapse._paramdict = nengo.Lowpass(0.1)._paramdict
					conn.synapse.tau = 0.1
					conn.synapse.default_size_in = 1
					conn.synapse.default_size_out = 1
				except:
					continue
		readout_info = [bio_probe, zeros_feedforward, poles_feedforward, d_feedforward]
		inputs = [[network, Simulator, decoder_pop[p], conn_feedback, readout_info, target_probe, False] for p in range(evo_popsize)]
		# fitnesses = np.array([evaluate(inputs[0]), evaluate(inputs[1]), evaluate(inputs[2])])  # debugging
		fitnesses = np.array(pool.map(evaluate, inputs))
		best_decoders = decoder_pop[np.argmin(fitnesses)]
		best_fitness = fitnesses[np.argmin(fitnesses)]
		fit_vs_gen.append([best_fitness])
		decay = np.exp(-g / 30.0)
		# decay = 1.0  # off
		""" repopulate decoders pops with mutated copies of the best individual """
		decoders_pop_new = []
		for p in range(evo_popsize):
			# my_decoders = best_decoders + rng.uniform(-decoders_delta, decoders_delta, size=best_decoders.shape) * decay
			# my_decoders = best_decoders + rng.normal(0, decoders_delta, size=best_decoders.shape) * decay
			d_feedback_delta = rng.normal(0, decoders_delta, size=d_feedback_init.shape) * decay
			for dec in range(d_feedback_delta.shape[0]):
				for dim in range(d_feedback_delta.shape[1]):
					if rng.uniform(0.0, 1.0) > mutation_rate: # set to zero with prob = 1-mutation_rate
						d_feedback_delta[dec][dim] = 0.0
			my_decoders = best_decoders + d_feedback_delta
			decoders_pop_new.append(my_decoders)
		decoder_pop = decoders_pop_new

	""" Grab the best recurrent decoders and plot training accuracy and fitness vs generation """
	d_feedback = best_decoders
	evaluate([network, Simulator, d_feedback, conn_feedback, readout_info, target_probe, True])

	fit_vs_gen = np.array(fit_vs_gen)
	import matplotlib.pyplot as plt
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(np.arange(0, evo_gen), fit_vs_gen)
	ax1.set(xlabel='Generation', ylabel='Fitness ($\hat{x}$ RMSE)')
	ax1.legend()
	figure.savefig('plots/evolution/feedback_%s.png' % id(bio_probe))

	np.savez(training_dir+training_file,
		d_feedback=d_feedback)

	return d_feedback