# import multiprocessing as mp
from pathos import multiprocessing as mp

import numpy as np

import copy

import nengo
from nengo.utils.numpy import rmse
from nengo.base import ObjView

from bioneuron_oracle import BahlNeuron, TrainedSolver

__all__ = ['spike_train']

def deref_objview(o):  # handle slicing
	return o.obj if isinstance(o, ObjView) else o

def spike_train(network, params, method="1-N", plots=False):
	
	dt = params['dt']
	tau_readout = params['tau_readout']
	n_processes = params['n_processes']
	popsize = params['popsize']
	generations = params['generations']
	delta_w = params['delta_w']
	w_0 = params['w_0']
	evo_seed = params['evo_seed']
	evo_t_final = params['evo_t_final']
	sim_seed = params['sim_seed']

	"""
	Evaluate the fitness of a network using the specified weights,
	where fitness = sum of rmse(act_bio, act_ideal) for each bio ensemble
	"""
	def evaluate(inputs):
		w_bio_p = inputs[0]  # new weights
		network = inputs[1]  # original network
		ens = inputs[2]  # ens to be trained
		bio_probes = inputs[3]  # dict of probes of bioensembles
		ideal_probes = inputs[4]  # dict of probes of ideal ensembles
		w_original = {}  # original weights

		"""
		replace all bioensembles with corresponding LIF ensembles
		except for the one that is being trained (e.g. so that bio1-bio2
		gives bio2 some spikes before bio1 is trained)
		"""
		ens_changed = {}
		for ens_test in network.ensembles:
			if (isinstance(ens_test.neuron_type, BahlNeuron)
					and ens_test is not ens):
				ens_changed[ens_test] = True
				with network:
					ens_test = ideal_ens[ens_test]

		for conn in network.connections:
			conn_post = deref_objview(conn.post)
			# select the ens to be trained
			if conn_post == ens and conn in w_bio_p:  # and conn not in saved_weights
				# save original weights
				w_original[conn] = conn.solver.weights_bio
				# update weights for this fitness evaluation
				conn.solver.weights_bio = w_bio_p[conn]

		with nengo.Simulator(network, dt=dt,
							 progress_bar=False, seed=sim_seed) as sim:
			sim.run(evo_t_final)

		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[bio_probes[ens]], dt=dt)
		act_lif = lpf.filt(sim.data[ideal_probes[ens]], dt=dt)

		rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
							 for i in range(act_bio.shape[1])])

		"""
		reset network back to original weights and neuron types
		"""
		for ens_test in network.ensembles:
			if ens_test in ens_changed:
				with network:
					ens_test= bio_ens[ens_test]
		for conn in network.connections:
			if conn_post == ens and conn in w_bio_p:
				conn.solver.weights_bio = w_original[conn]

		# return rmse_act  # ensemble-by-ensemble training
		return rmses_act  # neuron-by-neuron training

	"""
	Add a spike probe and a ideal lif ensemble for each bioensemble
	"""
	bio_ens = {}
	bio_probes = {}
	ideal_ens = {}
	ideal_probes = {}
	for ens in network.ensembles:
		if isinstance(ens.neuron_type, BahlNeuron):
			with network:
				bio_ens[ens] = ens
				bio_probes[ens] = nengo.Probe(ens.neurons, 'spikes')
				ideal_ens[ens] = nengo.Ensemble(
							 n_neurons=ens.n_neurons, dimensions=ens.dimensions,
							 seed=ens.seed, neuron_type=nengo.LIF(),
							 encoders=ens.encoders, gain=ens.gain, bias=ens.bias,
							 max_rates=ens.max_rates, intercepts=ens.intercepts,
							 label='temp_sub_spike_match')
				ideal_probes[ens] = nengo.Probe(ideal_ens[ens].neurons, 'spikes')

			"""
			connect everything going into ens to ideal_ens[ens]
			so that the ideal ensemble sees the same input as the bioensemble
			"""
			for conn in network.connections:
				if conn.post == ens:
					with network:
						nengo.Connection(conn.pre, ideal_ens[ens],
							synapse=conn.synapse, transform=conn.transform,
							solver=nengo.solvers.LstsqL2())  # TODO: add other conn features?


	pool = mp.ProcessingPool(nodes=n_processes)
	rng = np.random.RandomState(seed=evo_seed)

	# all_weights = {}
	for ens in bio_ens.iterkeys():  # For each bioensemble in the network:
		"""
		Figure out how many weight matrices need to be trained,
		then initialize a random weight matrix of the propper shape for each
		"""
		w_pop = [dict() for p in range(popsize)]
		for conn in network.connections:
			conn_pre = deref_objview(conn.pre)
			conn_post = deref_objview(conn.post)
			if (hasattr(conn, 'trained_weights')
					and conn.trained_weights == True
					and conn_post == ens):
					# and conn not in saved_weights):
				n_bio = conn_post.n_neurons
				# if not continue_training:  # starting a new optimization
				#	 conn.solver = TrainedSolver(
				#		 weights_bio = np.zeros((
				#			 conn_post.n_neurons, conn_pre.n_neurons, conn.n_syn)))
				#	 # heuristics for scaling w_0 and delta_w for conn statistics
				#	 w_0 *= 100.0 / conn.pre.n_neurons
				#	 for p in range(popsize):
				#		 w_pop[p][conn] = rng.uniform(-w_0, w_0,
				#			 size=conn.solver.weights_bio.shape)
				# else:  # starting from a specified weights_bio on conn's trained_solver
				for p in range(popsize):
					w_pop[p][conn] = conn.solver.weights_bio + \
						rng.normal(0, delta_w,
							size=conn.solver.weights_bio.shape)
		"""
		Train the weights using some evolutionary strategy
		"""
		# fit_vs_gen = np.zeros((generations))  # ensemble-by-ensemble training
		fit_vs_gen = np.zeros((generations, n_bio))  # neuron-by-neuron training
		for g in range(generations):
			inputs = [[w_pop[p], network, ens, bio_probes, ideal_probes]
					  for p in range(popsize)]
			# fitnesses = np.array([evaluate(inputs[0]), evaluate(inputs[1])])  # debugging
			fitnesses = np.array(pool.map(evaluate, inputs))

			"""
			Find the evo individual with the lowest fitness
			"""
			# ensemble-by-ensemble training
			# fit_vs_gen[g] = np.min(fitnesses)
			# w_best = copy.copy(w_pop[np.argmin(fitnesses)])
			# neuron-by-neuron training
			w_best = dict()
			for conn in w_pop[0].iterkeys():
				w_best[conn] = np.zeros_like(w_pop[0][conn])
				for nrn in range(fit_vs_gen.shape[1]):
					fit_vs_gen[g, nrn] = fitnesses[np.argmin(fitnesses[:,nrn]),nrn]
					w_best[conn][nrn] = w_pop[np.argmin(fitnesses[:,nrn])][conn][nrn]

			# Copy and mutate the best weight matrices for each new evo individual
			w_pop_new=[]
			for p in range(popsize):
				w_pop_new.append(dict())
				for conn in w_pop[p].iterkeys():
					decay = np.exp(-g / 10)  # decay of mutation rate over generations
					w_mutate = rng.normal(0, delta_w*decay, size=w_best[conn].shape)
					w_pop_new[p][conn] = w_best[conn] + w_mutate
			w_pop = copy.copy(w_pop_new)

			# update the connections' solvers in the network
			for conn in network.connections:
				if conn in w_pop[0]:
					conn.solver.weights_bio = w_best[conn]

		# for conn in network.connections:
		#	 if conn in w_pop[0]:
		#		 all_weights[conn] = conn.solver.weights_bio

		if plots:
			# import matplotlib.pyplot as plt
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(np.arange(0,generations), fit_vs_gen)
			ax1.set(xlabel='Generation', ylabel='Fitness (RMSE_act)')
			ax1.legend()
			figure.savefig('plots/fitness_vs_generation_%s.png' % ens)

	# TODO: save all connection weights
	# if name is None: name = str(network)  # untested
	# np.savez('weights/%s.npz' % name, all_weights = all_weights)

	return network


	# TODO: implement loading previously trained weights
	# problem: conn can't be used as a dictionary key becuause it changes
	# every time the network is rebuilt, even with identical seeds. So I can't
	# build the test network, run this function to train, build it again,
	# and check if the new conns are in the old conn list (without explicit labels)
	# try:
	#	 saved_weights = np.load('weights/%s.npz' % name)['all_weights']
	#	 print 'loading saved weights...'
	#	 print saved_weights
	#	 for conn in network.connections:
	#		 if (hasattr(conn, 'trained_weights')
	#				 and conn.trained_weights == True):
	#			 print conn
	#			 print saved_weights[conn]
	#			 conn.solver.weights_bio = saved_weights[conn]
	#	 return network
	# except IOError:
	#	 saved_weights = {}
	#	 print 'training weights...'