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
		w_bypassed = {}  # bonnections where pre/post has been subbed for LIF

		"""
		Replace all bioensembles with corresponding LIF ensembles
		so that bio1-bio2 gives bio2 some spikes before bio1 is trained
		To do this, go through each connection.
			-If it's a connection into a bioneuron, connect pre to ideal_ens[post]
			-If it's a connection out of a bioneuron, connect ideal_ens[pre] to post
			-If it's both, connect ideal_ens[pre] to ideal_ens[post]
		Store the created connections in temp_connections
		Store the weight matrices of the connections into bioneurons in w_bypassed
		Store the decoders for connections out of bioneurons (OracleSolvers) in w_bypassed
		Set all connection weights into all bioensemble zero,
			then restore at end of evaluate
		For the trained bioensemble, save the original weights,
			and set the weights according to w_bip_p,
			and create a new connection from ideal_ens[pre] to ens if pre is bioneuron
		"""
		temp_connections = {}
		for conn in network.connections:
			conn_pre = deref_objview(conn.pre)
			conn_post = deref_objview(conn.post)
			if not hasattr(conn_pre, 'neuron_type'): continue  # skip nodes
			if not hasattr(conn_post, 'neuron_type'): continue  # skip nodes
			# if conn in temp_connections:  continue # skip temporary connections
			pre_bioneuron = (isinstance(conn_pre.neuron_type, BahlNeuron) and hasattr(conn.solver, 'weights_bio'))
			post_bioneuron = (isinstance(conn_post.neuron_type, BahlNeuron) and hasattr(conn.solver, 'weights_bio'))
			trained_ens = (conn.post == ens and conn in w_bio_p and hasattr(conn.solver, 'weights_bio'))
			# trained_ens = (conn.post == ens)
			solver = nengo.solvers.LstsqL2()
			with network:
				if (pre_bioneuron and not post_bioneuron):
					temp = nengo.Connection(ideal_ens[conn.pre], conn_post,
								synapse=conn.synapse, transform=conn.transform,
								function=conn.function, solver=solver)
					temp_connections[temp] = 'in'
				if (not pre_bioneuron and post_bioneuron):
					temp = nengo.Connection(conn_pre, ideal_ens[conn.post],
								synapse=conn.synapse, transform=conn.transform,
								function=conn.function, solver=solver)
					temp_connections[temp] = 'out'
					if conn.post != ens:
						w_bypassed[conn] = conn.solver.weights_bio
						conn.solver.weights_bio = np.zeros_like(conn.solver.weights_bio)
				if (pre_bioneuron and post_bioneuron and conn.post != ens):  # todo: force to not create temp1 to temp2
					temp = nengo.Connection(ideal_ens[conn.pre], ideal_ens[conn.post],
								synapse=conn.synapse, transform=conn.transform,
								function=conn.function, solver=solver)
					temp_connections[temp] = 'through'
					if conn.post != ens:
						w_bypassed[conn] = conn.solver.weights_bio
						conn.solver.weights_bio = np.zeros_like(conn.solver.weights_bio)
				if (pre_bioneuron and trained_ens):  # todo: this creates temp1 to temp2 when it shouldn't
					solver_proxy = TrainedSolver(weights_bio = w_bio_p[conn])
					temp = nengo.Connection(ideal_ens[conn.pre], conn.post,
								trained_weights=True, n_syn=conn.n_syn,
								synapse=conn.synapse, transform=conn.transform,
								function=conn.function, solver=solver_proxy)
					temp_connections[temp] = 'proxy-spikes'
					conn.solver.weights_bio = np.zeros_like(w_bio_p[conn])  # only proxy spikes feedforward
				if (not pre_bioneuron and trained_ens):
					w_original[conn] = conn.solver.weights_bio
					conn.solver.weights_bio = w_bio_p[conn]  # update weights 

		# """test"""
		# print 'before'
		# if ens.label == 'bio2':
		# 	for conn in network.connections:
		# 		print conn
		# 		conn_post = deref_objview(conn.post)
		# 		if not hasattr(conn_post, 'neuron_type'): continue  # skip nodes
		# 		post_bioneuron = isinstance(conn_post.neuron_type, BahlNeuron)
		# 		temp_neuron = (conn in temp_connections)
		# 		if post_bioneuron:
		# 			print np.sum(conn.solver.weights_bio)
				# if temp_connections:
				# 	print conn.weights.T

		with nengo.Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(evo_t_final)

		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[bio_probes[ens]], dt=dt)
		act_lif = lpf.filt(sim.data[ideal_probes[ens]], dt=dt)

		rmses_act = np.array([rmse(act_bio[:,i], act_lif[:,i])
							 for i in range(act_bio.shape[1])])

		"""test"""
		# print 'after'
		# # if ens.label == 'bio':
		# for conn in network.connections:
		# 	# print conn
		# 	conn_post = deref_objview(conn.post)
		# 	if not hasattr(conn_post, 'neuron_type'): continue  # skip nodes
		# 	post_bioneuron = isinstance(conn_post.neuron_type, BahlNeuron)
		# 	temp_neuron = (conn in temp_connections)
		# 	if post_bioneuron:
		# 		print 'solver', conn, np.sum(conn.solver.weights_bio)
			# if temp_connections:
			# 	print 'temp', np.sum(sim.data[conn].weights.T)


		"""
		Remove temporary LIF connections and reset network back to original weights
		"""
		network.connections = [conn for conn in network.connections if conn not in temp_connections]
		for conn in network.connections:
			if conn_post == ens and conn in w_bio_p:
				conn.solver.weights_bio = w_original[conn]

		# """test"""
		# if ens.label == 'bio2':
		# 	print 'deleted'
		# 	for conn in network.connections:
		# 		print conn
		# 		conn_post = deref_objview(conn.post)
		# 		if not hasattr(conn_post, 'neuron_type'): continue  # skip nodes
		# 		post_bioneuron = isinstance(conn_post.neuron_type, BahlNeuron)
		# 		if post_bioneuron:
		# 			print np.sum(conn.solver.weights_bio)
		# assert False

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
							 label='temp_sub_spike_match_%s' %ens.label)
				ideal_probes[ens] = nengo.Probe(ideal_ens[ens].neurons, 'spikes')

	pool = mp.ProcessingPool(nodes=n_processes)
	rng = np.random.RandomState(seed=evo_seed)

	# all_weights = {}
	for ens in bio_ens.iterkeys():  # For each bioensemble in the network:
		"""
		Figure out how many weight matrices need to be trained,
		then initialize a random weight matrix of the propper shape for each
		"""
		if ens.label == 'untrained_bio': continue  # todo: hack
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
						rng.normal(0, w_0, size=conn.solver.weights_bio.shape)
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
					# print fitnesses, np.argmin(fitnesses[:,nrn])

			# for p in range(popsize):
			# 	for conn in w_pop[p]:
			# 		print 'wpop', p, np.sum(w_pop[p][conn])
			# for conn in w_best:
			# 	print 'wbest', conn, np.sum(w_best[conn])

			# Copy and mutate the best weight matrices for each new evo individual
			w_pop_new=[]
			for p in range(popsize):
				w_pop_new.append(dict())
				for conn in w_pop[p].iterkeys():
					decay = np.exp(-g / 10.0)  # decay of mutation rate over generations
					w_mutate = rng.normal(0, delta_w*decay, size=w_best[conn].shape)
					w_pop_new[p][conn] = w_best[conn] + w_mutate
			w_pop = copy.copy(w_pop_new)

			# update the connections' solvers in the network
			for conn in network.connections:
				if conn in w_pop[0]:
					conn.solver.weights_bio = w_best[conn]
					# print ens
					# print conn
					# print np.sum(conn.solver.weights_bio)

		# """test"""
		# print 'end'
		# for conn in network.connections:
		# 	print conn
		# 	conn_post = deref_objview(conn.post)
		# 	if not hasattr(conn_post, 'neuron_type'): continue  # skip nodes
		# 	post_bioneuron = isinstance(conn_post.neuron_type, BahlNeuron)
		# 	if post_bioneuron:
		# 		print np.sum(conn.solver.weights_bio)

		# for conn in network.connections:
		#	 if conn in w_pop[0]:
		#		 all_weights[conn] = conn.solver.weights_bio

		if plots:
			import matplotlib.pyplot as plt
			figure, ax1 = plt.subplots(1,1)
			ax1.plot(np.arange(0,generations), fit_vs_gen)
			ax1.set(xlabel='Generation', ylabel='Fitness (RMSE_act)')
			ax1.legend()
			figure.savefig('plots/fitness_vs_generation_%s.png' % ens)

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