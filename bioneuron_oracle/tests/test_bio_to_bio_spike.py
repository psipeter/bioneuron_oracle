import numpy as np
import nengo
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, TrainedSolver, spike_train

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
	freq_train = 1
	freq_test = 1
	seed_train = 1
	seed_test = 1

	dim = 1
	reg = 0.1
	t_final = 1.0
	cutoff = 0.1
	transform = -0.5

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
		'evo_t_final' :t_final,
	}

	def sim(
		w_pre_bio,
		w_bio_bio2,
		d_readout,
		d_readout2,
		evo_params,
		readout = 'LIF',
		signal='sinusoids',
		t_final=1.0,
		freq=1,
		seeds=1,
		transform=1,
		train=False,
		plot=False):

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
			bio2 = nengo.Ensemble(
				n_neurons=bio2_neurons,
				dimensions=dim,
				seed=bio2_seed,
				neuron_type=BahlNeuron(),
				# neuron_type=nengo.LIF(),
				radius=bio2_radius,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				label='bio2')
			lif2 = nengo.Ensemble(
				n_neurons=bio2.n_neurons,
				dimensions=bio2.dimensions,
				seed=bio2.seed,
				max_rates=nengo.dists.Uniform(min_rate, max_rate),
				# radius=bio.radius,
				neuron_type=nengo.LIF(),
				label='lif2')
			oracle = nengo.Node(size_in=dim, label='oracle')
			oracle2 = nengo.Node(size_in=dim, label='oracle2')
			temp = nengo.Node(size_in=dim, label='temp')
			temp2 = nengo.Node(size_in=dim, label='temp2')

			pre_bio_solver = TrainedSolver(weights_bio = w_pre_bio)
			bio_bio2_solver = TrainedSolver(weights_bio = w_bio_bio2)

			nengo.Connection(stim, pre, synapse=None)
			pre_bio = nengo.Connection(pre, bio,
				seed=conn_seed,
				synapse=tau,
				transform=1,
				trained_weights=True,
				solver = pre_bio_solver,
				n_syn=n_syn)
			nengo.Connection(pre, lif,
				synapse=tau,
				transform=1)
			bio_bio2 = nengo.Connection(bio, bio2,
				seed=conn2_seed,
				synapse=tau,
				transform=transform,
				trained_weights=True,
				solver = bio_bio2_solver,
				n_syn=n_syn)
			nengo.Connection(lif, lif2,
				synapse=tau,
				transform=transform)
			nengo.Connection(stim, oracle,
				synapse=tau,
				transform=1)
			nengo.Connection(oracle, oracle2,
				synapse=tau,
				transform=transform)
			conn_lif = nengo.Connection(lif, temp,
				synapse=tau,
				solver=nengo.solvers.LstsqL2(reg=reg))
			conn_lif2 = nengo.Connection(lif2, temp2,
				synapse=tau,
				# transform=transform,
				solver=nengo.solvers.LstsqL2(reg=reg))

			probe_stim = nengo.Probe(stim, synapse=None)
			probe_pre = nengo.Probe(pre, synapse=tau_readout)
			probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
			probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
			probe_oracle = nengo.Probe(oracle, synapse=tau_readout)
			probe_lif2 = nengo.Probe(lif2, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
			probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
			probe_lif2_spikes = nengo.Probe(lif2.neurons, 'spikes')
			probe_oracle2 = nengo.Probe(oracle2, synapse=tau_readout)


		"""
		Perform spike-match training on the pre-bio weights
		"""
		if train:
			network = spike_train(network, evo_params, plots=False)
			w_pre_bio_new = pre_bio.solver.weights_bio
			w_bio_bio2_new = bio_bio2.solver.weights_bio
		else:
			w_pre_bio_new = w_pre_bio
			w_bio_bio2_new = w_bio_bio2

		"""
		Run the simulation with the new w_pre_bio
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_final)
		lpf = nengo.Lowpass(tau_readout)
		act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt)
		act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt)
		act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt)
		act_lif2 = lpf.filt(sim.data[probe_lif2_spikes], dt=dt)
		x_target = sim.data[probe_oracle][:,0]
		x_target2 = sim.data[probe_oracle2][:,0]

		"""
		Calculate readout decoders by either
			- grabbing the ideal decoders from the LIF population, OR
			- applying the oracle method (simulate, collect spikes and target, solver)
		"""
		if readout == 'LIF':
			d_readout_new = sim.data[conn_lif].weights.T
			d_readout2_new = sim.data[conn_lif2].weights.T
			# print d_readout_new, d_readout2_new
			# print np.sum(act_bio), np.sum(act_bio2), np.sum(act_lif), np.sum(act_lif2)
		elif readout == 'oracle':
			d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, sim.data[probe_oracle])[0]
			d_readout2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio2, sim.data[probe_oracle2])[0]

		# from nengo.base import ObjView
		# def deref_objview(o):  # handle slicing
		# 	return o.obj if isinstance(o, ObjView) else o
		# for conn in network.connections:
		# 	conn_post = deref_objview(conn.post)
		# 	if not hasattr(conn_post, 'neuron_type'): continue  # skip nodes
		# 	post_bioneuron = isinstance(conn_post.neuron_type, BahlNeuron)
		# 	post_LIF = isinstance(conn_post.neuron_type, nengo.LIF)
		# 	if post_bioneuron:
		# 		print conn, np.sum(conn.solver.weights_bio)
		# 	if post_LIF:
		# 		print conn, np.sum(sim.data[conn].weights.T)
		# print 'test bio', np.sum(pre_bio.solver.weights_bio)
		# print 'test bio2', np.sum(bio_bio2.solver.weights_bio)
		"""
		Use the old readout decoders to estimate the bioneurons' outputs for plotting
		"""
		xhat_bio = np.dot(act_bio, d_readout)[:,0]
		xhat_lif = np.dot(act_lif, d_readout)[:,0]
		rmse_bio = rmse(x_target, xhat_bio)
		rmse_lif = rmse(x_target, xhat_lif)
		xhat_bio2 = np.dot(act_bio2, d_readout2)[:,0]
		xhat_lif2 = np.dot(act_lif2, d_readout2)[:,0]
		rmse_bio2 = rmse(x_target2, xhat_bio2)
		rmse_lif2 = rmse(x_target2, xhat_lif2)

		if plot:
			plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
			plt.plot(sim.trange(), xhat_lif, label='lif, rmse=%.5f' % rmse_lif)
			plt.plot(sim.trange(), x_target, label='oracle')
			plt.plot(sim.trange(), xhat_bio2, label='bio2, rmse=%.5f' % rmse_bio2)
			plt.plot(sim.trange(), xhat_lif2, label='lif2, rmse=%.5f' % rmse_lif2)
			plt.plot(sim.trange(), x_target2, label='oracle2')
			# plt.plot(sim.trange(), act_bio[:,0:1], label='bio1')
			# plt.plot(sim.trange(), act_lif[:,0:1], label='lif1')
			# plt.plot(sim.trange(), act_bio2[:,0:3], label='bio2')
			# plt.plot(sim.trange(), act_lif2[:,0:3], label='lif2')
			plt.xlabel('time (s)')
			plt.ylabel('$\hat{x}(t)$')
			plt.legend()

		# """
		# Plot tuning curves with the trained weights
		# """
		# if plot:
		# 	n_eval_points = 20
		# 	for i in range(4):
		# 		x_dot_e_bio = np.dot(
		# 			sim.data[probe_oracle],  # approximation to x_input
		# 			bio2.encoders[i])
		# 		x_dot_e_lif = np.dot(
		# 			sim.data[probe_oracle],
		# 			sim.data[lif2].encoders[i])
		# 		x_dot_e_vals_trained_bio = np.linspace(
		# 			np.min(x_dot_e_bio),
		# 			np.max(x_dot_e_bio), 
		# 			num=n_eval_points)
		# 		x_dot_e_vals_lif = np.linspace(
		# 			np.min(x_dot_e_lif),
		# 			np.max(x_dot_e_lif), 
		# 			num=n_eval_points)
		# 		Hz_mean_trained_bio = np.zeros((x_dot_e_vals_trained_bio.shape[0]))
		# 		Hz_stddev_trained_bio = np.zeros_like(Hz_mean_trained_bio)
		# 		Hz_mean_lif = np.zeros((x_dot_e_vals_lif.shape[0]))
		# 		Hz_stddev_lif = np.zeros_like(Hz_mean_lif)

		# 		for xi in range(x_dot_e_vals_trained_bio.shape[0] - 1):
		# 			ts_greater = np.where(x_dot_e_vals_trained_bio[xi] < sim.data[probe_oracle])[0]
		# 			ts_smaller = np.where(sim.data[probe_oracle] < x_dot_e_vals_trained_bio[xi + 1])[0]
		# 			ts = np.intersect1d(ts_greater, ts_smaller)
		# 			if ts.shape[0] > 0: Hz_mean_trained_bio[xi] = np.average(act_bio2[ts, i])
		# 			if ts.shape[0] > 1: Hz_stddev_trained_bio[xi] = np.std(act_bio2[ts, i])
		# 		for xi in range(x_dot_e_vals_lif.shape[0] - 1):
		# 			ts_greater = np.where(x_dot_e_vals_lif[xi] < sim.data[probe_oracle])[0]
		# 			ts_smaller = np.where(sim.data[probe_oracle] < x_dot_e_vals_lif[xi + 1])[0]
		# 			ts = np.intersect1d(ts_greater, ts_smaller)
		# 			if ts.shape[0] > 0: Hz_mean_lif[xi] = np.average(act_lif2[ts, i])
		# 			if ts.shape[0] > 1: Hz_stddev_lif[xi] = np.std(act_lif2[ts, i])

		# 		rmse_tuning_curve_trained = rmse(Hz_mean_trained_bio[:-2], Hz_mean_lif[:-2])
		# 		lifplot = plt.plot(x_dot_e_vals_lif[:-2], Hz_mean_lif[:-2],
		# 			label='ideal',
		# 			ls='dotted')
		# 		plt.fill_between(x_dot_e_vals_lif[:-2],
		# 			Hz_mean_lif[:-2]+Hz_stddev_lif[:-2],
		# 			Hz_mean_lif[:-2]-Hz_stddev_lif[:-2],
		# 			alpha=0.25,
		# 			facecolor=lifplot[0].get_color())
		# 		trained_bioplot = plt.plot(x_dot_e_vals_trained_bio[:-2], Hz_mean_trained_bio[:-2],
		# 			# marker='o',
		# 			ls='solid',
		# 			color=lifplot[0].get_color(),
		# 			label='trained bioneuron %s, RMSE=%.1f' % (i, rmse_tuning_curve_trained))
		# 		plt.fill_between(x_dot_e_vals_trained_bio[:-2],
		# 			Hz_mean_trained_bio[:-2]+Hz_stddev_trained_bio[:-2],
		# 			Hz_mean_trained_bio[:-2]-Hz_stddev_trained_bio[:-2],
		# 			alpha=0.75,
		# 			facecolor=lifplot[0].get_color())
		# 	plt.ylim(ymin=0)
		# 	plt.xlabel('$x \cdot e$')
		# 	plt.ylabel('firing rate')
		# 	plt.title('Tuning Curves')
		# 	plt.legend()

		return w_pre_bio_new, w_bio_bio2_new, d_readout_new, d_readout2_new, rmse_bio2


	"""
	Run the test
	"""
	weight_dir = '/home/pduggins/bioneuron_oracle/bioneuron_oracle/tests/weights/'
	weight_filename = 'w_bio_to_bio_1d_pre_bio.npz'
	weight2_filename = 'w_bio_to_bio_1d_bio_bio2.npz'
	try:
		w_pre_bio_init = np.load(weight_dir+weight_filename)['weights_bio']
		w_bio_bio2_init = np.load(weight_dir+weight2_filename)['weights_bio']
		to_train = False
	except IOError:
		w_pre_bio_init = np.zeros((bio_neurons, pre_neurons, n_syn))
		w_bio_bio2_init = np.zeros((bio2_neurons, bio_neurons, n_syn))
		to_train = True
	d_readout_init = np.zeros((bio_neurons, dim))
	d_readout2_init = np.zeros((bio2_neurons, dim))

	w_pre_bio_new, w_bio_bio2_new, d_readout_new, d_readout2_new, rmse_bio2 = sim(
		w_pre_bio=w_pre_bio_init,
		w_bio_bio2=w_bio_bio2_init,
		d_readout=d_readout_init,
		d_readout2=d_readout2_init,
		evo_params=evo_params,
		readout='oracle',
		signal='sinusoids',
		freq=freq_train,
		seeds=seed_train,
		transform=transform,
		t_final=t_final,
		train=to_train,
		plot=False)
	w_pre_bio_extra, w_bio_bio2_extra, d_readout_extra, d_readout2_extra, rmse_bio2 = sim(
		w_pre_bio=w_pre_bio_new,
		w_bio_bio2=w_bio_bio2_new,
		d_readout=d_readout_new,
		d_readout2=d_readout2_new,
		evo_params=evo_params,
		readout='oracle',
		signal='sinusoids',
		freq=freq_test,
		seeds=seed_test,
		transform=transform,
		t_final=t_final,
		train=False,
		plot=True)

	np.savez(weight_dir+weight_filename, weights_bio=w_pre_bio_new)
	np.savez(weight_dir+weight2_filename, weights_bio=w_bio_bio2_new)

	assert rmse_bio2 < cutoff
