import numpy as np
import nengo
import hyperopt
import json
from nengo.utils.numpy import rmse
from bioneuron_oracle import BahlNeuron, OracleSolver, get_signals
import matplotlib.pyplot as plt
import seaborn as sns

pre_neurons = 100
bio_neurons = 100
max_evals = 10
dim = 1
tau = 0.01
tau_readout = 0.01
dt = 0.001
min_rate = 150
max_rate = 200
radius = 1
freq = 1
max_freq = 5
rms = 0.5
t_transient = 1.0
t_train = 1.0  # 10
signal_type = 'sinusoids' #'white_noise'
seed = 2
reg = 0.01

def evaluate(P):
	signal, _, _ = get_signals(
		signal_type, seed, seed, freq, seed, t_transient,
		t_train, max_freq, rms, tau, dt)

	with nengo.Network(seed=seed) as model:
		stim = nengo.Node(lambda t: signal[int(t/dt)])
		pre = nengo.Ensemble(
			n_neurons=pre_neurons,
			dimensions=dim,
			seed=seed,
			neuron_type=nengo.LIF(),
			label='pre')
		bio = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim,
			seed=seed,
			neuron_type=BahlNeuron(),
			# neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05),
			# neuron_type=nengo.LIF(),
			max_rates=nengo.dists.Uniform(min_rate, max_rate),
			label='bio')
		alif = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim,
			seed=seed,
			# neuron_type=BahlNeuron(),
			neuron_type=nengo.AdaptiveLIF(tau_n=P['tau_n'], inc_n=P['inc_n']),
			# neuron_type=nengo.LIF(),
			max_rates=nengo.dists.Uniform(min_rate, max_rate),
			label='alif')
		oracle = nengo.Node(size_in=dim)

		nengo.Connection(stim, pre, synapse=None)
		nengo.Connection(pre, bio,
			weights_bias_conn=True,
			seed=seed,
			synapse=tau)
		nengo.Connection(pre, alif,
			synapse=tau,
			seed=seed)
		nengo.Connection(stim, oracle, synapse=nengo.Lowpass(tau))

		probe_bio = nengo.Probe(bio.neurons, 'spikes', synapse=nengo.Lowpass(tau_readout))
		probe_alif = nengo.Probe(alif.neurons, 'spikes', synapse=nengo.Lowpass(tau_readout))
		probe_oracle = nengo.Probe(oracle, synapse=nengo.Lowpass(tau_readout))
		probe_alif_value = nengo.Probe(alif, synapse=nengo.Lowpass(tau_readout))

	with nengo.Simulator(model, dt=dt, seed=seed) as sim:
		sim.run(t_transient + t_train)

	act_bio = sim.data[probe_bio]
	act_alif = sim.data[probe_alif]
	# loss = rmse(act_bio, act_alif)
	target = sim.data[probe_oracle]
	d_bio = nengo.solvers.LstsqL2(reg=reg)(act_bio, target)[0]
	d_alif = nengo.solvers.LstsqL2(reg=reg)(act_alif, target)[0]
	xhat_bio = np.dot(act_bio, d_bio)
	xhat_alif = np.dot(act_alif, d_alif)
	# xhat_alif = sim.data[probe_alif_value]
	loss = rmse(xhat_bio, xhat_alif)

	return {'loss': loss, 'status': hyperopt.STATUS_OK,
		'time': sim.trange(),
		'act_bio': act_bio, 'act_alif': act_alif,
		'xhat_bio': xhat_bio, 'xhat_alif': xhat_alif, 'target': target}

tau_n = hyperopt.hp.loguniform('tau_n', -5, -1)
inc_n = hyperopt.hp.loguniform('inc_n', -5, -1)
P = {'tau_n': tau_n, 'inc_n': inc_n}

trials = hyperopt.Trials()

best = hyperopt.fmin(evaluate,
	rstate=np.random.RandomState(seed=seed),
	space=P,
	algo=hyperopt.tpe.suggest,
	max_evals=max_evals,
	trials=trials
	)

losses = np.array([t['result']['loss'] for t in trials])
times = np.array([t['result']['time'] for t in trials])
acts_bio = np.array([t['result']['act_bio'] for t in trials])
acts_alif = np.array([t['result']['act_alif'] for t in trials])
xhats_bio = np.array([t['result']['xhat_bio'] for t in trials])
xhats_alif = np.array([t['result']['xhat_alif'] for t in trials])
targets = np.array([t['result']['target'] for t in trials])
time = times[np.argmin(losses)]
act_bio = acts_bio[np.argmin(losses)]
act_alif = acts_alif[np.argmin(losses)]
xhat_bio = xhats_bio[np.argmin(losses)]
xhat_alif = xhats_alif[np.argmin(losses)]
target = targets[np.argmin(losses)]

figure1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(time, act_bio, label='bio')
# ax1.plot(time, act_alif, label='alif')
ax1.plot(time, xhat_bio, label='bio')
ax1.plot(time, xhat_alif, label='alif')
ax1.plot(time, target, label='target')
# ax1.set(xlabel='time', ylabel='activity', title=best)
ax1.set(xlabel='time', ylabel='$\hat{x}$', title=best)
ax1.legend()

# figure2, ax2 = plt.subplots(1,1)
ax2.plot(range(len(trials)),losses)
ax2.set(xlabel='trial', ylabel='total loss', xlim=((t_transient,t_transient+1.0)))
plt.show()