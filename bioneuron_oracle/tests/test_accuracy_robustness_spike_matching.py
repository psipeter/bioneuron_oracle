from functools32 import lru_cache

from seaborn import set_palette, color_palette, tsplot

import numpy as np

import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.utils.numpy import rmse

from bioneuron_oracle import (BahlNeuron, prime_sinusoids, step_input,
                              TrainedSolver, spike_match_train)

def test_accuracy_vs_n_neurons(Simulator, plt):
    """
    Simulate a feedforward network with bioneurons whose input weights
    have been trained by with a spike matching approach(), varying
    the number of bio_neurons and measuring RMSE of xhat_bio
    """

    bio_neurons = np.array([5,10,15,20])
    n_avg = 3
    pre_seed = np.random.randint(0,9009,size=(bio_neurons.shape[0], n_avg))
    bio_seed = np.random.randint(0,9009,size=(bio_neurons.shape[0], n_avg))
    conn_seed = np.random.randint(0,9009,size=(bio_neurons.shape[0], n_avg))

    # Evolutionary Parameters
    evo_params = {
        'dt_nengo': 0.001,
        'tau_nengo': 0.05,
        'n_processes': 10,
        'popsize': 10,
        'generations' :10,
        'w_0': 1e-3,
        'delta_w' :1e-4,
        'evo_seed' :9,
        'evo_t_final' :1.0,
        'evo_signal': 'prime_sinusoids',
        'evo_max_freq': 5.0,
        'evo_signal_seed': 234,
        'evo_cutoff' :50.0,
    }

    def sim(Simulator, plt, evo_params,
            bio_neurons, bio_seed, pre_seed, conn_seed):

        # Nengo Parameters
        pre_neurons = 100
        tau_nengo = 0.05
        tau_neuron = 0.05
        dt_nengo = 0.001
        min_rate = 150
        max_rate = 200
        t_final = 1.0
        dim = 1
        n_syn = 1
        signal = 'prime_sinusoids'

        with nengo.Network() as network:

            if signal == 'prime_sinusoids':
                stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
            elif signal == 'step_input':
                stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
            elif signal == 'white_noise':
                stim = nengo.Node(lambda t: equalpower(
                                      t, dt_nengo, t_final, max_freq, dim,
                                      mean=0.0, std=1.0, seed=signal_seed))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            temp = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct()) 

            trained_solver = TrainedSolver(
                weights_bio = np.zeros((bio.n_neurons, pre.n_neurons, n_syn)))
            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, lif, synapse=tau_nengo)
            nengo.Connection(pre, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver = trained_solver,
                             n_syn=n_syn)
            nengo.Connection(stim, direct, synapse=tau_nengo)
            conn_ideal_out = nengo.Connection(lif, temp, synapse=tau_nengo,
                             solver=nengo.solvers.LstsqL2())

            probe_pre = nengo.Probe(pre, synapse=tau_nengo)
            probe_lif = nengo.Probe(lif)
            probe_direct = nengo.Probe(direct, synapse=None)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

        network = spike_match_train(network, method="1-N",
                                    params=evo_params, plots=False)
        
        with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
        oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
        ideal_solver = nengo.solvers.LstsqL2(reg=0.01)
        oracle_decoders = oracle_solver(act_bio, sim.data[probe_direct])[0]
        ideal_decoders = sim.data[conn_ideal_out].weights.T
        xhat_bio_oracle = np.dot(act_bio, oracle_decoders)
        xhat_bio_ideal = np.dot(act_bio, ideal_decoders)
        xhat_lif_ideal = np.dot(act_lif, ideal_decoders)
        rmse_bio_oracle = rmse(sim.data[probe_direct], xhat_bio_oracle)
        rmse_bio_ideal = rmse(sim.data[probe_direct], xhat_bio_ideal)
        rmse_lif = rmse(sim.data[probe_direct], xhat_lif_ideal)
        # rmse_lif = rmse(sim.data[probe_direct], sim.data[probe_lif])

        return rmse_bio_oracle, rmse_lif

    rmse_bio_mean = []
    rmse_bio_std = []
    rmse_lif_mean = []
    rmse_lif_std = []
    for i in range(bio_neurons.shape[0]):
        rmse_bio_i = []
        rmse_lif_i = []
        for j in range(bio_seed.shape[1]):
            rmse_bio, rmse_lif = sim(Simulator, plt, evo_params,
                bio_neurons[i], bio_seed[i,j], pre_seed[i,j], conn_seed[i,j])
            rmse_bio_i.append(rmse_bio)
            rmse_lif_i.append(rmse_lif)
        rmse_bio_mean.append(np.mean(rmse_bio_i))
        rmse_bio_std.append(np.std(rmse_bio_i))
        rmse_lif_mean.append(np.mean(rmse_lif_i))
        rmse_lif_std.append(np.std(rmse_lif_i))
    rmse_bio_mean = np.array(rmse_bio_mean)
    rmse_bio_std = np.array(rmse_bio_std)
    rmse_lif_mean = np.array(rmse_lif_mean)
    rmse_lif_std = np.array(rmse_lif_std)

    plt.subplot(1, 1, 1)
    plt.errorbar(bio_neurons, rmse_bio_mean, yerr=rmse_bio_std, label='bio')
    plt.errorbar(bio_neurons, rmse_lif_mean, yerr=rmse_lif_std, label='lif')
    plt.plot(bio_neurons, 1.0 / bio_neurons, label='theory')
    plt.xlabel('n_neurons')
    plt.ylabel('RMSE ($\hat{x}(t)$, $x(t)$)')
    # plt.title('decode')
    plt.legend()  # prop={'size':8}


def test_accuracy_vs_generations(Simulator, plt):
    """
    Simulate a feedforward network with bioneurons whose input weights
    have been trained by with a spike matching approach(), varying
    the number of bio_neurons and measuring RMSE of xhat_bio
    """

    generations = np.array([10,20,30,40])
    n_avg = 3
    pre_seed = np.random.randint(0,9009,size=(generations.shape[0], n_avg))
    bio_seed = np.random.randint(0,9009,size=(generations.shape[0], n_avg))
    conn_seed = np.random.randint(0,9009,size=(generations.shape[0], n_avg))

    # Evolutionary Parameters
    evo_params = {
        'dt_nengo': 0.001,
        'tau_nengo': 0.05,
        'n_processes': 10,
        'popsize': 10,
        'generations' :10,
        'w_0': 1e-3,
        'delta_w' :1e-4,
        'evo_seed' :9,
        'evo_t_final' :0.1,
        'evo_signal': 'prime_sinusoids',
        'evo_max_freq': 5.0,
        'evo_signal_seed': 234,
        'evo_cutoff' :50.0,
    }

    def sim(Simulator, plt, evo_params,
            generations, bio_seed, pre_seed, conn_seed):

        # Nengo Parameters
        pre_neurons = 100
        bio_neurons = 20
        tau_nengo = 0.05
        tau_neuron = 0.05
        dt_nengo = 0.001
        min_rate = 150
        max_rate = 200
        t_final = 1.0
        dim = 1
        n_syn = 1
        signal = 'prime_sinusoids'
        evo_params['generations'] = generations

        with nengo.Network() as network:

            if signal == 'prime_sinusoids':
                stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
            elif signal == 'step_input':
                stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))
            elif signal == 'white_noise':
                stim = nengo.Node(lambda t: equalpower(
                                      t, dt_nengo, t_final, max_freq, dim,
                                      mean=0.0, std=1.0, seed=signal_seed))

            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                 seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                 seed=bio_seed, neuron_type=BahlNeuron(),
                                 max_rates=nengo.dists.Uniform(min_rate, max_rate),
                                 intercepts=nengo.dists.Uniform(-1, 1))
            lif = nengo.Ensemble(n_neurons=bio.n_neurons, dimensions=bio.dimensions,
                                 seed=bio.seed, neuron_type=nengo.LIF(),
                                 max_rates=bio.max_rates, intercepts=bio.intercepts)
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct())
            temp = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                    neuron_type=nengo.Direct()) 

            trained_solver = TrainedSolver(
                weights_bio = np.zeros((bio.n_neurons, pre.n_neurons, n_syn)))
            nengo.Connection(stim, pre, synapse=None)
            nengo.Connection(pre, lif, synapse=tau_nengo)
            nengo.Connection(pre, bio,
                             seed=conn_seed,
                             synapse=tau_neuron,
                             trained_weights=True,
                             solver = trained_solver,
                             n_syn=n_syn)
            nengo.Connection(stim, direct, synapse=tau_nengo)
            conn_ideal_out = nengo.Connection(lif, temp, synapse=tau_nengo,
                             solver=nengo.solvers.LstsqL2())

            probe_pre = nengo.Probe(pre, synapse=tau_nengo)
            probe_lif = nengo.Probe(lif)
            probe_direct = nengo.Probe(direct, synapse=None)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')

        network = spike_match_train(network, method="1-N",
                                    params=evo_params, plots=False)
        
        with Simulator(network, dt=dt_nengo, progress_bar=False) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_lif = lpf.filt(sim.data[probe_lif_spikes], dt=dt_nengo)
        oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
        ideal_solver = nengo.solvers.LstsqL2(reg=0.01)
        oracle_decoders = oracle_solver(act_bio, sim.data[probe_direct])[0]
        ideal_decoders = sim.data[conn_ideal_out].weights.T
        xhat_bio_oracle = np.dot(act_bio, oracle_decoders)
        xhat_bio_ideal = np.dot(act_bio, ideal_decoders)
        xhat_lif_ideal = np.dot(act_lif, ideal_decoders)
        rmse_bio_oracle = rmse(sim.data[probe_direct], xhat_bio_oracle)
        rmse_bio_ideal = rmse(sim.data[probe_direct], xhat_bio_ideal)
        rmse_lif = rmse(sim.data[probe_direct], xhat_lif_ideal)
        # rmse_lif = rmse(sim.data[probe_direct], sim.data[probe_lif])

        return rmse_bio_oracle, rmse_lif

    rmse_bio_mean = []
    rmse_bio_std = []
    rmse_lif_mean = []
    rmse_lif_std = []
    for i in range(generations.shape[0]):
        rmse_bio_i = []
        rmse_lif_i = []
        for j in range(bio_seed.shape[1]):
            rmse_bio, rmse_lif = sim(Simulator, plt, evo_params,
                generations[i], bio_seed[i,j], pre_seed[i,j], conn_seed[i,j])
            rmse_bio_i.append(rmse_bio)
            rmse_lif_i.append(rmse_lif)
        rmse_bio_mean.append(np.mean(rmse_bio_i))
        rmse_bio_std.append(np.std(rmse_bio_i))
        rmse_lif_mean.append(np.mean(rmse_lif_i))
        rmse_lif_std.append(np.std(rmse_lif_i))
    rmse_bio_mean = np.array(rmse_bio_mean)
    rmse_bio_std = np.array(rmse_bio_std)
    rmse_lif_mean = np.array(rmse_lif_mean)
    rmse_lif_std = np.array(rmse_lif_std)

    plt.subplot(1, 1, 1)
    plt.errorbar(generations, rmse_bio_mean, yerr=rmse_bio_std, label='bio')
    plt.errorbar(generations, rmse_lif_mean, yerr=rmse_lif_std, label='lif')
    plt.xlabel('generations')
    plt.ylabel('RMSE ($\hat{x}(t)$, $x(t)$)')
    # plt.title('decode')
    plt.legend()  # prop={'size':8}

    assert True