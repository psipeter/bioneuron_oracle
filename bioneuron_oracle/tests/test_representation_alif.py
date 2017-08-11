import matplotlib.pyplot as plt
import numpy as np
import nengo

def test_representation_1d(Simulator, plt):

    tau = 0.01
    tau_readout = 0.01
    dt=0.001

    signal = 'sinusoids'
    freq = 10.0
    seed = 1
    transform = 1.0
    t_final = 1.2

    # d_JL = np.zeros((100,1))
    jl_mag = 1e1
    d_JL = np.random.RandomState(seed=seed).randn(100,1) * jl_mag
    jl_mag_2 = 1e1
    T_JL = np.random.RandomState(seed=seed).randn(1,1) * jl_mag_2

    with nengo.Network(seed=seed) as model:
        if signal == 'sinusoids':
            stim = nengo.Node(lambda t: 
                0.0 * (t < 1.0)
                + np.cos(2 * np.pi * freq * (t - 1.0)) * (t > 1.0))
        elif signal == 'white_noise':
            stim = nengo.Node(nengo.processes.WhiteSignal(
                period=t_final, high=max_freq, rms=rms, seed=seed))
        elif signal == 'step':
            stim = nengo.Node(lambda t:
                0.0 * (t < 1)
                + -1.0 * (1 < t < 2)
                + 1.0 * (2 < t < 3))
        elif signal == 'ramp':
            stim = nengo.Node(lambda t:
                0.0 * (t < 1)
                + (t-1) * (1 < t < 2)
                + 1.0 * (2 < t < 3))

        lif = nengo.Ensemble(100, 1, neuron_type=nengo.LIF())
        # alif = nengo.Ensemble(100, 1, neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05))
        alif = nengo.Ensemble(100, 1, neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05))
        alif2 = nengo.Ensemble(100, 2, neuron_type=nengo.AdaptiveLIF(tau_n=.01, inc_n=.05))
        oracle = nengo.Node(size_in=1)
        temp = nengo.Node(size_in=1)
        
        nengo.Connection(stim, lif, synapse=tau)
        # nengo.Connection(stim, alif, synapse=tau)
        nengo.Connection(stim, alif, synapse=tau)
        nengo.Connection(stim, alif2[:1], synapse=tau)
        nengo.Connection(alif, alif.neurons, transform=d_JL)
        nengo.Connection(alif2[1:], alif2[:1], transform=T_JL)
        nengo.Connection(stim, oracle, synapse=tau)
        conn_lif = nengo.Connection(lif, temp, synapse=None)
        
        p_oracle = nengo.Probe(oracle, synapse=tau_readout)
        p_lif = nengo.Probe(lif.neurons, 'spikes', synapse=tau_readout)
        p_alif = nengo.Probe(alif.neurons, 'spikes', synapse=tau_readout)
        p_alif2 = nengo.Probe(alif2.neurons, 'spikes', synapse=tau_readout)
        
    with nengo.Simulator(model, seed=seed) as sim:
        sim.run(t_final)

    d_lif = sim.data[conn_lif].weights.T
    # d_lif, _ = nengo.solvers.LstsqL2(reg=.1)(sim.data[p_lif][int(1.0/dt):], sim.data[p_oracle][int(1.0/dt):])
    d_alif, _ = nengo.solvers.LstsqL2(reg=.1)(sim.data[p_alif][int(1.0/dt):], sim.data[p_oracle][int(1.0/dt):])
    d_alif2, _ = nengo.solvers.LstsqL2(reg=.1)(sim.data[p_alif2][int(1.0/dt):], sim.data[p_oracle][int(1.0/dt):])
    # d_alif = np.hstack((d_alif, d_JL))

    rmse_lif = nengo.utils.numpy.rmse(sim.data[p_oracle][int(1.0/dt):], np.dot(sim.data[p_lif][int(1.0/dt):], d_lif))
    rmse_alif = nengo.utils.numpy.rmse(sim.data[p_oracle][int(1.0/dt):], np.dot(sim.data[p_alif][int(1.0/dt):], d_alif))
    rmse_alif2 = nengo.utils.numpy.rmse(sim.data[p_oracle][int(1.0/dt):], np.dot(sim.data[p_alif2][int(1.0/dt):], d_alif2))

    plt.plot(sim.trange()[int(1.0/dt):], sim.data[p_oracle][int(1.0/dt):], label="Oracle")
    plt.plot(sim.trange()[int(1.0/dt):], np.dot(sim.data[p_lif][int(1.0/dt):], d_lif), label="LIF, rmse=%.5f" % rmse_lif)
    plt.plot(sim.trange()[int(1.0/dt):], np.dot(sim.data[p_alif][int(1.0/dt):], d_alif), label="Adaptive LIF, rmse=%.5f" % rmse_alif)
    plt.plot(sim.trange()[int(1.0/dt):], np.dot(sim.data[p_alif2][int(1.0/dt):], d_alif2), label="Adaptive LIF 2, rmse=%.5f" % rmse_alif2)
    # plt.plot(sim.trange()[int(1.0/dt):], np.dot(sim.data[p_alif][int(1.0/dt):], d_alif)[:,1:], label="Adaptive LIF JL, rmse=%.5f" % rmse_alif)
    plt.legend()