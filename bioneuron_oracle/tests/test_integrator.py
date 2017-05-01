import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from bioneuron_oracle.solver import BioSolver
from nengo.utils.numpy import rmse
import seaborn as sns

def test_integrator_no_jl_dims(plt):

    """
    Simulate a network [stim]-[LIF]-[BIO]-[BIO]-[Probe]
                             -[Direct]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """
    pre_neurons=200
    bio_neurons=200
    post_neurons=200
    tau_nengo=0.01
    tau_neuron=0.01
    dt_nengo=0.001
    dt_neuron=0.0001
    pre_seed=3
    bio_seed=6
    post_seed=9
    t_final=1.0
    dim=1
    n_syn=10
    signal='prime_sinusoids'
    decoders_bio=None

    cutoff=0.3

    def sim(w_train, decoders_bio=None,plots=False):

        if decoders_bio is None:
            decoders_bio=np.zeros((bio_neurons, dim))

        with nengo.Network() as model:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                seed=pre_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                                seed=bio_seed, neuron_type=BahlNeuron())
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                     neuron_type=nengo.Direct())

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron, weights_bias_conn=True)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(bio, bio, synapse=tau_nengo, solver=bio_solver)

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_bio = nengo.Probe(bio, synapse=tau_neuron, solver=bio_solver)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)

        with nengo.Simulator(model,dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
        rmse_bio=rmse(sim.data[probe_bio], sim.data[probe_direct])

        if plots:
            sns.set(context='poster')
            plt.subplot(1,1,1)
            plt.plot(sim.trange(), sim.data[probe_bio],
                     label='[STIM]-[LIF]-[BIO]-[probe]')
            plt.plot(sim.trange(), sim.data[probe_direct],
                     label='[STIM]-[LIF]-[LIF_EQ]-[Direct]')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            legend3 = plt.legend() #prop={'size':8}

        return decoders_bio_new, rmse_bio

    decoders_bio_new, rmse_bio = sim(w_train=1.0, decoders_bio=None)
    # decoders_bio_new, rmse_bio, rmse_bio2 = sim(w_train=0.5, decoders_bio=decoders_bio_new)
    decoders_bio_new, rmse_bio = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_bio < cutoff
    assert rmse_bio2 < cutoff


def test_integrator_with_jl_dims(plt):

    """
    Simulate a network [stim]-[LIF]-[BIO]-[BIO]-[Probe]
                             -[Direct]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """
    pre_neurons=200
    bio_neurons=200
    post_neurons=200
    tau_nengo=0.01
    tau_neuron=0.01
    dt_nengo=0.001
    dt_neuron=0.0001
    pre_seed=3
    bio_seed=6
    pre2_seed=9
    t_final=1.0
    dim=1
    jl_dims=3
    n_syn=10
    signal='prime_sinusoids'
    decoders_bio=None

    cutoff=0.3

    def sim(w_train, decoders_bio=None,plots=False):

        if decoders_bio is None:
            decoders_bio=np.zeros((bio_neurons, dim))

        with nengo.Network() as model:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                seed=pre_seed, neuron_type=nengo.LIF())
            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                seed=pre2_seed, neuron_type=nengo.LIF())
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                seed=bio_seed, neuron_type=BahlNeuron())
            direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                     neuron_type=nengo.Direct())

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
            if jl_dims > 0:
	            # todo: magnitude should scale with n_neurons (maybe 1./n^2)?
	            jl_decoders = np.random.RandomState(seed=333).randn(bio_neurons, jl_dims) * 1e-4
	            bio_solver.decoders_bio = np.hstack((bio_solver.decoders_bio, jl_decoders))

            nengo.Connection(stim, lif, synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(lif, bio[:dim],
            				 synapse=tau_neuron, weights_bias_conn=True)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, pre, synapse=None)
            # oracle on the recurrent connection (trans=tau makes integrator)
            nengo.Connection(pre, bio[:dim],
            				 synapse=tau_neuron, transform=tau_neuron)
            # full connectivity w/ jl_dims on recurrent bio connection
            nengo.Connection(bio, bio, synapse=tau_nengo, solver=bio_solver)

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_bio = nengo.Probe(bio, synapse=tau_neuron, solver=bio_solver)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)

        with nengo.Simulator(model,dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
        rmse_bio=rmse(sim.data[probe_bio][:,:dim], sim.data[probe_direct])

        if plots:
            sns.set(context='poster')
            plt.subplot(1,1,1)
            plt.plot(sim.trange(), sim.data[probe_bio][:,:dim], label='BIO[:dim]')
            plt.plot(sim.trange(), sim.data[probe_direct], label='Direct')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            legend3 = plt.legend() #prop={'size':8}

        return decoders_bio_new, rmse_bio

    decoders_bio_new, rmse_bio = sim(w_train=1.0, decoders_bio=None)
    # decoders_bio_new, rmse_bio, rmse_bio2 = sim(w_train=0.5, decoders_bio=decoders_bio_new)
    decoders_bio_new, rmse_bio = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_bio < cutoff
    assert rmse_bio2 < cutoff