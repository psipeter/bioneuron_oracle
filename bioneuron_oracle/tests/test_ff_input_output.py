import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from nengo.utils.numpy import rmse
import seaborn as sns

pre_neurons=100
bio_neurons=20
post_neurons=100
tau_nengo=0.01
tau_neuron=0.01
dt_nengo=0.001
dt_neuron=0.0001
pre_seed=3
bio_seed=6
post_seed=9
t_final=1.0
dim=2
assert dim % 2 == 0
n_syn=10
signal='prime_sinusoids'
decoders_bio=None


def test_two_inputs(plt):

    """
    Simulate a network [stim1]-[LIF1]-[BIO]
                       [stim2]-[LIF2]-[BIO]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff=0.4

    with nengo.Network() as model:

        stim1 = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final)[0:dim])
        stim2 = nengo.Node(lambda t: prime_sinusoids(t, 2*dim, t_final)[dim:2*dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                 neuron_type=nengo.Direct())

        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(stim2, lif2, synapse=None)
        nengo.Connection(lif1, bio, synapse=tau_neuron, weights_bias_conn=True)
        nengo.Connection(lif2, bio, synapse=tau_neuron)
        nengo.Connection(stim1, direct, synapse=tau_nengo)
        nengo.Connection(stim2, direct, synapse=tau_nengo)

        probe_stim1 = nengo.Probe(stim1, synapse=None)
        probe_stim2 = nengo.Probe(stim2, synapse=None)
        probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
        probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')

    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)
    
    sns.set(context='poster')
    plt.subplot(1,1,1)
    rmse_bio=rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio dim 1, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    legend3 = plt.legend() #prop={'size':8}
    assert rmse_bio < cutoff


def test_slicing(plt):

    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][1]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff=0.4

    with nengo.Network() as model:

        stim1 = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final)[0:dim/2])
        stim2 = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final)[dim/2:dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                            seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim/2,
                            seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                 neuron_type=nengo.Direct())

        nengo.Connection(stim1, lif1, synapse=None)
        nengo.Connection(stim2, lif2, synapse=None)
        nengo.Connection(lif1, bio[0], synapse=tau_neuron, weights_bias_conn=True)
        nengo.Connection(lif2, bio[1], synapse=tau_neuron)
        nengo.Connection(stim1, direct[0], synapse=tau_nengo)
        nengo.Connection(stim2, direct[1], synapse=tau_nengo)

        probe_stim1 = nengo.Probe(stim1, synapse=None)
        probe_stim2 = nengo.Probe(stim2, synapse=None)
        probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
        probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
    
    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)

    # todo: call NEURON garbage collection

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    assert np.sum(act_bio) > 0.0
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio = np.dot(act_bio, decoders_bio)
    
    sns.set(context='poster')
    plt.subplot(1,1,1)
    rmse_bio_1=rmse(sim.data[probe_direct][:,0:dim/2], xhat_bio[:,0:dim/2])
    rmse_bio_2=rmse(sim.data[probe_direct][:,dim/2:dim], xhat_bio[:,dim/2:dim])
    plt.plot(sim.trange(), xhat_bio[:,0:dim/2], label='bio dim 1, rmse=%.5f' % rmse_bio_1)
    plt.plot(sim.trange(), xhat_bio[:,dim/2:dim], label='bio dim 2, rmse=%.5f' % rmse_bio_2)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    legend3 = plt.legend() #prop={'size':8}
    assert rmse_bio_1 < cutoff
    assert rmse_bio_2 < cutoff


def test_two_outputs(plt):

    """
    Simulate a network [stim]-[LIF]-[BIO]-[LIF1]
                                         -[LIF2]
                             -[Direct]-[Direct_out]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

    cutoff=0.3

    def sim_two_outputs(w_train=1, decoders_bio=None):

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
            lif1 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                seed=post_seed, neuron_type=nengo.LIF())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim/2,
                                seed=2*post_seed, neuron_type=nengo.LIF())
            direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                     neuron_type=nengo.Direct())

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron, weights_bias_conn=True)
            conn1 = nengo.Connection(bio[0], lif1, synapse=tau_nengo)
            conn2 = nengo.Connection(bio[1], lif2, synapse=tau_nengo)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct_out, synapse=tau_nengo)
            #todo - test output to node, direct, etc

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

            conn1.decoders_bio = (1. - w_train) * decoders_bio[:,0]
            conn2.decoders_bio = (1. - w_train) * decoders_bio[:,1]
            print conn1.decoders_bio
            print conn2.decoders_bio

        with nengo.Simulator(model,dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])

        sns.set(context='poster')
        plt.subplot(1,1,1)
        rmse_lif1=rmse(sim.data[probe_lif1], sim.data[probe_direct_out][:,0])
        rmse_lif2=rmse(sim.data[probe_lif2], sim.data[probe_direct_out][:,1])
        plt.plot(sim.trange(), sim.data[probe_lif1],
                 label='[STIM]-[LIF]-[BIO]-[LIF1], rmse=%.5f' % rmse_lif1)
        plt.plot(sim.trange(), sim.data[probe_lif2],
                 label='[STIM]-[LIF]-[BIO]-[LIF2], rmse=%.5f' % rmse_lif2)
        plt.plot(sim.trange(), sim.data[probe_direct_out][:,0],
                 label='[STIM]-[LIF]-[LIF_EQ]-[Direct][0]')
        plt.plot(sim.trange(), sim.data[probe_direct_out][:,1],
                 label='[STIM]-[LIF]-[LIF_EQ]-[Direct][1]')
        plt.xlabel('time (s)')
        plt.ylabel('$\hat{x}(t)$')
        plt.title('decode')
        legend3 = plt.legend() #prop={'size':8}

        return decoders_bio_new, rmse_lif1, rmse_lif2

    decoders_bio_new, rmse_lif1, rmse_lif2 = sim_two_outputs(w_train=1, decoders_bio=None)
    decoders_bio_new, rmse_lif1, rmse_lif2 = sim_two_outputs(w_train=0, decoders_bio=decoders_bio_new)

    assert rmse_lif1 < cutoff
    assert rmse_lif2 < cutoff

                # conn3.decoders_bio = (1 - w_train)*decoders_bio  # HACK
#         if jl_dims > 0:
#             # TODO: magnitude should scale with n_neurons somehow (maybe 1./n^2)?
#             jl_decoders = np.random.RandomState(seed=333).randn(n_neurons, jl_dims) * 1e-4
#             conn.oracle_decoders = np.hstack((conn.oracle_decoders, jl_decoders))