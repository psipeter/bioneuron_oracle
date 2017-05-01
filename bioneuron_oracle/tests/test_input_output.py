import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from bioneuron_oracle.solver import BioSolver
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

def test_transform_in(plt):
    cutoff=0.3
    transform=-0.5
    with nengo.Network() as model:
        """
        Simulate a feedforward network [stim]-[LIF]-(transform)-[BIO]
        test passes if rmse_bio < cutoff
        """

        if signal == 'prime_sinusoids':
            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
        elif signal == 'step_input':
            stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim, 
                                neuron_type=nengo.Direct(),)

        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, bio, 
                         synapse=tau_neuron,
                         transform=transform, 
                         weights_bias_conn=True)
        nengo.Connection(stim, direct,
                         transform=transform,
                         synapse=tau_nengo)

        probe_stim = nengo.Probe(stim, synapse=None)
        probe_pre = nengo.Probe(pre, synapse=tau_nengo)
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
    rmse_bio = rmse(sim.data[probe_direct], xhat_bio)
    plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
    plt.plot(sim.trange(), sim.data[probe_direct], label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    legend3 = plt.legend() #prop={'size':8}
    assert rmse_bio < cutoff

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


def test_slice_in(plt):

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


def test_slice_out(plt):

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

    cutoff=0.1

    def sim(w_train=1, decoders_bio=None, plots=False):

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

            bio_solver1 = BioSolver(decoders_bio = (1. - w_train) * decoders_bio[:,0])
            bio_solver2 = BioSolver(decoders_bio = (1. - w_train) * decoders_bio[:,1])

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron, weights_bias_conn=True)
            nengo.Connection(bio[0], lif1, synapse=tau_nengo, solver=bio_solver1)
            nengo.Connection(bio[1], lif2, synapse=tau_nengo, solver=bio_solver2)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, direct_out, synapse=tau_nengo)
            #todo - test output to node, direct, etc

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_bio = nengo.Probe(bio[0], synapse=None, solver=bio_solver1)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif1 = nengo.Probe(lif1, synapse=tau_nengo)
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

        with nengo.Simulator(model,dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])

        if plots:
            sns.set(context='poster')
            plt.subplot(1,1,1)
            rmse_lif1=rmse(sim.data[probe_lif1], sim.data[probe_direct_out][:,0])
            rmse_lif2=rmse(sim.data[probe_lif2], sim.data[probe_direct_out][:,1])
            plt.plot(sim.trange(), sim.data[probe_bio], label='[STIM]-[LIF]-[BIO][0]-[probe]')
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

    decoders_bio_new, rmse_lif1, rmse_lif2 = sim(w_train=1, decoders_bio=None)
    decoders_bio_new, rmse_lif1, rmse_lif2 = sim(w_train=0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_lif1 < cutoff
    assert rmse_lif2 < cutoff


def test_transform_out(plt):

    """
    Simulate a network [stim]-[LIF]-[BIO]-(transform)-[BIO2]
                             -[Direct]-(transform)-[Direct_Out]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    """

    cutoff=0.3
    transform=-0.5

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
            pre = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                seed=bio_seed, neuron_type=nengo.LIF())
            lif2 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim, 
                                seed=post_seed, neuron_type=nengo.LIF())
            direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                     neuron_type=nengo.Direct())

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron,
                             weights_bias_conn=True)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(bio, lif2, synapse=tau_nengo,
                             transform=transform, solver=bio_solver)
            nengo.Connection(direct, direct_out, synapse=tau_nengo,
                             transform=transform)
            #todo - test output to node, direct, etc

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_bio = nengo.Probe(bio, synapse=tau_neuron,
                                    solver=bio_solver)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_lif2 = nengo.Probe(lif2, synapse=tau_nengo)
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

        with nengo.Simulator(model,dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])
        rmse_lif=rmse(sim.data[probe_lif2], sim.data[probe_direct])

        if plots:
            sns.set(context='poster')
            plt.subplot(1,1,1)
            plt.plot(sim.trange(), sim.data[probe_lif2],
                     label='[STIM]-[LIF]-[BIO]-[LIF2]')
            plt.plot(sim.trange(), sim.data[probe_direct_out],
                     label='[STIM]-[Direct]-[Direct_Out]')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            legend3 = plt.legend() #prop={'size':8}

        return decoders_bio_new, rmse_lif

    decoders_bio_new, rmse_lif = sim(w_train=1.0, decoders_bio=None)
    decoders_bio_new, rmse_lif = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_lif < cutoff


def test_bio_to_bio(plt):

    """
    Simulate a network [stim]-[LIF]-[BIO]-[BIO2]
                             -[Direct]-[Direct_Out]
    decoders_bio: decoders out of [BIO] that are trained
                  by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

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
            pre = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim,
                                seed=bio_seed, neuron_type=nengo.LIF())
            bio2 = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                                seed=post_seed, neuron_type=BahlNeuron())
            direct_out = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                     neuron_type=nengo.Direct())

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)

            nengo.Connection(stim, lif, synapse=None)
            nengo.Connection(lif, bio, synapse=tau_neuron, weights_bias_conn=True)
            nengo.Connection(lif, direct, synapse=tau_nengo)
            nengo.Connection(direct, pre, synapse=None)
            nengo.Connection(pre, bio2, synapse=tau_nengo, transform=w_train)
            nengo.Connection(bio, bio2, synapse=tau_nengo, solver=bio_solver)
            nengo.Connection(direct, direct_out, synapse=tau_nengo)
            #todo - test output to node, direct, etc

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_bio = nengo.Probe(bio, synapse=tau_neuron, solver=bio_solver)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_bio2_spikes = nengo.Probe(bio2.neurons, 'spikes')
            probe_direct = nengo.Probe(direct, synapse=tau_nengo)
            probe_direct_out = nengo.Probe(direct_out, synapse=tau_nengo)

        with nengo.Simulator(model,dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        act_bio2 = lpf.filt(sim.data[probe_bio2_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        assert np.sum(act_bio2) > 0.0
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, sim.data[probe_direct_out])
        decoders_bio_new2, info2 = solver(act_bio2, sim.data[probe_direct_out])
        xhat_bio = np.dot(act_bio, decoders_bio_new)
        xhat_bio2 = np.dot(act_bio2, decoders_bio_new2)
        rmse_bio=rmse(xhat_bio, sim.data[probe_direct])
        rmse_bio2=rmse(xhat_bio2, sim.data[probe_direct_out])

        if plots:
            sns.set(context='poster')
            plt.subplot(1,1,1)
            # plt.plot(sim.trange(), sim.data[probe_bio],
            #          label='[STIM]-[LIF]-[BIO]-[probe]')
            # plt.plot(sim.trange(), xhat_bio,
            #          label='[STIM]-[LIF]-[BIO], rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), xhat_bio2,
                     label='[STIM]-[LIF]-[BIO]-[BIO2], rmse=%.5f' % rmse_bio2)
            # plt.plot(sim.trange(), sim.data[probe_direct],
            #          label='[STIM]-[LIF]-[LIF_EQ]-[Direct]')
            plt.plot(sim.trange(), sim.data[probe_direct_out],
                     label='[STIM]-[Direct]-[Direct_Out]')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            legend3 = plt.legend() #prop={'size':8}

        return decoders_bio_new, rmse_bio, rmse_bio2

    decoders_bio_new, rmse_bio, rmse_bio2 = sim(w_train=1.0, decoders_bio=None)
    # decoders_bio_new, rmse_bio, rmse_bio2 = sim(w_train=0.5, decoders_bio=decoders_bio_new)
    decoders_bio_new, rmse_bio, rmse_bio2 = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_bio < cutoff
    assert rmse_bio2 < cutoff
