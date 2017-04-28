import nengo
import numpy as np
import neuron
import seaborn as sns
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from nengo.utils.matplotlib import rasterplot
from nengo.utils.numpy import rmse

pre_neurons=100
bio_neurons=20
tau_nengo=0.01
tau_neuron=0.01
dt_nengo=0.001
dt_neuron=0.0001
pre_seed=3
bio_seed=6
t_final=1.0
dim=2
n_syn=10
signal='prime_sinusoids'
decoders_bio=None

with nengo.Network() as model:
    """
    Simulate a feedforward network [stim]-[LIF]-[BIO]
    and compare to [stim]-[LIF]-[LIF].
    """

    if signal == 'prime_sinusoids':
        stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
    elif signal == 'step_input':
        stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))

    pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                        seed=pre_seed, neuron_type=nengo.LIF())
    bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                        seed=bio_seed, neuron_type=BahlNeuron())
    # lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
    #                     neuron_type=nengo.LIF(), seed=bio_seed)
    direct = nengo.Ensemble(n_neurons=1, dimensions=dim, 
                            neuron_type=nengo.Direct(),)

    nengo.Connection(stim, pre, synapse=None)
    nengo.Connection(pre, bio, synapse=tau_neuron, weights_bias_conn=True)
    # nengo.Connection(pre,lif,synapse=tau_nengo)
    nengo.Connection(stim, direct, synapse=tau_nengo)

    probe_stim = nengo.Probe(stim, synapse=None)
    # probe_pre = nengo.Probe(pre,synapse=tau_nengo)
    # probe_lif = nengo.Probe(lif,synapse=tau_nengo)
    probe_direct = nengo.Probe(direct, synapse=tau_nengo)
    # probe_pre_spikes = nengo.Probe(pre.neurons,'spikes')
    probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
    # probe_lif_spikes = nengo.Probe(lif.neurons,'spikes')
    
with nengo.Simulator(model,dt=dt_nengo) as sim:
    sim.run(t_final)

# todo: call NEURON garbage collection

# Generate decoders and a basic decoding for comparison
lpf = nengo.Lowpass(tau_nengo)
solver = nengo.solvers.LstsqL2(reg=0.01)
act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
decoders_bio_old, info = solver(act_bio, sim.data[probe_direct])
xhat_bio_old=np.dot(act_bio,decoders_bio_old)
rmse_bio_old=rmse(sim.data[probe_direct],xhat_bio_old)
# rmse_lif=rmse(sim.data[probe_direct],sim.data[probe_lif])


def test_ff_new_LIF_old_decoders(plt):
    """
    Change the LIF input (seed) but decode with original decoders
    This tests the generalizability of decoders for novel bio activities

    test passes if:
        - RMSE (xhat_bio_mixed, xhat_bio_new) < $cutoff$
        - rmse_mixed < cutoff
    """
    pre_seed=9
    cutoff_mixed=0.3
    cutoff_compare=0.3

    with nengo.Network() as model:
        """
        Simulate a feedforward network [stim]-[LIF]-[BIO]
        and compare to [stim]-[LIF]-[LIF].
        """

        if signal == 'prime_sinusoids':
            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
        elif signal == 'step_input':
            stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            seed=bio_seed, neuron_type=BahlNeuron())
        # lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
        #                     neuron_type=nengo.LIF(), seed=bio_seed)
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim, 
                                neuron_type=nengo.Direct(),)

        nengo.Connection(stim, pre, synapse=None)
        nengo.Connection(pre, bio, synapse=tau_neuron, weights_bias_conn=True)
        # nengo.Connection(pre,lif,synapse=tau_nengo)
        nengo.Connection(stim, direct, synapse=tau_nengo)

        probe_stim = nengo.Probe(stim, synapse=None)
        # probe_pre = nengo.Probe(pre,synapse=tau_nengo)
        # probe_lif = nengo.Probe(lif,synapse=tau_nengo)
        probe_direct = nengo.Probe(direct, synapse=tau_nengo)
        # probe_pre_spikes = nengo.Probe(pre.neurons,'spikes')
        probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
        # probe_lif_spikes = nengo.Probe(lif.neurons,'spikes')
        
    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)

    # todo: call NEURON garbage collection

    # Generate a new decoding using the old decoders and new activities
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio_mixed=np.dot(act_bio,decoders_bio_old)
    rmse_bio_mixed=rmse_bio_old=rmse(sim.data[probe_direct],xhat_bio_mixed)
    # Generate a new decoding using the new decoders and new activities
    decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio_new=np.dot(act_bio,decoders_bio_new)
    rmse_bio_new=rmse_bio_old=rmse(sim.data[probe_direct],xhat_bio_new)
    # Calculate the RMSE of the mixed and new decoding
    rmse_compare=rmse_bio_old=rmse(xhat_bio_mixed,xhat_bio_new)

    plt.subplot(1,1,1)
    plt.plot(sim.trange(),xhat_bio_old,label='old, rmse=%.5f'%rmse_bio_old)
    plt.plot(sim.trange(),xhat_bio_mixed,label='mixed, rmse=%.5f'%rmse_bio_mixed)
    plt.plot(sim.trange(),xhat_bio_new,label='new, rmse=%.5f'%rmse_bio_new)
    # plt.plot(sim.trange(),sim.data[probe_lif],label='lif, rmse=%.5f'%rmse_lif)
    plt.plot(sim.trange(),sim.data[probe_direct],label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    legend3=plt.legend() #prop={'size':8}

    assert rmse_bio_mixed < cutoff_mixed
    assert rmse_compare < cutoff_compare


def test_ff_new_signal_old_decoders(plt):
    """
    Change the and the signal type (step_input) but decode with original decoders
    This tests the generalizability of decoders for novel activities and targets

    test passes if:
        - RMSE (xhat_bio_mixed, xhat_bio_new) < $cutoff$
        - rmse_mixed < cutoff
    """
    pre_seed = 3
    bio_seed = 6
    signal = 'step_input'
    cutoff_mixed = 0.5
    cutoff_compare = 0.5

    with nengo.Network() as model:
        """
        Simulate a feedforward network [stim]-[LIF]-[BIO]
        and compare to [stim]-[LIF]-[LIF].
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

        nengo.Connection(stim,pre,synapse=None)
        nengo.Connection(pre, bio, synapse=tau_neuron, weights_bias_conn=True)
        nengo.Connection(stim,direct,synapse=tau_nengo)

        probe_stim = nengo.Probe(stim,synapse=None)
        probe_direct = nengo.Probe(direct,synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons,'spikes')
        
    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)

    # todo: call NEURON garbage collection

    # Generate a new decoding using the old decoders and new activities
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio_mixed=np.dot(act_bio,decoders_bio_old)
    rmse_bio_mixed=rmse_bio_old=rmse(sim.data[probe_direct],xhat_bio_mixed)
    # Generate a new decoding using the new decoders and new activities
    decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio_new=np.dot(act_bio,decoders_bio_new)
    rmse_bio_new=rmse_bio_old=rmse(sim.data[probe_direct],xhat_bio_new)
    # Calculate the RMSE of the mixed and new decoding
    rmse_compare=rmse_bio_old=rmse(xhat_bio_mixed,xhat_bio_new)

    plt.subplot(1,1,1)
    plt.plot(sim.trange(),xhat_bio_mixed,label='mixed, rmse=%.5f'%rmse_bio_mixed)
    plt.plot(sim.trange(),xhat_bio_new,label='new, rmse=%.5f'%rmse_bio_new)
    plt.plot(sim.trange(),sim.data[probe_direct],label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    legend3=plt.legend() #prop={'size':8}

    assert rmse_bio_mixed < cutoff_mixed
    assert rmse_compare < cutoff_compare



def test_ff_new_LIF_new_signal_old_decoders(plt):
    """
    Change the LIF input (seed) and the signal type (step_input) 
    but decode with original decoders
    This further tests the generalizability of decoders for novel bio activities
    and signals

    test passes if:
        - RMSE (xhat_bio_mixed, xhat_bio_new) < $cutoff$
        - rmse_mixed < cutoff
    """
    pre_seed=9
    signal = 'step_input'
    cutoff_mixed=0.5
    cutoff_compare=0.5

    with nengo.Network() as model:
        """
        Simulate a feedforward network [stim]-[LIF]-[BIO]
        and compare to [stim]-[LIF]-[LIF].
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

        nengo.Connection(stim,pre,synapse=None)
        nengo.Connection(pre, bio, synapse=tau_neuron, weights_bias_conn=True)
        nengo.Connection(stim,direct,synapse=tau_nengo)

        probe_stim = nengo.Probe(stim,synapse=None)
        probe_direct = nengo.Probe(direct,synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons,'spikes')
        
    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)

    # todo: call NEURON garbage collection

    # Generate a new decoding using the old decoders and new activities
    lpf = nengo.Lowpass(tau_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.01)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    xhat_bio_mixed=np.dot(act_bio,decoders_bio_old)
    rmse_bio_mixed=rmse_bio_old=rmse(sim.data[probe_direct],xhat_bio_mixed)
    # Generate a new decoding using the new decoders and new activities
    decoders_bio_new, info = solver(act_bio, sim.data[probe_direct])
    xhat_bio_new=np.dot(act_bio,decoders_bio_new)
    rmse_bio_new=rmse_bio_old=rmse(sim.data[probe_direct],xhat_bio_new)
    # Calculate the RMSE of the mixed and new decoding
    rmse_compare=rmse_bio_old=rmse(xhat_bio_mixed,xhat_bio_new)

    plt.subplot(1,1,1)
    plt.plot(sim.trange(),xhat_bio_mixed,label='mixed, rmse=%.5f'%rmse_bio_mixed)
    plt.plot(sim.trange(),xhat_bio_new,label='new, rmse=%.5f'%rmse_bio_new)
    plt.plot(sim.trange(),sim.data[probe_direct],label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    legend3=plt.legend() #prop={'size':8}

    assert rmse_bio_mixed < cutoff_mixed
    assert rmse_compare < cutoff_compare
