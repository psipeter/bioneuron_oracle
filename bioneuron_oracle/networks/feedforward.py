import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from nengo.utils.matplotlib import rasterplot
# import matplotlib.pyplot as plt
import seaborn as sns
import pytest


def feedforward(pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
                dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
                decoders_bio=None, plots={'spikes','voltage','decode'},
                simulator=None, plt=None, seed=None, plt_name=''):
    """
    Simulate a feedforward network [stim]-[LIF]-[BIO]
    and compare to [stim]-[LIF]-[LIF].
    
    signal: 'prime_sinusoids' or 'step_input'
    decoders_bio: decoders for [BIO]-[probe] from a previous simulation
    """

    with nengo.Network() as model:

        if signal == 'prime_sinusoids':
            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))
        elif signal == 'step_input':
            stim = nengo.Node(lambda t: step_input(t, dim, t_final, dt_nengo))

        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            seed=bio_seed, neuron_type=BahlNeuron())
        lif = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            neuron_type=nengo.LIF(), seed=bio_seed)
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim, 
                                neuron_type=nengo.Direct(),)

        nengo.Connection(stim,pre,synapse=None)
        nengo.Connection(pre,bio,synapse=tau_neuron)
        nengo.Connection(pre,lif,synapse=tau_nengo)
        nengo.Connection(stim,direct,synapse=tau_nengo)

        probe_stim = nengo.Probe(stim,synapse=None)
        probe_pre = nengo.Probe(pre,synapse=tau_nengo)
        probe_lif = nengo.Probe(lif,synapse=tau_nengo)
        probe_direct = nengo.Probe(direct,synapse=tau_nengo)
        probe_pre_spikes = nengo.Probe(pre.neurons,'spikes')
        probe_bio_spikes = nengo.Probe(bio.neurons,'spikes')
        probe_lif_spikes = nengo.Probe(lif.neurons,'spikes')
        
    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)
        

    sns.set(context='poster')
    if 'spikes' in plots:
        '''spike raster for PRE, BIO and comparison LIF ensembles'''
        plt.subplot(3, 1, 1)
        rasterplot(sim.trange(),sim.data[probe_pre_spikes], use_eventplot=True)
        plt.xlabel('time (s)')
        plt.ylabel('neuron')
        # plt.title('pre neurons')
        plt.subplot(3, 1, 2)
        rasterplot(sim.trange(),sim.data[probe_bio_spikes], use_eventplot=True)
        plt.xlabel('time (s)')
        plt.ylabel('neuron')
        # plt.title('bioneuron')
        plt.subplot(3, 1, 3)
        rasterplot(sim.trange(),sim.data[probe_lif_spikes], use_eventplot=True)
        plt.xlabel('time (s)')
        plt.ylabel('neuron')
        plt.ylabel('lif neuron')

    if 'voltage' in plots:
        '''voltage trace for a specific bioneuron'''
        plt.subplot(1, 1, 1)
        bio_idx = 0
        neuron = bio.neuron_type.neurons[bio_idx]
        plt.plot(np.array(neuron.t_record),np.array(neuron.v_record))
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')
        plt.title('bioneuron voltage')

    if 'decode' in plots:
        '''decoded output of bioensemble'''
        plt.subplot(1,1,1)
        lpf = nengo.Lowpass(tau_nengo)
        solver = nengo.solvers.LstsqL2(reg=0.01)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        if decoders_bio is None:
            decoders_bio, info = solver(act_bio, sim.data[probe_direct])
        xhat_bio=np.dot(act_bio,decoders_bio)
        rmse_bio=np.sqrt(np.average((
            sim.data[probe_direct]-xhat_bio)**2))
        rmse_lif=np.sqrt(np.average((
            sim.data[probe_direct]-sim.data[probe_lif])**2))
        plt.plot(sim.trange(),xhat_bio,label='bio, rmse=%.5f'%rmse_bio)
        plt.plot(sim.trange(),sim.data[probe_lif],
            label='lif, rmse=%.5f'%rmse_lif)
        plt.plot(sim.trange(),sim.data[probe_direct],label='direct')
        plt.xlabel('time (s)')
        plt.ylabel('$\hat{x}(t)$')
        plt.title('decode')
        legend3=plt.legend() #prop={'size':8}
    
    # todo: call NEURON garbage collection
    return decoders_bio, rmse_bio, rmse_lif