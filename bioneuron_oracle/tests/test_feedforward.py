import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from nengo.utils.matplotlib import rasterplot
import matplotlib.pyplot as plt
import seaborn as sns


def feedforward(pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
                dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
                decoders_bio=None, plots={'spikes','voltage','decode'}):
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
        figure, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        rasterplot(sim.trange(),sim.data[probe_pre_spikes],ax=ax1,
                    use_eventplot=True)
        ax1.set(ylabel='pre',yticks=([]))
        rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax2,
                    use_eventplot=True)
        ax2.set(ylabel='bioneuron',yticks=([]))
        rasterplot(sim.trange(),sim.data[probe_lif_spikes],ax=ax3,
                    use_eventplot=True)
        ax3.set(xlabel='time (s)',ylabel='lif') #,yticks=([])

    if 'voltage' in plots:
        '''voltage trace for a specific bioneuron'''
        figure2, ax3 = plt.subplots(1, 1, sharex=True)
        bio_idx = 0
        neuron = bio.neuron_type.neurons[bio_idx]
        ax3.plot(np.array(neuron.t_record),np.array(neuron.v_record))
        ax3.set(xlabel='time (ms)', ylabel='Voltage (mV)')

    if 'decode' in plots:
        '''decoded output of bioensemble'''
        figure3, ax4 = plt.subplots(1,1,sharex=True)
        lpf = nengo.Lowpass(tau_nengo)
        solver = nengo.solvers.LstsqL2(reg=0.01)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        if decoders_bio==None:
            decoders_bio, info = solver(act_bio, sim.data[probe_direct])
        xhat_bio=np.dot(act_bio,decoders_bio)
        rmse_bio=np.sqrt(np.average((
            sim.data[probe_direct]-xhat_bio)**2))
        rmse_lif=np.sqrt(np.average((
            sim.data[probe_direct]-sim.data[probe_lif])**2))
        ax4.plot(sim.trange(),xhat_bio,label='bio, rmse=%.5f'%rmse_bio)
        ax4.plot(sim.trange(),sim.data[probe_lif],
            label='lif, rmse=%.5f'%rmse_lif)
        ax4.plot(sim.trange(),sim.data[probe_direct],label='direct')
        ax4.set(ylabel='$\hat{x}(t)$')#,ylim=((ymin,ymax)))
        legend3=ax4.legend() #prop={'size':8}
    plt.show()
    
    # todo: call NEURON garbage collection
    return decoders_bio, rmse_bio, rmse_lif

def unit_test():
    pre_neurons=100
    bio_neurons=50
    tau_nengo=0.01
    tau_neuron=0.01
    dt_nengo=0.001
    dt_neuron=0.0001
    pre_seed=3
    bio_seed=6
    t_final=1.0
    dim=2
    n_syn=5
    signal='prime_sinusoids'

    decoders_bio=None
    plots={'spikes','voltage','decode'}
    d_1, rmse_bio, rmse_lif = feedforward(
        pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
        dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
        decoders_bio, plots)

    assert(rmse_bio < 0.3)
    assert(rmse_lif < 0.2)

# """Use random decoders instead of those computed by the oracle"""
# decoders_bio=np.random.RandomState(seed=123).uniform(
#     np.min(d_1),np.max(d_1),size=d_1.shape)
# plots={'decode'}
# d_2 = feedforward(pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
#                 dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
#                 decoders_bio, plots)

# """Change the LIF input (seed) but decode with original decoders"""
# decoders_bio=d_1
# plots={'decode'}
# d_3 = feedforward(pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
#                 dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
#                 d_1, plots)

# """New LIF (seed), new signal, old decoders"""
# decoders_bio=d_1
# signal='step_input'
# plots={'decode'}
# d_4 = feedforward(pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
#                 dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
#                 d_1, plots)