import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from nengo.utils.matplotlib import rasterplot
import matplotlib.pyplot as plt
import seaborn as sns


def mutli_input_output(pre_neurons, bio_neurons, post_neurons,
                       tau_nengo, tau_neuron,
                       dt_nengo, dt_neuron, pre_seed, bio_seed, post_seed,
                       t_final, dim,
                       bio_decoders=None, w_train=0, jl_dims=0,
                       plots={'spikes','voltage','decode'}):

    """
    Simulate a network [stim1]-[LIF1]-[BIO]-[LIF3]
                       [stim2]-[LIF2]-[BIO]-[LIF4]
    bio_decoders: decoders out of [BIO] that are trained
                     by the oracle method (iterative)
    w_train: soft-mux parameter that governs what fraction of
             [BIO]-out decoders are computed
             randomly vs from the oracle method
    jl_dims: extra dimensions for the oracle training
             (Johnson-Lindenstrauss lemma)
    """

    if bio_decoders is None:
        bio_decoders = np.zeros((bio_neurons,dim))
        
    with nengo.Network() as model:

        stim1 = nengo.Node(lambda t: 0.5*prime_sinusoids(t, 2*dim, t_final)[:dim])
#         stim2 = nengo.Node(lambda t: 0.5*prime_sinusoids(t, 2*dim, t_final)[dim:2*dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=pre_seed, neuron_type=nengo.LIF())
#         lif2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
#                             seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            seed=bio_seed, neuron_type=BahlNeuron())
        lif3 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim, 
                            neuron_type=nengo.LIF(), seed=post_seed)
#         lif4 = nengo.Ensemble(n_neurons=post_neurons, dimensions=dim, 
#                             neuron_type=nengo.LIF(), seed=2*post_seed)
        direct_eq = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                 neuron_type=nengo.Direct())
        direct3 = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                 neuron_type=nengo.Direct())
#         direct4 = nengo.Ensemble(n_neurons=1, dimensions=dim,
#                                  neuron_type=nengo.Direct())

        nengo.Connection(stim1,lif1,synapse=None)
#         nengo.Connection(stim2,lif2,synapse=None)
        nengo.Connection(lif1,bio,synapse=tau_neuron)
#         nengo.Connection(lif2,bio,synapse=tau_neuron)
        conn3 = nengo.Connection(bio,lif3,synapse=tau_neuron,
                                 transform=[[w_train]])
#         conn4 = nengo.Connection(bio,lif4,synapse=tau_neuron)
        nengo.Connection(stim1,direct_eq,synapse=tau_nengo)
#         nengo.Connection(stim2,direct_eq,synapse=tau_nengo)
        nengo.Connection(direct_eq,direct3,synapse=tau_nengo)
#         nengo.Connection(direct_eq,direct4,synapse=tau_nengo)

        probe_stim1 = nengo.Probe(stim1,synapse=None)
#         probe_stim2 = nengo.Probe(stim2,synapse=None)
        probe_lif1 = nengo.Probe(lif1,synapse=tau_nengo)
#         probe_lif2 = nengo.Probe(lif2,synapse=tau_nengo)
        probe_lif3 = nengo.Probe(lif3,synapse=tau_nengo)
#         probe_lif4 = nengo.Probe(lif4,synapse=tau_nengo)
        probe_direct3 = nengo.Probe(direct3,synapse=tau_nengo)
#         probe_direct4 = nengo.Probe(direct4,synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons,'spikes')
        probe_bio = nengo.Probe(bio,)
    
        conn3.bio_decoders = (1 - w_train)*bio_decoders  # HACK
#         if jl_dims > 0:
#             # TODO: magnitude should scale with n_neurons somehow (maybe 1./n^2)?
#             jl_decoders = np.random.RandomState(seed=333).randn(n_neurons, jl_dims) * 1e-4
#             conn.oracle_decoders = np.hstack((conn.oracle_decoders, jl_decoders))
        
    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)
        
    assert np.count_nonzero(sim.data[probe_bio] > 1.0/dt_nengo) == 0
    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    y = sim.data[probe_direct3]
    solver = nengo.solvers.LstsqL2(reg=0.1)
    bio_decoders, info = solver(act_bio, y)
    y_hat = np.dot(act_bio, bio_decoders)
    
    sns.set(context='poster')
    if 'spikes' in plots:
        '''spike raster for BIO ensemble'''
        figure, ax1 = plt.subplots(1, 1, sharex=True)
        rasterplot(sim.trange(),sim.data[probe_bio_spikes],ax=ax1,
                    use_eventplot=True)
        ax1.set(ylabel='bioneuron',yticks=([]))
    if 'voltage' in plots:
        '''voltage trace for a specific bioneuron'''
        figure2, ax1 = plt.subplots(1, 1, sharex=True)
        bio_idx = 0
        neuron = bio.neuron_type.neurons[bio_idx]
        ax1.plot(np.array(neuron.t_record),np.array(neuron.v_record))
        ax1.set(xlabel='time (ms)', ylabel='Voltage (mV)')
    if 'decode' in plots:
        '''decoded output of lif3 and lif4'''
        figure3, (ax3, ax4) = plt.subplots(2,1,sharex=True)
        rmse_lif3 = np.sqrt(np.average((
            sim.data[probe_direct3]-sim.data[probe_lif3])**2))
#         rmse_lif4 = np.sqrt(np.average((
#             sim.data[probe_direct4]-sim.data[probe_lif4])**2))
        ax3.plot(sim.trange(),sim.data[probe_lif3],
            label='lif, rmse=%.5f'%rmse_lif3)
        ax3.plot(sim.trange(),sim.data[probe_direct3],label='direct3')        
#         ax4.plot(sim.trange(),sim.data[probe_lif4],
#             label='lif, rmse=%.5f'%rmse_lif4)
#         ax4.plot(sim.trange(),sim.data[probe_direct4],label='direct4')
        ax3.set(ylabel='$\hat{x}(t)$')#,ylim=((ymin,ymax)))
#         ax4.set(ylabel='$\hat{x}(t)$')#,ylim=((ymin,ymax)))
        legend3=ax3.legend() #prop={'size':8}
#         legend4=ax4.legend() #prop={'size':8}
    plt.show()
    
    print 'weights used in this simulation', sim.data[conn3].weights
    print 'bio_decoders to be used in next simulation', bio_decoders
#     print sim.data[conn4].weights
    # todo: call NEURON garbage collection
    return bio_decoders


"""Unit Test"""
pre_neurons=100
bio_neurons=50
post_neurons=100
tau_nengo=0.01
tau_neuron=0.01
dt_nengo=0.001
dt_neuron=0.0001
pre_seed=3
bio_seed=6
post_seed=9
t_final=0.1
dim=1
n_syn=1
bio_decoders=None
w_trains=[1.0,0.0]
plots={'spikes','voltage','decode'}
for w_train in w_trains:
    d_new = mutli_input_output(pre_neurons, bio_neurons, post_neurons, tau_nengo, tau_neuron,
                    dt_nengo, dt_neuron, pre_seed, bio_seed, post_seed, t_final, dim, 
                    bio_decoders, w_train, plots)
    bio_decoders = d_new