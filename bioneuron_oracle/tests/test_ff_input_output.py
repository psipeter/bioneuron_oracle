import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from nengo.utils.numpy import rmse
import seaborn as sns


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

def test_slicing(plt):

    """
    Simulate a network [stim1]-[LIF1]-[BIO][0]
                       [stim2]-[LIF2]-[BIO][1]
    test passes if:
        - rmse_bio < cutoff for both sets of dimenions
          (slicing preserves vectors)
    """

    cutoff=0.2

    with nengo.Network() as model:

        stim1 = nengo.Node(lambda t: 0.5*prime_sinusoids(t, 2*dim, t_final)[:dim])
        stim2 = nengo.Node(lambda t: 0.5*prime_sinusoids(t, 2*dim, t_final)[dim:2*dim])

        lif1 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=pre_seed, neuron_type=nengo.LIF())
        lif2 = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                            seed=2*pre_seed, neuron_type=nengo.LIF())
        bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim, 
                            seed=bio_seed, neuron_type=BahlNeuron())
        direct = nengo.Ensemble(n_neurons=1, dimensions=dim,
                                 neuron_type=nengo.Direct())

        nengo.Connection(stim1,lif1,synapse=None)
        nengo.Connection(stim2,lif2,synapse=None)
        nengo.Connection(lif1,bio[0],synapse=tau_neuron)
        nengo.Connection(lif2,bio[1],synapse=tau_neuron)
        nengo.Connection(stim1,direct[0],synapse=tau_nengo)
        nengo.Connection(stim2,direct[1],synapse=tau_nengo)

        probe_stim1 = nengo.Probe(stim1,synapse=None)
        probe_stim2 = nengo.Probe(stim2,synapse=None)
        probe_lif1 = nengo.Probe(lif1,synapse=tau_nengo)
        probe_lif2 = nengo.Probe(lif2,synapse=tau_nengo)
        probe_direct = nengo.Probe(direct3,synapse=tau_nengo)
        probe_bio_spikes = nengo.Probe(bio.neurons,'spikes')
    
    with nengo.Simulator(model,dt=dt_nengo) as sim:
        sim.run(t_final)

    # todo: call NEURON garbage collection

    lpf = nengo.Lowpass(tau_nengo)
    act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
    solver = nengo.solvers.LstsqL2(reg=0.1)
    decoders_bio, info = solver(act_bio, sim.data[probe_direct3])
    xhat = np.dot(act_bio, decoders_bio)
    
    sns.set(context='poster')
    plt.subplot(1,1,1)
    rmse_bio_1=rmse(sim.data[probe_direct][:,0:dim],xhat_bio[:,0:dim])
    rmse_bio_2=rmse(sim.data[probe_direct][:,dim:2*dim],xhat_bio[:,dim:2*dim])
    plt.plot(sim.trange(),xhat_bio[:,0:dim],label='bio dim 1, rmse=%.5f'%rmse_bio_1)
    plt.plot(sim.trange(),xhat_bio[:,dim:2*dim],label='bio dim 2, rmse=%.5f'%rmse_bio_2)
    plt.plot(sim.trange(),sim.data[probe_direct],label='direct')
    plt.xlabel('time (s)')
    plt.ylabel('$\hat{x}(t)$')
    plt.title('decode')
    legend3=plt.legend() #prop={'size':8}
    assert rmse_bio_1 < cutoff
    assert rmse_bio_2 < cutoff
    
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

            # conn3.bio_decoders = (1 - w_train)*bio_decoders  # HACK
#         if jl_dims > 0:
#             # TODO: magnitude should scale with n_neurons somehow (maybe 1./n^2)?
#             jl_decoders = np.random.RandomState(seed=333).randn(n_neurons, jl_dims) * 1e-4
#             conn.oracle_decoders = np.hstack((conn.oracle_decoders, jl_decoders))