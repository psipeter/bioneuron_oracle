import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from bioneuron_oracle.solver import BioSolver
from nengo.utils.numpy import rmse
import seaborn as sns
from nengolib.signal import s


def test_integrator(plt):

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

    pre_neurons=100
    bio_neurons=50
    post_neurons=100
    tau_nengo=0.1
    tau_neuron=0.1
    dt_nengo=0.001
    dt_neuron=0.0001
    lif_seed=3
    bio_seed=6
    pre_seed=9
    t_final=1.0
    dim=1
    jl_dims=0
    n_syn=10
    signal='prime_sinusoids'
    decoders_bio=None
    cutoff=0.1

    def sim(w_train, decoders_bio=None,plots=False):

        if decoders_bio is None:
            decoders_bio=np.zeros((bio_neurons, dim))

        with nengo.Network() as model:

            stim = nengo.Node(lambda t: prime_sinusoids(t, dim, t_final))

            lif = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                seed=lif_seed, neuron_type=nengo.LIF())
            pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=dim,
                                seed=pre_seed, neuron_type=nengo.LIF(),
                                max_rates=nengo.dists.Uniform(50,100),
                                intercepts=nengo.dists.Uniform(-1.0, 1.0))
            bio = nengo.Ensemble(n_neurons=bio_neurons, dimensions=dim+jl_dims,
                                seed=bio_seed, neuron_type=BahlNeuron())
            integral = nengo.Node(size_in=1)

            bio_solver = BioSolver(decoders_bio=(1.0 - w_train) * decoders_bio)
            if jl_dims > 0:
	            # todo: magnitude should scale with n_neurons (maybe 1./n^2)?
	            jl_decoders = np.random.RandomState(seed=333).randn(bio_neurons, jl_dims) * 1e-4
	            bio_solver.decoders_bio = np.hstack((bio_solver.decoders_bio, jl_decoders))

            nengo.Connection(stim, lif, synapse=None)
            # connect feedforward to non-jl_dims of bio
            nengo.Connection(lif, bio[:dim], weights_bias_conn=True,
                         	 synapse=tau_neuron, transform=tau_neuron)
            nengo.Connection(bio, bio, synapse=tau_neuron, solver=bio_solver)

            nengo.Connection(stim, integral, synapse=1/s)  # integrator
            nengo.Connection(integral, pre, synapse=None)
            nengo.Connection(pre, bio[:dim],  # oracle connection
                             synapse=tau_neuron, transform=w_train)

            probe_stim = nengo.Probe(stim, synapse=None)
            probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
            probe_integral = nengo.Probe(integral, synapse=None)

        with nengo.Simulator(model,dt=dt_nengo) as sim:
            sim.run(t_final)

        lpf = nengo.Lowpass(tau_nengo)
        act_bio = lpf.filt(sim.data[probe_bio_spikes], dt=dt_nengo)
        assert np.sum(act_bio) > 0.0
        target = sim.data[probe_integral]
        solver = nengo.solvers.LstsqL2(reg=0.1)
        decoders_bio_new, info = solver(act_bio, target)
        xhat_bio = np.dot(act_bio, decoders_bio_new)
        rmse_bio = rmse(xhat_bio, target)

        if plots:
            sns.set(context='poster')
            plt.subplot(1,1,1)
            plt.plot(sim.trange(), sim.data[probe_stim], label='stim')
            plt.plot(sim.trange(), xhat_bio, label='bio, rmse=%.5f' % rmse_bio)
            plt.plot(sim.trange(), target, label='target')
            plt.xlabel('time (s)')
            plt.ylabel('$\hat{x}(t)$')
            plt.title('decode')
            legend3 = plt.legend()

        return decoders_bio_new, rmse_bio

    decoders_bio_new, rmse_bio = sim(w_train=1.0, decoders_bio=None, plots=False)
    decoders_bio_new, rmse_bio = sim(w_train=0.0, decoders_bio=decoders_bio_new, plots=True)

    assert rmse_bio < cutoff
