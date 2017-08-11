import numpy as np

import neuron

import nengo
from nengo.base import ObjView
from nengo.builder import Builder, Operator, Signal
from nengo.builder.connection import build_decoders, BuiltConnection
from nengo.builder.ensemble import get_activities
from nengo.exceptions import BuildError
from nengo.utils.builder import full_transform

from bioneuron_oracle.bahl_neuron import BahlNeuron
from bioneuron_oracle.solver import OracleSolver, TrainedSolver

__all__ = []


class Bahl(object):
    """Adaptor between step_math and NEURON."""

    def __init__(self):
        super(Bahl, self).__init__()
        self.synapses = {}
        self.cell = neuron.h.Bahl()
        self.v_record = neuron.h.Vector()
        self.v_record.record(self.cell.soma(0.5)._ref_v)
        self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
        self.t_record = neuron.h.Vector()
        self.t_record.record(neuron.h._ref_t)
        self.spikes = neuron.h.Vector()
        self.ap_counter.record(neuron.h.ref(self.spikes))
        self.num_spikes_last = 0
        self._clean = False

    def update(self):
        if self._clean:
            raise RuntimeError("cannot update() after cleanup()")
        count = len(self.spikes) - self.num_spikes_last
        self.num_spikes_last = len(self.spikes)
        return count, np.asarray(self.v_record)[-1]

    def cleanup(self):
        if self._clean:
            raise RuntimeError("cleanup() may only be called once")
        self.v_record.play_remove()
        self.t_record.play_remove()
        self.spikes.play_remove()
        self._clean = True


class ExpSyn(object):
    """
    Conductance-based synapses.
    There are two types, excitatory and inhibitory,
    with different reversal potentials.
    If the synaptic weight is above zero, initialize an excitatory synapse,
    else initialize an inhibitory syanpse with abs(weight).
    """

    def __init__(self, sec, weight, tau, loc, e_exc=0.0, e_inh=-80.0):
        self.tau = tau
        self.loc = loc
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.ExpSyn(sec)
        self.syn.tau = 1000*self.tau  # no more 2x multiply
        self.weight = weight
        if self.weight >= 0.0:
            self.syn.e = self.e_exc
        else:
            self.syn.e = self.e_inh
        # time of spike arrival assigned in nengo step
        self.spike_in = neuron.h.NetCon(None, self.syn)        
        self.spike_in.weight[0] = abs(self.weight)


class SimBahlNeuron(Operator):
    """
    Operator to simulate the states of a bioensemble through time.
    """

    def __init__(self, neuron_type, neurons, output, voltage, states):
        super(SimBahlNeuron, self).__init__()
        self.neuron_type = neuron_type
        self.neurons = neurons

        self.reads = [states[0]]
        self.sets = [output, voltage]
        self.updates = []
        self.incs = []

    def make_step(self, signals, dt, rng):
        output = signals[self.output]
        voltage = signals[self.voltage]
        time = signals[self.time]

        def step_nrn():
            self.neuron_type.step_math(dt, output, self.neurons, voltage, time)
        return step_nrn

    @property
    def time(self):
        return self.reads[0]

    @property
    def output(self):
        return self.sets[0]

    @property
    def voltage(self):
        return self.sets[1]


class TransmitSpikes(Operator):
    """
    Operator to deliver spikes from the presynaptic population
    into a bioensemble.
    """

    def __init__(self, ens_pre, ens_post, neurons, spikes, states):
        super(TransmitSpikes, self).__init__()
        self.ens_pre = ens_pre
        self.ens_post = ens_post
        self.neurons = neurons
        self.time = states[0]
        self.reads = [spikes, states[0]]
        self.updates = []
        self.sets = []
        self.incs = []

    @property
    def spikes(self):
        return self.reads[0]

    def make_step(self, signals, dt, rng):
        spikes = signals[self.spikes]
        time = signals[self.time]

        def step():
            t_neuron = (time.item()-dt)*1000
            for n in range(spikes.shape[0]):
                num_spikes = int(spikes[n]*dt + 1e-9)
                for _ in range(num_spikes):
                    for nrn in self.neurons:
                        for syn in nrn.synapses[self.ens_pre][n]:
                            syn.spike_in.event(t_neuron)
        return step


@Builder.register(BahlNeuron)
def build_bahlneuron(model, neuron_type, neurons):
    ens = neurons.ensemble
    bahl_neurons = [Bahl() for _ in range(ens.n_neurons)]
    neuron.init()

    model.sig[neurons]['voltage'] = Signal(
        np.zeros(ens.n_neurons),
        name='%s.voltage' % ens.label)
    op = SimBahlNeuron(neuron_type=neuron_type,
                       neurons=bahl_neurons,
                       output=model.sig[neurons]['out'],
                       voltage=model.sig[neurons]['voltage'],
                       states=[model.time])

    # Initialize specific encoders and gains,
    # if this hasn't already been done
    if (not hasattr(ens, 'encoders') or
            not isinstance(ens.encoders, np.ndarray)):
        rng = np.random.RandomState(seed=ens.seed)
        # encoders, gains = gen_encoders_gains_manual(
        #     ens.n_neurons,
        #     ens.dimensions,
        #     rng)
        encoders, gains = gen_encoders_gains_LIF(
            ens.n_neurons,
            ens.dimensions,
            ens.max_rates,
            ens.intercepts,
            ens.radius,
            ens.seed)
        ens.encoders = encoders
        ens.gain = gains
        ens.bias = np.zeros_like(gains)
        # Note: setting encoders/gains/biases in this way doesn't really
        # respect the high-level ordering of the nengo build process.
        # This can generate hard-to-track problems related to these attributes.
        # However, setting them like 'neurons' are set below may not be possible
        # because these attributes are used in more places in the build process.
    model.add_op(op)

    assert neurons not in model.params
    model.params[neurons] = bahl_neurons


@Builder.register(nengo.Connection)
def build_connection(model, conn):
    """
    Method to build connections into bioensembles.
    Calculates the optimal decoders for this conneciton as though
    the presynaptic ensemble was connecting to a hypothetical LIF ensemble.
    These decoders are used to calculate the synaptic weights
    in init_connection().
    Adds a transmit_spike operator for this connection to the model
    """

    def deref_objview(o):
        return o.obj if isinstance(o, ObjView) else o

    conn_pre = deref_objview(conn.pre)
    conn_post = deref_objview(conn.post)

    if isinstance(conn_pre, nengo.Ensemble) and \
       isinstance(conn_pre.neuron_type, BahlNeuron):
        if (not isinstance(conn.solver, OracleSolver) and 
                not isinstance(conn.solver, TrainedSolver) and
                hasattr(conn_post, 'label') and conn_post.label != 'temp_sub_spike_match'):  # TODO: less hacky
            raise BuildError("Connections from bioneurons must provide "
                             "a OracleSolver or TrainedSolver"
                            " (got %s from %s to %s)"
                             % (conn.solver, conn_pre, conn_post))

    if isinstance(conn_post, nengo.Ensemble) and \
       isinstance(conn_post.neuron_type, BahlNeuron):
        # conn_pre must output spikes to connect to bioneurons
        # TODO: other error handling?
        # TODO: detect this earlier inside BioConnection __init__
        if not isinstance(conn_pre, nengo.Ensemble) or \
           'spikes' not in conn_pre.neuron_type.probeable:
            raise BuildError("May only connect spiking neurons (pre=%s) to "
                             "bioneurons (post=%s)" % (conn_pre, conn_post))

        rng = np.random.RandomState(model.seeds[conn])
        model.sig[conn]['in'] = model.sig[conn_pre]['out']
        transform = full_transform(conn, slice_pre=False)

        """
        Given a parcicular connection, labeled by conn.pre,
        compute the optimal decoders, generate locations for synapses,
        then create a synapse with weight equal to
        w_ij=np.dot(d_i,alpha_j*e_j)+w_bias, where
            - d_i is the presynaptic decoder computed by a nengo solver,
            - e_j is the single bioneuron encoder
            - w_bias is a weight perturbation that emulates bias
        Afterwards add synapses to bioneuron.synapses and call neuron.init().
        """
        # initialize synaptic locations and weights
        syn_loc = get_synaptic_locations(
            rng, conn_pre.n_neurons, conn_post.n_neurons,
            conn.syn_sec, conn.n_syn, seed=model.seeds[conn])
        syn_weights = np.zeros((
            conn_post.n_neurons, conn_pre.n_neurons, syn_loc.shape[2]))

        # emulated biases in weight space
        if conn.weights_bias_conn:
            # weights_bias = gen_weights_bias_manual(
            #     conn_pre.n_neurons,
            #     conn_post.n_neurons,
            #     rng)
            weights_bias = gen_weights_bias_LIF(
                conn_pre.n_neurons,
                conn_pre.dimensions,
                conn_pre.max_rates,
                conn_pre.intercepts,
                conn_pre.radius,
                conn_pre.seed,
                conn_post.n_neurons,
                conn_post.dimensions,
                conn_post.max_rates,
                conn_post.intercepts,
                conn_post.radius,
                conn_post.seed)

        # Grab decoders from this connections OracleSolver
        # TODO: fails for slicing into TrainedSolver (?)
        eval_points, weights, solver_info = build_decoders(
                model, conn, rng, transform)

        # unit test that synapse and weight arrays are compatible shapes
        if (conn.weights_bias_conn and
                not syn_loc.shape[:-1] == weights_bias.T.shape):
            raise BuildError("Shape mismatch: syn_loc=%s, weights_bias=%s"
                             % (syn_loc.shape[:-1], weights_bias))

        if conn.trained_weights:
            weights = solver_info['weights_bio']  # TODO: better workaround to grab correctly shaped weights_bio
            if (weights.ndim != 3 or
                weights.shape != (conn_post.n_neurons, conn_pre.n_neurons,
                                  conn.n_syn)):
                raise BuildError("Bad weight matrix shape: expected %s, got %s"
                                 % ((conn_post.n_neurons, conn_pre.n_neurons,
                                    conn.n_syn), weights.shape))
            if isinstance(conn.post, ObjView):
                raise BuildError("Slicing into bioneurons with\
                    spike-match-trained weights is not implemented:\
                    these bioneurons don't have encoders")

        # normalize the area under the ExpSyn curve to compensate for effect of tau
        times = np.arange(0, 1.0, 0.001)  # to 1s by dt=0.001
        k_norm = np.linalg.norm(np.exp((-times/conn.synapse.tau)),1)
        # print np.sum(np.exp(-times/conn.synapse.tau)/k_norm)
        # print k_norm

        neurons = model.params[conn_post.neurons]  # set in build_bahlneuron
        if conn.trained_weights:  # hyperopt trained weights
            for j, bahl in enumerate(neurons):
                assert isinstance(bahl, Bahl)
                loc = syn_loc[j]            
                tau = conn.synapse.tau
                bahl.synapses[conn_pre] = np.empty(
                    (loc.shape[0], loc.shape[1]), dtype=object)
                for pre in range(loc.shape[0]):
                    for syn in range(loc.shape[1]):
                        section = bahl.cell.apical(loc[pre, syn])
                        w_ij = weights[j, pre, syn] / conn.n_syn / k_norm # TODO: better n_syn scaling
                        syn_weights[j, pre, syn] = w_ij
                        synapse = ExpSyn(section, w_ij, tau, loc[pre, syn])
                        bahl.synapses[conn_pre][pre][syn] = synapse

        else:  # oracle weights
            for j, bahl in enumerate(neurons):
                assert isinstance(bahl, Bahl)
                d_in = 1e+2 * weights.T * 1e-1
                loc = syn_loc[j]
                if conn.weights_bias_conn:
                    w_bias = weights_bias[:, j]
                tau = conn.synapse.tau
                encoder = conn_post.encoders[j]
                gain = conn_post.gain[j]
                bahl.synapses[conn_pre] = np.empty(
                    (loc.shape[0], loc.shape[1]), dtype=object)
                for pre in range(loc.shape[0]):
                    # if conn.synaptic_encoders or conn.synaptic_gains:  # todo
                    #     seed = conn_post.seed + j + pre
                    #     n_syn = loc.shape[0] * loc.shape[1]
                    #     dim = conn_post.dimensions
                    #     syn_encoders, syn_gains = gen_encoders_gains_LIF(n_syn, dim, seed)
                    for syn in range(loc.shape[1]):
                        section = bahl.cell.apical(loc[pre, syn])
                        # if conn.synaptic_encoders or conn.synaptic_gains:  # todo
                        #     encoder = syn_encoders[syn]
                        #     gain = syn_gains[syn]
                        w_ij = np.dot(d_in[pre], gain * encoder)
                        if conn.weights_bias_conn:
                            w_ij += w_bias[pre]
                        w_ij = w_ij / conn.n_syn / k_norm
                        syn_weights[j, pre, syn] = w_ij
                        synapse = ExpSyn(section, w_ij, tau, loc[pre, syn])
                        bahl.synapses[conn_pre][pre][syn] = synapse
        neuron.init()

        model.add_op(TransmitSpikes(
            conn_pre, conn_post, neurons,
            model.sig[conn_pre]['out'], states=[model.time]))
        model.params[conn] = BuiltConnection(eval_points=eval_points,
                                             solver_info=solver_info,
                                             transform=transform,
                                             weights=syn_weights)

    else:  # normal connection
        return nengo.builder.connection.build_connection(model, conn)

def gen_encoders_gains_manual(n_neurons, dimensions, rng):
    enc_mag = 1e+0  # todo: pass as parameter
    gain_mag = 1e+4
    encoders = rng.uniform(-enc_mag, enc_mag, size=(n_neurons, dimensions))
    # dist = nengo.dists.UniformHypersphere()
    # encoders = nengo.dists.get_samples(dist, n=n_neurons, d=dimensions, rng=rng)
    gains = rng.uniform(-gain_mag, gain_mag, size=n_neurons)
    return encoders, gains

def gen_weights_bias_manual(pre_n_neurons, post_n_neurons, rng):
    w_bias_mag = 1e-2 # todo: pass as parameter
    w_bias = rng.uniform(-w_bias_mag, w_bias_mag, size=(pre_n_neurons, post_n_neurons))
    return w_bias

def gen_encoders_gains_LIF(n_neurons,
                    dimensions,
                    max_rates,
                    intercepts,
                    radius,
                    seed):
    """
    Alternative to gain_bias() for bioneurons.
    Called in custom build_connections().
    """
    with nengo.Network(add_to_container=False) as pre_model:
        lif = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=dimensions,
                    neuron_type=nengo.LIF(),
                    max_rates=max_rates, 
                    intercepts=intercepts,
                    radius=radius,
                    seed=seed)
    with nengo.Simulator(pre_model) as pre_sim:
        encoders = pre_sim.data[lif].encoders
        gains = pre_sim.data[lif].gain
    return encoders, gains

def gen_weights_bias_LIF(pre_n_neurons,
                pre_dimensions,
                pre_max_rates,
                pre_intercepts,
                pre_radius,
                pre_seed,
                post_n_neurons,
                post_dimensions,
                post_max_rates,
                post_intercepts,
                post_radius,
                post_seed):
    """
    Build a pre-simulation network to draw biases from Nengo,
    then return a weight matrix that emulates the bias
    (by adding weights to the synaptic weights in init_connection().
    TODO: add max_rates and other scaling properties.
    """
    with nengo.Network(add_to_container=False) as pre_model:
        pre = nengo.Ensemble(
                n_neurons=pre_n_neurons,
                dimensions=pre_dimensions,
                max_rates=pre_max_rates,
                intercepts=pre_intercepts,
                radius=pre_radius,
                seed=pre_seed)
        lif = nengo.Ensemble(
                n_neurons=post_n_neurons,
                dimensions=post_dimensions,
                max_rates=post_max_rates,
                intercepts=post_intercepts,
                radius=post_radius,
                seed=post_seed,
                neuron_type=nengo.LIF())
    with nengo.Simulator(pre_model) as pre_sim:
        pre_activities = get_activities(pre_sim.data[pre], pre,
                                        pre_sim.data[pre].eval_points)
        biases = pre_sim.data[lif].bias
    # Desired output function Y -- just repeat "bias" m times
    Y = np.tile(biases, (pre_activities.shape[0], 1))
    # TODO: check weights vs decoders
    weights_bias = 7e+1 * nengo.solvers.LstsqL2(reg=0.01)(pre_activities, Y)[0] * 1e-1
    return weights_bias


def get_synaptic_locations(rng, pre_neurons, n_neurons,
                           syn_sec, n_syn, seed):
    """Choose one:"""
    # TODO: make syn_distribution an optional parameters of nengo.Connection
    # same locations per connection and per bioneuron
#     rng2=np.random.RandomState(seed=333)
#     syn_locations=np.array([rng2.uniform(0,1,size=(pre_neurons,n_syn))
#         for n in range(n_neurons)])
    # same locations per connection and unique locations per bioneuron
#     rng2=np.random.RandomState(seed=333)
#     syn_locations=rng2.uniform(0,1,size=(n_neurons,pre_neurons,n_syn))
    # unique locations per connection and per bioneuron (uses conn's rng)
    syn_locations = rng.uniform(0, 1, size=(n_neurons, pre_neurons, n_syn))
    return syn_locations