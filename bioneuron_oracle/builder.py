import numpy as np

import neuron

import nengo
from nengo.base import ObjView
from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Copy, DotInc, Reset
from nengo.builder.connection import (build_decoders, BuiltConnection)
from nengo.builder.ensemble import get_activities
from nengo.exceptions import BuildError
from nengo.utils.builder import full_transform

from bioneuron_oracle.bahl_neuron import BahlNeuron

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

    def __init__(self, sec, weight, tau, e_exc=0.0, e_inh=-80.0):
        self.tau = tau
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
        encoders, gains = gen_encoders_gains(
            ens.n_neurons, ens.dimensions, ens.seed)
        ens.encoders = encoders
        ens.gain = gains
    model.add_op(op)
    neuron.init()

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
            weights_bias = gen_weights_bias(
                conn_pre.n_neurons, conn_post.n_neurons, conn_pre.dimensions,
                conn_post.dimensions, conn_pre.seed, conn_post.seed)

        # Grab decoders from this connections BioSolver
        eval_points, weights, solver_info = build_decoders(
            model, conn, rng, transform)

        # unit test that synapse and weight arrays are compatible shapes
        # todo: more tests?
        if (conn.weights_bias_conn and
                not syn_loc.shape[:-1] == weights_bias.T.shape):
            raise BuildError("Shape mismatch: syn_loc=%s, weights_bias=%s"
                             % (syn_loc.shape[:-1], weights_bias))

        neurons = model.params[conn_post.neurons]  # set in build_bahlneuron
        for j, bahl in enumerate(neurons):
            assert isinstance(bahl, Bahl)
            d_in = weights.T
            loc = syn_loc[j]
            if conn.weights_bias_conn:
                w_bias = weights_bias[:, j]
            tau = conn.synapse.tau
            encoder = conn_post.encoders[j]
            gain = conn_post.gain[j]
            bahl.synapses[conn_pre] = np.empty(
                (loc.shape[0], loc.shape[1]), dtype=object)
            for pre in range(loc.shape[0]):
                for syn in range(loc.shape[1]):
                    section = bahl.cell.apical(loc[pre, syn])
                    w_ij = np.dot(d_in[pre], gain * encoder)
                    if conn.weights_bias_conn:
                        w_ij += w_bias[pre]
                    w_ij /= conn.n_syn  # TODO: better n_syn scaling
                    syn_weights[j, pre, syn] = w_ij
                    synapse = ExpSyn(section, w_ij, tau)
                    bahl.synapses[conn_pre][pre][syn] = synapse
        neuron.init()

        model.add_op(TransmitSpikes(
            conn_pre, conn_post, neurons,
            model.sig[conn_pre]['out'], states=[model.time]))
        model.params[conn] = BuiltConnection(eval_points=eval_points,
                                             solver_info=solver_info,
                                             transform=transform,
                                             weights=syn_weights)

    elif (isinstance(conn_pre, nengo.Ensemble) and
          isinstance(conn_pre.neuron_type, BahlNeuron)):
        # TODO: this is redundant with nengo's build_connection() except
        # for one line change, which may break things down the road,
        # but it was the only way I could get transforms and slilces
        # out of bioneurons to work
        rng = np.random.RandomState(model.seeds[conn])

        # Get input and output connections from pre and post
        def get_prepost_signal(is_pre):
            target = conn.pre_obj if is_pre else conn.post_obj
            key = 'out' if is_pre else 'in'

            if target not in model.sig:
                raise BuildError("Building %s: the %r object %s is not in the "
                                 "model, or has a size of zero."
                                 % (conn, 'pre' if is_pre else 'post', target))
            if key not in model.sig[target]:
                raise BuildError(
                    "Building %s: the %r object %s has a %r size of zero."
                    % (conn, 'pre' if is_pre else 'post', target, key))

            return model.sig[target][key]

        model.sig[conn]['in'] = get_prepost_signal(is_pre=True)
        model.sig[conn]['out'] = get_prepost_signal(is_pre=False)

        weights = None
        eval_points = None
        solver_info = None
        signal_size = conn.size_out
        post_slice = conn.post_slice
        transform = full_transform(conn, slice_pre=False)

        # Figure out the signal going across this connection
        in_signal = model.sig[conn]['in']
        eval_points, weights, solver_info = build_decoders(
            model, conn, rng, transform)
        if conn.solver.weights:
            model.sig[conn]['out'] = model.sig[conn.post_obj.neurons]['in']
            signal_size = conn.post_obj.neurons.size_in
            post_slice = None  # don't apply slice later

        # Add operator for applying weights
        model.sig[conn]['weights'] = Signal(
            weights, name="%s.weights" % conn, readonly=True)
        signal = Signal(np.zeros(signal_size), name="%s.weighted" % conn)
        model.add_op(Reset(signal))

        """ElementwiseInc breaks transforms out of bioensembles"""
        # op = ElementwiseInc if weights.ndim < 2 else DotInc
        op = DotInc

        model.add_op(op(model.sig[conn]['weights'],
                        in_signal,
                        signal,
                        tag="%s.weights_elementwiseinc" % conn))

        # Add operator for filtering
        if conn.synapse is not None:
            signal = model.build(conn.synapse, signal)

        # Store the weighted-filtered output in case we want to probe it
        model.sig[conn]['weighted'] = signal

        # Copy to the proper slice
        model.add_op(Copy(
            signal, model.sig[conn]['out'], dst_slice=post_slice,
            inc=True, tag="%s" % conn))

        model.params[conn] = BuiltConnection(eval_points=eval_points,
                                             solver_info=solver_info,
                                             transform=transform,
                                             weights=weights)

    else:  # normal connection
        return nengo.builder.connection.build_connection(model, conn)


def gen_encoders_gains(n_neurons, dimensions, seed):
    """
    Alternative to gain_bias() for bioneurons.
    Called in custom build_connections().
    TODO: add max_rates and other scaling properties.
    """
    with nengo.Network() as pre_model:
        lif = nengo.Ensemble(n_neurons=n_neurons, dimensions=dimensions,
                             neuron_type=nengo.LIF(), seed=seed)
    with nengo.Simulator(pre_model) as pre_sim:
        encoders = pre_sim.data[lif].encoders
        gains = pre_sim.data[lif].gain
    return encoders, gains


def gen_weights_bias(pre_neurons, n_neurons, in_dim,
                     out_dim, pre_seed, bio_seed):

    """
    Build a pre-simulation network to draw biases from Nengo,
    then return a weight matrix that emulates the bias
    (by adding weights to the synaptic weights in init_connection().
    """

    with nengo.Network(label='preliminary') as pre_model:
        pre = nengo.Ensemble(n_neurons=pre_neurons, dimensions=in_dim,
                             seed=pre_seed)
        lif = nengo.Ensemble(n_neurons=n_neurons, dimensions=out_dim,
                             neuron_type=nengo.LIF(), seed=bio_seed)
    with nengo.Simulator(pre_model) as pre_sim:
        pre_activities = get_activities(pre_sim.data[pre], pre,
                                        pre_sim.data[pre].eval_points)
        biases = pre_sim.data[lif].bias
    # Desired output function Y -- just repeat "bias" m times
    Y = np.tile(biases, (pre_activities.shape[0], 1))
    solver = nengo.solvers.LstsqL2(reg=0.01)  # todo: test other solvers?
    weights_bias, info = solver(pre_activities, Y)
    return weights_bias


def get_synaptic_locations(rng, pre_neurons, n_neurons,
                           syn_sec, n_syn, seed):
    """Choose one:"""
    # todo: make syn_distribution an optional parameters of nengo.Connection
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
