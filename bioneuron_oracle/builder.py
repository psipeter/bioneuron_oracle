import nengo
import numpy as np
import neuron
from nengo.builder import Builder, Operator, Signal
from nengo.dists import get_samples
from nengo.builder.connection import build_decoders
from nengo.builder.connection import BuiltConnection
from nengo.builder.ensemble import get_activities
from BahlNeuron import BahlNeuron, Bahl, ExpSyn
from nengo.exceptions import NengoException, BuildError


class SimBahlNeuron(Operator):
    """
    Operator to simulate the states of a bioensemble through time.
    """

    def __init__(self, neuron_type, n_neurons, output, voltage, states):
        super(SimBahlNeuron, self).__init__()
        self.neuron_type = neuron_type
        self.reads = [states[0]]
        self.sets = [output, voltage]
        self.updates = []
        self.incs = []
        self.neuron_type.neurons = [Bahl() for _ in range(n_neurons)]
        # self.inputs stores the input decoders, connection weights,
        # synaptic locations, and synaptic filter (tau)
        # for each connection into bionrn
        self.inputs = {}

    def make_step(self, signals, dt, rng):
        output = signals[self.output]
        voltage = signals[self.voltage]
        time = signals[self.time]

        def step_nrn():
            self.neuron_type.step_math(dt, output, self.neuron_type.neurons,
                                       voltage, time)
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
def build_bahlneuron(model, neuron_type, ens):
    model.sig[ens]['voltage'] = Signal(np.zeros(ens.ensemble.n_neurons),
                                       name='%s.voltage' % ens.ensemble.label)
    op = SimBahlNeuron(neuron_type=neuron_type,
                       n_neurons=ens.ensemble.n_neurons,
                       output=model.sig[ens]['out'],
                       voltage=model.sig[ens]['voltage'],
                       states=[model.time])
    # Initialize specific encoders and gains,
    # if this hasn't already been done
    # todo: should this go in build_bahlneuron() or SimBahlNeuron() instead?
    if not hasattr(ens.ensemble, 'encoders') or\
       not isinstance(ens.ensemble.encoders, np.ndarray):
            encoders, gains = gen_encoders_gains(
                ens.ensemble.n_neurons, ens.ensemble.dimensions,
                ens.ensemble.seed)
            ens.ensemble.encoders = encoders
            ens.ensemble.gain = gains
    model.add_op(op)


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
    conn_into_bioneuron = (
        isinstance(conn.post, nengo.Ensemble) and
        isinstance(conn.post.neuron_type, BahlNeuron))
    if hasattr(conn.pre, 'neuron_type'):
        conn_out_bioneuron = (
            isinstance(conn.pre.neuron_type, BahlNeuron) and
            isinstance(conn.post, nengo.Ensemble))
    else:
        conn_out_bioneuron = False

    if conn_into_bioneuron:
        # conn.pre must output spikes to connect to bioneurons
        if not hasattr(conn.pre, 'neuron_type'):
            raise BuildError("%s must transmit spikes to bioneurons in %s"\
                             % (conn.pre, conn.post))
            # raise BioConnectionError(conn.pre, conn.post)
                # 
        if not 'spikes' in conn.pre.neuron_type.probeable:
            raise BuildError("%s must transmit spikes to bioneurons in %s"\
                 % (conn.pre, conn.post))
            # raise BioConnectionError(conn.pre, conn.post)
                   # "input %s to bioensemble %s must transmit spikes"\
                   # % (conn.pre, conn.post))
        # Todo: other error handling
        rng = np.random.RandomState(model.seeds[conn])
        model.sig[conn]['in'] = model.sig[conn.pre]['out']
        transform = get_samples(conn.transform, conn.size_out,
                                d=conn.size_mid, rng=rng)

        """
        Given a parcicular connection, labeled by id(conn.pre),
        compute the optimal decoders, generate locations for synapses,
        then create a synapse with weight equal to
        w_ij=np.dot(d_i,alpha_j*e_j)+w_bias, where
            - d_i is the optimal presynaptic decoder,
            - e_j is the single bioneuron encoder
            - w_bias is a weight perturbation that emulates bias
        Afterwards add synapses to bioneuron.synapses and call neuron.init().
        """
        conn.syn_loc = get_synaptic_locations(
            rng, conn.pre.n_neurons, conn.post.n_neurons,
            conn.syn_sec, conn.n_syn, seed=model.seeds[conn])
        conn.syn_weights = np.zeros((
            conn.post.n_neurons, conn.pre.n_neurons, conn.syn_loc.shape[2]))
        # emulated biases in weight space
        if conn.weights_bias_conn:
            conn.weights_bias = gen_weights_bias(
                conn.pre.n_neurons, conn.post.n_neurons, conn.pre.dimensions,
                conn.post.dimensions, conn.pre.seed, conn.post.seed)
        eval_points, weights, solver_info = build_decoders(
            model, conn, rng, transform)

        for bionrn in range(len(conn.post.neuron_type.neurons)):
            bioneuron = conn.post.neuron_type.neurons[bionrn]
            d_in = weights.T  # untested for BIO-BIO connections
            loc = conn.syn_loc[bionrn]
            if hasattr(conn,'weights_bias'):  
                w_bias = conn.weights_bias[:,bionrn]
            tau = conn.synapse.tau
            encoder = conn.post.encoders[bionrn]
            gain = conn.post.gain[bionrn]
            bioneuron.synapses[id(conn.pre)] = np.empty((loc.shape[0],loc.shape[1]),dtype=object)
            for pre in range(loc.shape[0]):
                for syn in range(loc.shape[1]):
                    section = bioneuron.cell.apical(loc[pre,syn])
                    w_ij = np.dot(d_in[pre],gain*encoder)
                    if hasattr(conn,'weights_bias'):  
                        w_ij += w_bias[pre]
                    w_ij /= conn.n_syn  # todo: better n_syn scaling
                    conn.syn_weights[bionrn,pre,syn] = w_ij  # update full weight matrix
                    synapse = ExpSyn(section,w_ij,tau)  # create the synapse
                    bioneuron.synapses[id(conn.pre)][pre][syn] = synapse            
        neuron.init()    
        
        model.add_op(TransmitSpikes(
            id(conn.pre), id(conn.post), conn.post.neuron_type.neurons,
            model.sig[conn.pre]['out'], states=[model.time]))
        # note: transform, weights do not affect X-to-BIO connections (todo: untested)
        model.params[conn] = BuiltConnection(eval_points=eval_points,
                                                solver_info=solver_info,
                                                transform=transform,
                                                weights=conn.syn_weights)

    if conn_out_bioneuron:
        rng = np.random.RandomState(model.seeds[conn])
        model.sig[conn]['in'] = model.sig[conn.pre]['out']
        transform = get_samples(conn.transform, conn.size_out,
                                d=conn.size_mid, rng=rng)
        # Update decoders out of bioensembles, presumably by using the Oracle method.
        eval_points, weights, solver_info = build_decoders(
            model, conn, rng, transform)
        if conn.bio_decoders is not None:
            #todo: error handing for correct shape
            weights = conn.bio_decoders
        model.params[conn] = BuiltConnection(eval_points=eval_points,
                                            solver_info=solver_info,
                                            transform=transform,
                                            weights=weights)
    else: #normal connection
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
    Y = np.tile(biases, (pre_activities.shape[0],1))
    solver = nengo.solvers.LstsqL2(reg=0.01)  # todo: test other solvers?
    weights_bias, info = solver(pre_activities,Y)
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
    syn_locations=rng.uniform(0,1,size=(n_neurons,pre_neurons,n_syn))
    return syn_locations


class BioConnectionError(BuildError):
    """
    An error occured while building connections into
    or out of a bioensemble
    """
    # def __init__(self, pre, post):
    #     self.pre = pre
    #     self.post = post