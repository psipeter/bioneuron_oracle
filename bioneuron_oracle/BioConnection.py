# I'm probably just missing something simple here with the monkeypatch.
# I modeled it off the links you put in the googledoc, but I keep getting
# maximum recursion depth exceeded errors when I call the super() constructor.
from nengo import Connection as NengoConnection
from nengo.solvers import LstsqL2

class BioConnection(NengoConnection):
    """
    Extends nengo.Connection to take additional parameters
    and support oracle decoder updating
    """
    def __init__(self, pre, post, solver=LstsqL2(), 
                 syn_sec={'apical'},  n_syn=1,
                 weights_bias_conn=True, bio_decoders=None, **kwargs):
        """
        syn_sec: the section(s) of the NEURON model on which to distribute synapses
        n_syn: number of synapses on the bioneuron per presynaptic neuron
        weight_bias_conn: (bool) use this connection to emulate bioneuron biases
        """
#         self.pre = pre
#         self.post = post
#         self.solver = solver
        self.syn_sec = syn_sec
        self.n_syn = n_syn
        self.weights_bias_conn = weights_bias_conn
        self.bio_decoders = bio_decoders
        super(BioConnection, self).__init__(pre, post, solver=solver, **kwargs)  # todo: recursion depth error