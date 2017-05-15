from nengo import Connection as NengoConnection

__all__ = ['BioConnection']


class BioConnection(NengoConnection):
    """
    Extends nengo.Connection to take additional parameters
    and support oracle decoder updating
    """

    def __init__(self, pre, post, syn_sec={'apical'}, n_syn=1,
                 weights_bias_conn=False, decoders_bio=None, 
                 synaptic_encoders=False, synaptic_gains=False, 
                 trained_weights=False, **kwargs):
        """
        syn_sec: the section(s) of the NEURON model on which
                    to distribute synapses
        n_syn: number of synapses on the bioneuron per presynaptic neuron
        weight_bias_conn: use this connection to emulate bioneuron biases
        """
        self.syn_sec = syn_sec
        self.n_syn = n_syn
        self.weights_bias_conn = weights_bias_conn
        self.decoders_bio = decoders_bio
        self.synaptic_encoders = synaptic_encoders
        self.synaptic_gains = synaptic_gains
        self.trained_weights = trained_weights

        super(BioConnection, self).__init__(pre, post, **kwargs)
