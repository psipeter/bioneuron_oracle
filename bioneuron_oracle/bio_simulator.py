from nengo import Simulator as NengoSimulator
from nengo.ensemble import Neurons

from bioneuron_oracle.bahl_neuron import BahlNeuron

__all__ = ['BioSimulator']


class BioSimulator(NengoSimulator):

    def close(self):
        for key, value in self.data.iteritems():
            if isinstance(key, Neurons) and \
               isinstance(key.ensemble.neuron_type, BahlNeuron):
                for bahl in value:
                    bahl.cleanup()
        return super(BioSimulator, self).close()
