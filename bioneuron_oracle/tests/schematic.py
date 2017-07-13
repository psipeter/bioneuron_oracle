import nengo
import numpy

with nengo.Network() as model:
    def f(x):
        return x
        
    stim = nengo.Node(0)
    pre = nengo.Ensemble(1,1)
    bio = nengo.Ensemble(1,1)
    ideal = nengo.Ensemble(1,1)
    oracle = nengo.Node(size_in=1)
    a_bio = nengo.Node(size_in=1)
    a_ideal = nengo.Node(size_in=1)
    x_target = nengo.Node(size_in=1)
    
    nengo.Connection(stim,pre)
    nengo.Connection(stim,oracle, function=f)
    nengo.Connection(pre,bio)
    nengo.Connection(pre,ideal)
    nengo.Connection(bio,a_bio)
    nengo.Connection(ideal,a_ideal)
    nengo.Connection(oracle,x_target)