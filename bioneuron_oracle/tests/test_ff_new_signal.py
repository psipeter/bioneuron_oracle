import nengo
import numpy as np
import neuron
from bioneuron_oracle.BahlNeuron import BahlNeuron, Bahl, ExpSyn
from bioneuron_oracle.custom_signals import prime_sinusoids, step_input
from nengo.utils.matplotlib import rasterplot
# import matplotlib.pyplot as plt
import seaborn as sns
import pytest
from bioneuron_oracle.feedforward import feedforward


def test_feedforward_new_signal(Simulator, plt, seed):
    pre_neurons=100
    bio_neurons=50
    tau_nengo=0.01
    tau_neuron=0.01
    dt_nengo=0.001
    dt_neuron=0.0001
    bio_seed=6
    t_final=0.5
    dim=2
    n_syn=5
    signal='prime_sinusoids'

    """Basic functionality"""
    pre_seed=3
    decoders_bio=None
    d_1, rmse_bio, rmse_lif = feedforward(
        pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
        dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
        decoders_bio, plots, plt=plt, plt_name='basic')

    """New LIF (seed), new signal, old decoders"""
    pre_seed=9
    decoders_bio=d_1
    signal='step_input'
    plots={'decode'}
    d_4, rmse_bio, rmse_lif = feedforward(
        pre_neurons, bio_neurons, tau_nengo, tau_neuron, dt_nengo, 
        dt_neuron, pre_seed, bio_seed, t_final, dim, signal, 
        decoders_bio, plots, plt=plt, plt_name='signal')
    assert(rmse_bio < 0.4)
