*************************************************************************
BIOnengo - incorporating biologically-realistic NEURON models into the NEF
*************************************************************************

Description
===========

This respository includes methods for training synaptic weights into an emsemble of biologically realistic neurons such that it performs representation and dynamics according to the NEF. These 'bioneurons' are implemented in NEURON based off of a reduced model of a pyramidal neuron by Bahl et al (2012). The core code defines a bioneuron class that includes the NEURON objects needed to track voltage, spikes, etc, as well as the required nengo methods to run these cells in simulation. The code also redefines some classes in the nengo builder to allow construction of bioensembles and transmission of spikes to the bioneurons' synapses. During the build, synapses are created at specified locations on the bioneurons' dendrites, and are assigned a synaptic weight that has been is trained using either a spike-matching approach or the oracle method. The repo also provides tests for the efficacy of these procedures by finding the error in the bioensemble's decoded output for various choices of encoder, decoder, and dynamical system.


Install NEURON and other dependencies
=====================================

change --prefix to another directory. If you're using virtualenv and virtualenvwrapper, this will be your your /HOME/USER/.local/directory

    pip install nengo matplotlib seaborn numpy pandas

    wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.tar.gz
    
    tar xzf nrn-7.4.tar.gz
    
    cd nrn-7.4
    
    ./configure --prefix=/home/$USER/.local --without-iv --with-nrnpython
    
    make
    
    (sudo) make install
    
    cd src/nrnpython
    
    python setup.py install
    
    cd ../../..
    
Install channel mechanisms for the bahl.hoc NEURON model
========================================================

To run the NEURON model, bahl.hoc, you must download the .hoc file and .mod files for every ion channel present in the model. These come with the repo, but must be compiled with the NEURON you've just installed locally.

Be sure to change the path to your .local directory to compile the NEURON channel mechanisms.

    git clone https://github.com/psipeter/bioneuron_oracle.git
    
    cd bioneuron_oracle/NEURON_models/channels/
    
    /home/$USER/.local/x86_64/bin/nrnivmodl
    
If the python kernel crashes at some point and you get an error in the terminal that says "NEURON: syntax error [...] insert ih", it means that the channels weren't initialized properly.
