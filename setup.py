#!/usr/bin/env python

import os
from setuptools import setup

name = 'bioneuron_oracle'
root = os.path.dirname(os.path.realpath(__file__))

def check_dependencies():
    install_requires = []
    try:
        import nengo
    except ImportError:
        install_requires.append('nengo')
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import neuron
    except ImportError:
        install_requires.append('NEURON')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    try:
        import seaborn
    except ImportError:
        install_requires.append('seaborn')
    return install_requires

def readme():
    with open('README.rst') as f:
        return f.read()

if __name__ == "__main__":
    install_requires = check_dependencies()
    setup(
        name=name,
        version='1.0',
        description='incorporate NEURON models into nengo',
        url='https://github.com/psipeter/bioneuron_oracle',
        author='Peter Duggins',
        author_email='psipeter@gmail.com',
        packages=['bioneuron_oracle'],
        long_description=readme(),
        install_requires=install_requires,
        include_package_data=True,
        zip_safe=False
    )
