import numpy as np
import nengo
from nengolib.signal import s, z

__all__ = ['prime_sinusoids', 'step_input', 'equalpower', 'get_signal', 'get_stim_deriv']


def get_signal(
    signal_type,
    network_seed=1,
    sim_seed=1,
    freq=1,
    seeds=1,
    t_transient=1.0,
    t_final=1.0,
    max_freq=1,
    rms=1,
    tau=0.1,
    dt=0.001):

    with nengo.Network(seed=network_seed) as network:
        if signal_type == 'sinusoids':
            stim = nengo.Node(lambda t: 
                np.cos(2 * np.pi * freq * (t-1)))
        elif signal_type == 'white_noise':
            stim = nengo.Node(nengo.processes.WhiteSignal(
                period=t_final, high=max_freq, rms=rms, seed=seeds))
        deriv = nengo.Node(size_in=1)
        nengo.Connection(stim, deriv, synapse=(1.0 - ~z) / dt)
        p_stim = nengo.Probe(stim, synapse=None)
        p_deriv = nengo.Probe(deriv, synapse=None)
    with nengo.Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_final)
    transient = np.array(0.0 * np.arange(0, t_transient+dt, dt))
    stimulus = np.hstack((transient, sim.data[p_stim][:,0]))
    derivative = np.hstack((transient, sim.data[p_deriv][:,0]))
    return stimulus, derivative

def get_stim_deriv(
    signal,
    order=1,
    network_seed=1,
    sim_seed=1,
    freq=1,
    seeds=1,
    t_final=1.0,
    max_freq=1,
    rms=1,
    tau=0.1,
    dt=0.001):

    with nengo.Network(seed=network_seed) as network:
        if signal == 'sinusoids':
            stim = nengo.Node(lambda t: 
                np.cos(2 * np.pi * freq * (t-1)))
        elif signal == 'white_noise':
            stim = nengo.Node(nengo.processes.WhiteSignal(
                period=t_final, high=max_freq, rms=rms, seed=seeds))
        deriv_node = nengo.Node(size_in=1)
        nengo.Connection(stim, deriv_node, synapse=((1.0 - ~z) / dt)**order)
        deriv_probe = nengo.Probe(deriv_node, synapse=tau)
    with nengo.Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_final)
    deriv_signal = sim.data[deriv_probe]
    deriv_mag = max(abs(np.min(deriv_signal)), abs(np.max(deriv_signal)))
    deriv_trans = 1.0 / deriv_mag
    return deriv_trans

def get_stim_extra(
    signal,
    network_seed=1,
    sim_seed=1,
    freq=1,
    seeds=1,
    t_final=1.0,
    max_freq=1,
    rms=1,
    tau=0.1,
    dt=0.001):

    with nengo.Network(seed=network_seed) as network:
        if signal == 'sinusoids':
            stim = nengo.Node(lambda t: 
                np.cos(2 * np.pi * freq * (t-1)))
        elif signal == 'white_noise':
            stim = nengo.Node(nengo.processes.WhiteSignal(
                period=t_final, high=max_freq, rms=rms, seed=seeds))
        extra_node = nengo.Node(size_in=1)
        nengo.Connection(stim, extra_node, synapse=(tau*(1.0 - ~z) / dt) + 1)
        extra_probe = nengo.Probe(extra_node, synapse=tau)
    with nengo.Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
        sim.run(t_final)
    extra_signal = sim.data[extra_probe]
    extra_mag = max(abs(np.min(extra_signal)), abs(np.max(extra_signal)))
    extra_trans = 1.0 / extra_mag
    return extra_trans

def prime_sinusoids(t, dim, t_final, f_0=1):
    # todo: generate primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
              47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
              103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
              157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    our_primes = primes[:dim]
    t_per_dim = t_final/dim
    epoch = int(t/t_per_dim)
    reordered_primes = [our_primes[(p + epoch) % len(our_primes)]
                        for p in range(len(our_primes))]
    frequencies = np.pi * np.array(reordered_primes)
    # values = np.array([np.sin(f_0 * w * t) for w in frequencies])
    # use cos() since integral will have pos and neg values
    values = np.array([np.cos(f_0 * w * t) for w in frequencies])
    return values


def step_input(t, dim, t_final, dt, n_eval_points=10):

    """
    Creates an (dim x n_eval_points) array where each row contains
    a uniform tiling from -1 to 1.
    To sample the dim-dimensional space, take every combination of values
    in this array.
    To do this, assume that each value is played for t_per_value time,
    and begin by taking eval_points[d,0] for each dimension.
    Then leave all but 1 dimension the same, and take
    eval_points[dim,1] from that dimension.
    Repeat until all values of the d_last dimesion are sampled.
    Then take eval_points[d,0] for d=0...dim-1, eval_points[dim-1,1]
    for the 2nd to last dimension,and repeat recursively up the array.

    n_eval_points: number of evaluation points to sample for each dimension
    """

    n_vals = n_eval_points ** dim
    n_times = int(t_final/dt)
    assert n_vals < n_times, "must be at least one timestep per constant value"
    t_per_val = int(n_times / n_vals)
    x_vals = np.linspace(-1, 1, n_eval_points)
    eval_points = np.zeros((dim, n_eval_points))
    for d in range(dim):
        eval_points[d] = x_vals

    def get_eval_point(t, eval_points):
        t_idx = int(t / dt)
        idxs = np.zeros((dim)).astype(int)
        for d in range(dim):
            idxs[d] = int(t_idx / t_per_val / (n_eval_points ** d))\
                % n_eval_points
        return [eval_points[d, idxs[d]] for d in range(dim)]

    return np.array(get_eval_point(t, eval_points))


def equalpower(t, dt, t_final, max_freq, dim, mean=0.0, std=1.0, seed=None):
    # from functools32 import lru_cache
    # @lru_cache(maxsize=None)
    def gen_equalpower(dt, t_final, max_freq, mean, std, n=None, seed=seed):
        """
        Eric Hunsberger's Code for his 2016 Tech Report

        Generate a random signal with equal power below a maximum frequency
        Parameters
        ----------
        dt : float
            Time difference between consecutive signal points [in seconds]
        t_final : float
            Length of the signal [in seconds]
        max_freq : float
            Maximum frequency of the signal [in Hertz]
        mean : float
            Signal mean (default = 0.0)
        std : float
            Signal standard deviation (default = 1.0)
        n : integer
            Number of signals to generate
        Returns
        -------
        s : array_like
            Generated signal(s), where each row is a time, and each column a
            signal
        """
        import numpy.random as npr

        rng = npr.RandomState(seed=seed)
        vector_out = n is None
        n = 1 if n is None else n

        df = 1. / t_final    # fundamental frequency

        nt = int(np.round(t_final / dt))  # number of time points / frequencies
        nf = int(np.round(max_freq / df))  # number of non-zero frequencies
        assert nf < nt

        theta = 2*np.pi*rng.rand(n, nf)
        B = np.cos(theta) + 1.0j * np.sin(theta)

        A = np.zeros((n, nt), dtype=np.complex)
        A[:, 1:nf+1] = B
        A[:, -nf:] = np.conj(B)[:, ::-1]

        S = np.fft.ifft(A, axis=1).real

        S = (std / S.std(axis=1))[:, None] * (
            S - S.mean(axis=1)[:, None] + mean)
        if vector_out:
            return S.flatten()
        else:
            return S

    t_idx = int(t / dt)  # time index
    out = []
    for d in range(dim):
        signal = gen_equalpower(dt, t_final+dt, max_freq, mean, std, 1, (d+1)*seed)
        value = signal[0][t_idx]
        out.append(value)

    return np.array(out)
