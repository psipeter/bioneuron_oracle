import nengo
import numpy as np
import neuron

def prime_sinusoids(t, dim, t_final):
    # todo: generate primes
    primes=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 
            47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
            103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 
            157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    our_primes=primes[:dim]
    t_per_dim=t_final/dim
    epoch=int(t/t_per_dim)
    reordered_primes=[our_primes[(p + epoch) % len(our_primes)] for p in range(len(our_primes))]
    frequencies=np.pi*np.array(reordered_primes)
    values = np.array([np.sin(w*t) for w in frequencies])
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

    n_vals=n_eval_points**dim
    n_times=int(t_final/dt)
    assert n_vals < n_times, "must be at least one timestep per constant value"
    t_per_val=int(n_times/n_vals)
    x_vals=np.linspace(-1,1,n_eval_points)
    eval_points=np.zeros((dim,n_eval_points))
    for d in range(dim):
        eval_points[d]=x_vals

    def get_eval_point(t,eval_points):
        t_idx=int(t/dt)
        idxs=np.zeros((dim)).astype(int)
        for d in range(dim):
            idxs[d]=int(t_idx / t_per_val / (n_eval_points ** d)) % n_eval_points
        return [eval_points[d,idxs[d]] for d in range(dim)]

    return np.array(get_eval_point(t,eval_points))