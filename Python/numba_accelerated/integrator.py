import numpy as np
import numba as nb
from numba import prange

"""
Some handy, numba compiled integration functions.
"""


@nb.jit(nopython=True, parallel=True)
def kick(v, a, dt):
    v += a * dt


@nb.jit(nopython=True, parallel=True)
def drift(x, v, dt):
    x += v * dt


@nb.jit(nopython=True, parallel=True)
def grav_bruteforce(m, p, a, g):
    # numba likes loops and numpy instructions
    # The outer loop can be parallelized, because the acceleration of particle i is independent of the acceleration
    # of particle j
    # The inner loop can not be parallelized, since each iteration overwrites a[i]

    # for all particles
    for i in prange(m.shape[0]):        # prange = numba parallel range
        a[i] = np.zeros(3)

        # for all other particles
        for j in range(m.shape[0]):
            # no self interaction
            if i == j:
                continue

            dist_vec = p[i] - p[j]
            dist = np.sqrt(np.sum(dist_vec * dist_vec))
            a[i] += -g * m[j] * dist_vec / (dist ** 3)

    return a

@nb.jit(nopython=True, parallel=True)
def find_a_max(a):
    return np.max(np.abs(a))