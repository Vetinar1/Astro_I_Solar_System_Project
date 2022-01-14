import numpy as np
import numba as nb
import sys
import time

# Technically, "from X import *" is frowned upon because it can easily clutter your namespace, or lead to
# unforeseen interactions. Doesn't matter here
from solar_system import *
from integrator import *

# Here's a hack to make sure we can print whole numpy arrays to single lines:
np.set_printoptions(
    threshold=sys.maxsize,
    linewidth=1000, # increase as needed
    sign=" "
)

M_UNIT = M_SUN
L_UNIT = AU
T_UNIT = YR
V_UNIT = L_UNIT / T_UNIT
G = 6.674e-8 * M_UNIT * (T_UNIT ** 2) / (L_UNIT ** 3)

T_MAX = 100
MIN_DT = 0.1
DT_OUTPUT = 0.1  # output interval
OUTFILE = "solar_system.txt"

# Python scripts are evaluated top to bottom, no matter whats in them. Functions do not exist until the interpreter
# reaches the function definition.
# Numba considers all outer-scope variables defined at the point of the function definition as compile time constants.
@nb.jit(nopython=True)
def run(m, p, v, compile_only=False):
    # Here's a quick hack: The function is compiled by numpy the first time it is encountered, but that adds to the
    # execution time. On the other hand we also don't want to wastefully run the function during the compilation step
    # because that adds to the "compilation" time...
    # So I added this flag to abort the function execution immediately.
    if compile_only:
        return

    t = 0
    next_output_time = 0

    # initial accelerations
    a = np.zeros(p.shape)
    grav_bruteforce(m, p, a, G)

    # At these scales, output is always going to be a bottleneck
    # Unfortunately, output options in nopython mode are very restricted. We can't use format strings or
    # np.savetxt tricks. So just construct a really big array instead...
    out = np.zeros(
        (
            int(T_MAX / DT_OUTPUT), # the number of entries we need
            1 + 6 * m.shape[0]      # 3 positions and 3 velocities for every body, as well as one timestamp
        )
    )
    c = 0 # counter

    while t < T_MAX:
        if t >= next_output_time:
            out[c, 0] = t
            out[c, 1:3*m.shape[0]+1] = p.flatten()
            out[c, 3*m.shape[0]+1:]  = v.flatten()

            next_output_time += DT_OUTPUT
            c += 1

        # find a_max and adaptive timestep
        a_max = find_a_max(a)
        dt = MIN_DT / a_max

        # update system
        # kick
        v += a * 0.5 * dt
        # drift
        p += v * dt
        a = grav_bruteforce(m, p, a, G)
        # kick
        v += a * 0.5 * dt

        t += dt

    return out


# This code is only executed if the script is run as... a script. It has no effect if it is instead loaded as a module
# I do not intend to use this as a module, but it still a nice way to keep functions and instructions apart
if __name__ == "__main__":

    mass = get_solar_system_masses(M_UNIT)
    pos  = get_solar_system_positions(L_UNIT)
    vel  = get_solar_system_velocities(V_UNIT)


    print("Compiling...")
    run(mass, pos, vel, compile_only=True)

    print("Running...")
    time1 = time.time()         # not the most accurate or sophisticated way to time things, but good enough
    out = run(mass, pos, vel)
    time2 = time.time()

    # Format strings are extremely useful, even if you cant remember all the bazillion options:
    # https://docs.python.org/3/library/string.html#formatstrings
    print(f"Took {time2 - time1:.2f}s")

    # should be around 20 times faster than the other version
    # A bit less if you consider that this version does not include a perturber
    # Honestly, not a *huge* difference!

    np.savetxt("out.csv", out)