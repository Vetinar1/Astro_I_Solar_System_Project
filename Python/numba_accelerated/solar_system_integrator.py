import numpy as np
import numba as nb
import sys

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
G = 6.674e-8 * M_UNIT * T_UNIT / (L_UNIT ** 3)

T_MAX = 100
MIN_DT = 0.1
DT_OUTPUT = 0.1  # output interval
OUTFILE = "solar_system.txt"

# Python scripts are evaluated top to bottom, no matter whats in them. Functions do not exist until the interpreter
# reaches the function definition.
# Numba considers all outer-scope variables defined at the point of the function definition as compile time constants.
@nb.jit(nopython=True)
def run(m, p, v):
    t = 0
    next_output_time = 0

    # initial accelerations
    a = np.zeros(p.shape)
    grav_bruteforce(m, p, a, G)

    while t < T_MAX:
        if t >= next_output_time:
            # At these scales, output is always going to be a bottleneck
            # Unfortunately, output options in nopython mode are very restricted. We can't use format strings or
            # np.savetxt tricks. So just construct a really long string instead...
            out_str = str(t) + " "
            for i in range(pos.shape[0]):
                out_str += str(pos[i][0] + " " + pos[i][1] + " " + pos[i][2] + " ")
            print(out_str)
            next_output_time += DT_OUTPUT
            pass

        # find a_max and adaptive timestep
        a_max = find_a_max(a)
        dt = MIN_DT / a_max

        # update system
        kick(p, v, 0.5*dt)
        drift(p, v, dt)
        grav_bruteforce(m, p, a, G)
        kick(p, v, 0.5*dt)

        t += dt

# This code is only executed if the script is run as... a script. It has no effect if it is instead loaded as a module
# I do not intend to use this as a module, but it still a nice way to keep functions and instructions apart
if __name__ == "__main__":


    mass = get_solar_system_masses(M_UNIT)
    pos  = get_solar_system_positions(L_UNIT)
    vel  = get_solar_system_velocities(V_UNIT)

    run(mass, pos, vel)