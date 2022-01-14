import numpy as np

"""
Some constants, and functions to get solar system "constants" in different units.
"""

# I just dump all of these in the global namespace. Not necessarily pretty, but since this script isnt going to be
# part of something larger its perfectly fine
# We could collect them in struct and pass them around, however
# - These are constants that we need almost everywhere, but never change
# - Numba doesn't handle classes too well
#
# Note that all the constants are written in caps. Python doesn't support "true" constants;
# by convention we write them in allcaps and just never change them

M_SUN = 1.989e33        # Sun mass in g
M_MOON = 7.342e25       # Moon mass in g
M_EARTH = 5.97219e27    # Earth mass in g
AU = 1.495978707e13     # AU in cm
R_EARTH = 6371e5        # Earth diameter in cm
D_MOON = 384400e5       # distance moon in cm
YR = 3.1556926e7        # year in s
PC = 3.085678e18        # parsec in cm
KPC = 3.085678e21       # kiloparsec in cm
MPC = 3.085678e24       # megaparsec in cm


# solar system: a numpy array each for mass (N), position (N, 3), velocity (N, 3), and acceleration (N, 3)
# if we were thorough we would, of course, define the masses as constants. but who cares
def get_solar_system_masses(m_unit):
    sol_mass = np.array([
        1.988544e33/m_unit,     # sun
        3.302e26/m_unit,        # mercury
        4.8685e27/m_unit,       # venus
        15.97219e27/m_unit,     # earth
        6.4185e26/m_unit,       # mars
        1.89813e30/m_unit,      # jupiter
        5.68319e29/m_unit,      # saturn
        8.68103e28/m_unit,      # uranus lol
        1.0241e27/m_unit,       # neptune
        1.307e25/m_unit,        # pluto (not actually a planet)
    ])
    return sol_mass

def get_solar_system_positions(l_unit):
    sol_pos = np.array([
        [   # sun
            4.685928291891263e+05/(l_unit/1.e5),
            9.563194923290641e+05/(l_unit/1.e5),
            -1.533341587127076e+04/(l_unit/1.e5)
        ],
        [   # mercury
            -4.713579828527527e+07/(l_unit/1.e5),
            -4.631957178347297e+07/(l_unit/1.e5),
            5.106488259447999e+05/(l_unit/1.e5)
        ],
        [   # venus
            1.087015383199374e+08/(l_unit/1.e5),
            -7.281577953082427e+06/(l_unit/1.e5),
            -6.381857167679189e+06/(l_unit/1.e5)
        ],
        [   # earth
            -4.666572753335893e+07/(l_unit/1.e5),
            1.403043145802726e+08/(l_unit/1.e5),
            1.493509552154690e+04/(l_unit/1.e5)
        ],
        [   # mars
            7.993300729834399e+07/(l_unit/1.e5),
            -1.951269688004358e+08/(l_unit/1.e5),
            -6.086301544224218e+06/(l_unit/1.e5)
        ],
        [   # jupiter
            -4.442444431519640e+08/(l_unit/1.e5),
            -6.703061523285834e+08/(l_unit/1.e5),
            1.269185734527490e+07/(l_unit/1.e5)
        ],
        [   # saturn
            -4.890566777017240e+07/(l_unit/1.e5),
            -1.503979857988314e+09/(l_unit/1.e5),
            2.843053033246052e+07/(l_unit/1.e5)
        ],
        [   # uranus
            -9.649665981767261e+08/(l_unit/1.e5),
            -2.671478218630915e+09/(l_unit/1.e5),
            2.586047227024674e+06/(l_unit/1.e5)
        ],
        [   # neptune
            2.238011384258528e+08/(l_unit/1.e5),
            4.462979506400823e+09/(l_unit/1.e5),
            -9.704945189848828e+07/(l_unit/1.e5)
        ],
        [   # pluto (still not a  planet)
            1.538634961725572e+09/(l_unit/1.e5),
            6.754880920368265e+09/(l_unit/1.e5),
            -1.168322135333601e+09/(l_unit/1.e5)
        ]
    ])
    return sol_pos

def get_solar_system_velocities(v_unit):
    sol_vel = np.array([
        [   # sun
            -1.278455768585727e-02/(v_unit/1.e5),
            6.447692564652730e-03/(v_unit/1.e5),
            3.039394044840682e-04/(v_unit/1.e5)
        ],
        [   # mercury
            2.440414864241152e+01/(v_unit/1.e5),
            -3.230927714856684e+01/(v_unit/1.e5),
            -4.882735649260043e+00/(v_unit/1.e5)
        ],
        [   # venus
            2.484508425171419e+00/(v_unit/1.e5),
            3.476687455583895e+01/(v_unit/1.e5),
            3.213482419270903e-01/(v_unit/1.e5)
        ],
        [   # earth
            -2.871599709379560e+01/(v_unit/1.e5),
            -9.658668417740959e+00/(v_unit/1.e5),
            -2.049066619477902e-03/(v_unit/1.e5)
        ],
        [   # mars
            2.337340830878404e+01/(v_unit/1.e5),
            1.117498287104724e+01/(v_unit/1.e5),
            -3.459891064580085e-01/(v_unit/1.e5)
        ],
        [   # jupiter
            1.073596630262369e+01/(v_unit/1.e5),
            -6.599122996686262e+00/(v_unit/1.e5),
            -2.139417332332738e-01/(v_unit/1.e5)
        ],
        [   # saturn
            9.121308225757311e+00/(v_unit/1.e5),
            -3.524504589006163e-01/(v_unit/1.e5),
            -3.554364061038437e-01/(v_unit/1.e5)
        ],
        [   # uranus
            6.352626804478141e+00/(v_unit/1.e5),
            -2.630553214528946e+00/(v_unit/1.e5),
            -9.234330561966453e-02/(v_unit/1.e5)
        ],
        [   # neptune
            -5.460590042066011e+00/(v_unit/1.e5),
            3.078261976854122e-01/(v_unit/1.e5),
            1.198212503409012e-01/(v_unit/1.e5)
        ],
        [   # pluto
            -3.748709222608039e+00/(v_unit/1.e5),
            3.840130094300949e-01/(v_unit/1.e5),
            1.063222714737127e+00/(v_unit/1.e5)
        ]
    ])
    return sol_vel

