# This code is for the 3-state cell death model proposed in the JoB paper
# The model is based on tissue hypoxia caused by micro-occlusions, which has been simulated by a Green's function method in physiologically representative cerebral capillary cubes

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

kf,kb,ke = 6e-5, 4e-5, -0.1754/86400

# kf - forward rate constant
# kb - backward rate constant
# ke - extravasation rate constant fitted from van der Wijk et al., 2019, Stroke (unit of time: second)
# Note that kf and kb have not been fitted or validated

# Cell death model, h[0] - alive, h[1] - dead, h[2] - blockage fraction, 1-h[0]-h[1] - vulnerable
# The cells were assumed to have 3 states: alive, vulnerable and dead, where A+V+D=1
# The vulnerable compartment represents the cells that may have experienced energy impairment but still have potential for recovery
# So the pathway between alive and vulnerable is reversible, but the pathway between vulnerable and dead is irreversible

def cell_death(h, t):
	v = 1-h[0]-h[1]
    # sigmoidal relationship between tissue hypoxic fraction and vessel blockage fraction: simulated by a Green's function method
	hypo = 1/(1+np.exp(-(30.71*h[2]-5.35)))
	return [-kf*hypo*h[0]+kb*(1-hypo)*v, kf*hypo*v, ke*h[2]]

# An example solution - variables and parameters can be reset
b_init = 0.15 # initial blockage fraction - the fraction of vessels occluded by microthrombi in the network
ts = np.linspace(0, 604800, 10080)   # timestep
hi = [1, 0, b_init]   # initial values
hs = odeint(cell_death, hi, ts)   # solve the odes

A = hs[:,0]
V = [1]*len(hs[:,0]) - hs[:,0] - hs[:,1]
D = hs[:,1]
B = hs[:,2]
H = 1/(1+np.exp(-(30.19*B-5.11)))

# plot the example solution

fig,ax=plt.subplots()
day = ts/86400   # change unit to days for plotting

ax.plot(day, A, color='blue', label='Alive')
ax.plot(day, V, color='green', label='Vulnerable')
ax.plot(day, D, color='red', label='Dead')
# ax.plot(day, B, color='purple', label='Blockage')
ax.plot(day, H, color='orange', label='Hypoxia')

ax.set_xlabel("Time (day)")
ax.set_ylabel("Cell fraction")
ax.set_xlim(0,7)
ax.set_ylim(0,1)
ax.set_title('$\mathregular{B_0=}$'+str(b_init*100)+'%', y=-0.25)
ax.legend()

plt.show()