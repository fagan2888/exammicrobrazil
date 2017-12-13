'''
Created on 13 de dez de 2017

@author: luisfantozzialvarez
'''

import numpy as np
import matplotlib.pyplot as mp
from Ulyssea import Ulyssea

tau_pi, tau_omega, r, phi_f, phi_i, s_i, s_f, rho, K_f, K_i, b, eta, a, A = \
 0.35, 0.1, 0.08, 0.45, 0.15, 0.30, 0.13, 0.3, 1.4, 0.3, 0.3, 0.5, 0.62, 0.3
 
 
grid_delta = np.linspace(0, 1, 10000)
 
informality = []
diff = []
welfare = []

for delta in grid_delta:
    economy =  Ulyssea(tau_pi, tau_omega, r, phi_f, phi_i, s_i, s_f, rho, K_f, K_i, b, eta, a, A, delta)
    results = economy.SolveModel(np.array([0.5,0.5,0.5]))
    informality.append(results[0])
    diff.append(results[1])
    welfare.append(results[2])

mp.plot(grid_delta, informality, 'k-')
mp.grid()
mp.title('Equilibrium degree of informality ($1 - \\gamma$) as a function of $\\delta$')
mp.xlabel('$\\delta$')
mp.ylabel('$($1 - \\gamma$)')
mp.savefig('informality.pdf')
mp.show()

mp.plot(grid_delta, 100*np.array(diff), 'k-')
mp.grid()
mp.title('Equilibrium wage differential in % ( $100 \\cdot \\left[\\frac{w_F - w_I}{w_I}\\right]$) as a function of $\\delta$')
mp.xlabel('$\\delta$')
mp.ylabel('$100 \\cdot \\frac{w_F - w_I}{w_I}$')
mp.savefig('wagegap.pdf')
mp.show()



mp.plot(grid_delta, welfare, 'k-')
mp.grid()
mp.title('Equilibrium welfare (W) as a function of $\\delta$')
mp.xlabel('$\\delta$')
mp.ylabel('$W$')
mp.savefig('welfare.pdf')
mp.show()