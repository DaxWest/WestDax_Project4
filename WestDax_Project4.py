import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

#setting constants
h_bar = 1
m = 1/2
t_init = 0

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    '''

    :param nspace: number of spatial grid points
    :param ntime: number of time steps to be evolved
    :param tau: time step
    :param method: Default 'ftcs', takes strings: 'ftcs' or 'crank'
    :param length: size of spatial grid. Default is 200 (-100 to +100)
    :param potential: a 1D array of the spatial index values for which the potential must be V(x)=1
    :param wparam: list of parameters for initial conditions in the form [sigma0, x0, k0]
    :return: a 2D array containing phi_x, phi_t and, prob which are all 1D arrays
    '''
    sigma0, x0, k0 = wparam[0], wparam[1], wparam[2]

    #intitializing these for now, they will contain more information later
    phi_x = 1
    phi_t = 1
    prob = 1

    eqn_sol = np.array([[phi_x], [phi_t], [prob]])
    return eqn_sol

def sch_plot(x,t, P, output='psi', save=False):
    '''

    :param x: phi_x output from sch_eqn (a 1D array)
    :param t: phi_t output from sch_eqn (a 1D array)
    :param P: prob output from sch_eqn (a 1D array)
    :param output: takes a str ('psi' or 'prob') which designates the output of the function
    :param save: allows the user to choose to save the figure when set as True. Default False.
    :return: psi (plot of the real part of the schrodinger function) or prob (plot of the particle probability density)
    '''

    x_pos = x
    t_val = t
    prob_density = P

    if output == 'psi':
        #this is the real portion
        if save == True:
            figure = plt.savefig()

    elif output == 'prob':
        #probablity density
        if save == True:
            figure = plt.savefig()

    return figure

