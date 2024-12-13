import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

#setting constants
h_bar = 1
m = 1/2
t_init = 0

#from lab 10
def make_tridiagonal(N, b, d, a):
    '''
    :param N:
    :param b:
    :param d:
    :param a:
    :return:
    '''

    matrix = d*np.identity(N)+a*np.diagflat(np.ones(N-1),1)+b*np.diagflat(np.ones(N-1),-1)

    return matrix

#from lab 10
def spectral_radius(matrix):
    '''
    :param matrix:takes a matrix and returns its maximum eigen values
    :return:
    '''

    eigen = np.linalg.eig(matrix)

    return np.max(np.abs(eigen[0]))


def make_initialcond(wparam, x_position):
    '''
    :param wparam: 1D array of parameters for initial conditions in the form [sigma0, x0, k0]
    :param x_position:
    :return:
    '''
    sigma0, x0, k0 = wparam[0], wparam[1], wparam[2]
    x_i = x_position
    #Gaussian wave packet as time t=0
    psi_i = (1/np.sqrt(sigma0 * np.sqrt(np.pi))) * (np.exp(1j * k0 * x_i)) * (np.exp(-(x_i-x0)**2)/(2 * sigma0**2))

    return psi_i

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    '''
    :param nspace: number of spatial grid points
    :param ntime: number of time steps to be evolved
    :param tau: time step
    :param method: Default 'ftcs', takes strings: 'ftcs' or 'crank'
    :param length: size of spatial grid. Default is 200 (-100 to +100)
    :param potential: a 1D array of the spatial index values for which the potential must be V(x)=1
    :param wparam: 1D array of parameters for initial conditions in the form [sigma0, x0, k0]
    :return: a 2D array containing psi, psi_x, psi_t and, prob which are all 1D arrays
    '''
    h_bar = 1
    m = 1 / 2
    t_init = 0

    sigma0, x0, k0 = wparam[0], wparam[1], wparam[2]
    x_position = np.linspace(-length/2, length/2, nspace, endpoint=False)
    t_step = np.arange(t_init, ntime*tau, tau)
    step = length / (nspace-1)
    H_const = -(h_bar**2) / (2 * m * (step**2)) #for use in Hamiltonian
    identity = np.identity(nspace)

    psi_shape = (ntime, nspace)
    psi_intital = make_initialcond(wparam, x_position)
    psi = np.zeros(shape=psi_shape, dtype=complex)
    psi[0, :] = psi_intital

    if method == 'ftcs':
        H = (1j * tau / h_bar) * make_tridiagonal(nspace, H_const, (1 - (2 * H_const)), H_const)
        coeff_ftcs = identity - H

        #this method needs a stability check
        if spectral_radius(coeff_ftcs) > 1:
            return 'Solution will not be stable.'
        else:
            for i in range(ntime-1):
                # FTCS method, eqn: 9.32
                psi[i+1, :] = np.dot(coeff_ftcs, psi[i, :])

    elif method == 'crank':
        H = (1j * tau / (2*h_bar)) * make_tridiagonal(nspace, H_const, (-2 * H_const), H_const)

        coeff_crank = np.dot()

        for i in range(ntime):
            # Crank method, eqn: 9.40
            psi[i, :] = np.dot(coeff_crank, psi[i-1, :])
        #psi_n_1 = (identity + (1j * tau / (2*h_bar)) * H)**(-1) * (identity - (1j * tau / (2*h_bar)) * H) * phi_n

    else:
        return 'No valid integration method was selected.'

    #intitializing these for now, they will contain more information later
    psi_x = x_position
    psi_t = t_step
    prob = np.abs(psi * np.conjugate(psi))

    eqn_sol = [psi, psi_x, psi_t, prob]
    return eqn_sol

def sch_plot(sch_sol, output=['psi', 'prob'], save=[True, True], file_name=['psi_plot', 'prob_plot']):
    ''''''

    psi, x_pos, t_val, prob = sch_sol[0], sch_sol[1], sch_sol[2], sch_sol[3]

    if output[0] == 'psi':
        #this is the real portion
        figure1 = plt.figure()
        if save[0] == True:
            plt.savefig(f'{file_name[0]}.png')

    elif output[1] == 'prob':
        #probablity density
        if save[1] == True:
            plt.savefig(f'{file_name[1]}.png')

    else:
        return 'No figures produced.'

    return 'Figure(s) saved under given name as png.'

#user inputs
nspace = int(input('Choose the number of spatial grid points to be used (function works for nspace = 500): '))
ntime = int(input('Choose the number of time steps to be evolved (function works for ntime = 500): '))
tau = float(input('Choose time step to be used (function works for tau = 0.1): '))
length = int(input('Choose width of solution (function works for length = 200): '))

method_choice = str(input('Choose a solution method- 1. FTCS, 2. Crank-Nicholson: '))
if method_choice == 1 or 'ftcs' in method_choice.lower():
    method = 'ftcs'
elif method_choice == 2 or 'crank' in method_choice.lower():
    method = 'crank'
else:
    method = method_choice

output = ['psi', 'prob']
for i in range(2):
    choices = ['psi', 'prob']
    output_choice = str(input(f'Do you wish to make a plot for {choices[i]} (Y/N)? '))
    if output_choice.lower() == 'y':
        output[i] = choices[i]
    else:
        output[i] = 'none'

save = [False, False]
if output[0] == 'psi' or output[1] == 'prob':
    for i in range(2):
        choices = ['psi', 'prob']
        save_choice = str(input(f'Do you wish to save the plot for {choices[i]} (Y/N)?'))
        if save_choice.lower() == 'y':
            save[i] = True
        else:
            save[i] = False

file_name=['psi_plot', 'prob_plot']
for i in range(2):
    if save[i] == True:
        file_choice = str(input(f'Enter desired file name: '))
        file_name[i] = file_choice

