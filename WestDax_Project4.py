import numpy as np
import matplotlib.pyplot as plt
#github repository link:

#from lab 10 originally, had to modify because it was not working properly
def make_tridiagonal(N, b, d, a):
    '''
    Forms a tri-diagonal matrix of size N x N given three values b, d, a which make up the below diagonal, diagonal and above diagonal values respectively.\

    :param N: integer, sets the dimensions of the returned matrix
    :param b: integer, value in position one below the diagonal
    :param d: integer, value in position of the diagonal
    :param a: integer, value in position one above the diagonal
    :return:
    '''
    #initializing matrix
    matrix = np.zeros((N, N))

    #fills tridigonal positions of matrix
    b_matrix = b * np.eye(N, k=-1)
    d_matrix = d * np.eye(N, k=0)
    a_matrix = a * np.eye(N, k=1)

    #combines all four matrices to make tridiagonal matrix
    matrix = matrix + b_matrix + d_matrix + a_matrix

    #adds the two values that are not covered by the previous code (in the corners)
    matrix[0, -1] = b
    matrix[-1, 0] = a

    return matrix

#from lab 10
def spectral_radius(matrix):
    '''
    Returns the maximum value of the magnitude of the eigenvalues for a given matrix.

    :param matrix:takes a matrix and returns its maximum eigen values
    :return: maximum magnitude of eigen values
    '''

    #finds the eigen values, their magnitude and the maximum value
    eigen = np.linalg.eig(matrix)
    max_eigen = np.max(np.abs(eigen[0]))
    return max_eigen


def make_initialcond(wparam, x_position):
    '''
    A function which returns the initial conditions of a system in the form of a Gaussian wave packet at time t=0 given some initial values. Return is complex.

    :param wparam: 1D array of parameters for initial conditions in the form [sigma0, x0, k0]
    :param x_position: ndarray of the space of the system
    :return: the initial value of the wavepacket
    '''

    #intial conditions
    sigma0, x0, k0 = wparam[0], wparam[1], wparam[2]
    x_i = x_position

    #Gaussian wave packet as time t=0
    psi_i = (1/np.sqrt(sigma0 * np.sqrt(np.pi))) * (np.exp(1j * k0 * x_i)) * (np.exp(-(x_i-x0)**2)/(2 * sigma0**2))

    return psi_i

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10, 0, 0.5]):
    '''
    Solves the one-dimensional, time-dependent Schrodinger equation using the explicit Forward-Time, Central-Space (FTCS) or the Crank-Nicolson scheme.

    :param nspace: number of spatial grid points
    :param ntime: number of time steps to be evolved
    :param tau: time step
    :param method: Default 'ftcs', takes strings: 'ftcs' or 'crank'
    :param length: size of spatial grid. Default is 200 (-100 to +100)
    :param potential: 1D array of the spatial index values for which the potential must be V(x)=1
    :param wparam: list of parameters for initial conditions in the form [sigma0, x0, k0]
    :return: a 2D array containing psi, psi_x, psi_t and, prob which are all 1D arrays
    '''

    #setting constants
    h_bar = 1
    m = 1 / 2

    #takes function arguments to make needed variables
    x_position = np.linspace(-length / 2, length / 2, nspace, endpoint=False)
    t_step = np.arange(0, ntime * tau, tau)
    step = length / (nspace - 1)
    H_const = -(h_bar ** 2) / (2 * m * (step ** 2))  # for use in Hamiltonian
    identity = np.identity(nspace)

    #initializes the container for psi values
    psi_shape = (ntime, nspace)
    psi_intital = make_initialcond(wparam, x_position)
    psi = np.zeros(psi_shape, dtype=complex)
    psi[0, :] = psi_intital

    #ftcs solution method
    if method == 'ftcs':
        H = (1j * tau / h_bar) * make_tridiagonal(nspace, H_const, (1 - (2 * H_const)), H_const)
        coeff_ftcs = identity - H
        check = spectral_radius(H)
        # this method needs a stability check
        if check-1 > 1e-10:
            raise ValueError('Solution will not be stable.')
        else:
            for i in range(ntime - 1):
                # FTCS method, eqn: 9.32 ftcs
                psi[i + 1, :] = np.dot(coeff_ftcs, psi[i, :])

    #crank solution method
    elif method == 'crank':
        H = (1j * tau / (2 * h_bar)) * make_tridiagonal(nspace, H_const, (-2 * H_const), H_const)
        coeff_crank = np.dot(np.linalg.inv(identity + H), identity - H)

        for i in range(1, ntime):
            # Crank method, eqn: 9.40
            psi[i, :] = np.dot(coeff_crank, psi[i - 1, :])

    #catch for improper method selection
    else:
        raise ValueError('No valid integration method was selected.')

    #fills varibles for function output
    psi_x = x_position
    psi_t = t_step
    prob = np.abs(psi * np.conjugate(psi))
    eqn_sol = [psi, psi_x, psi_t, prob]

    return eqn_sol

def sch_plot(sch_sol, output=['psi', 'prob'], save=[True, True], file_name=['psi_plot', 'prob_plot']):
    '''
    Plots the results of the solution to the one-dimensional, time-dependent Schrodinger equation and/or the total probability depending on the given optional values. Offers options to automatically save plots.

    :param sch_sol: the sch_equ function results to be plotted
    :param output: a list, decides which of the two plots (both, one or, neither) will be produced. Default is to produce both
    :param save: a list, decides which of the two plots (both, one or, neither) will be saved. Default is to save both
    :param file_name: a list, sets the file name under which the plots will be saved. Default is 'psi_plot', 'prob_plot'
    :return: produces plots (if any) and saves those plots  as ‘.png’ (if desired).
    '''

    #setting variables from function arguments
    psi, x_pos, t_val, prob = sch_sol[0], sch_sol[1], sch_sol[2], sch_sol[3]
    #for use with enumerate
    enum = len(sch_sol[0]) / 5

    #generates psi position plot
    if output[0] == 'psi':
        # this is the real portion
        fig1 = plt.figure(figsize=(8,6))
        for i, T in enumerate(t_val):
            if i % enum == 0:
                plt.plot(x_pos, np.real(psi[i]))
        plt.title('Schrodinger Wave Equation Results')
        plt.xlabel('Position (x)')
        plt.ylabel(r'$\psi$ (x, t)')
        plt.show()

        #saves plot if so desired
        if save[0] == True:
            plt.savefig(f'{file_name[0]}.png')

    #generates particle probability plot
    if output[1] == 'prob':
        # probablity density
        fig2 = plt.figure(figsize=(8,6))
        for i, T in enumerate(t_val):
            if i % enum == 0:
                plt.plot(x_pos, prob[i])
        plt.title('Particle Probability Density')
        plt.xlabel('Position (x)')
        plt.ylabel(r'$|\psi|^{2}$ (x, t)')
        plt.show()

        #saves plot if so desired
        if save[1] == True:
            plt.savefig(f'{file_name[1]}.png')

    #informs useer that no plots were selected to be produced
    else:
        return print('No figures produced.')

    #informs user that the chosen plots were saved
    if save[0] or save[1] == True:
        return print('Figure(s) saved under given name as png.')

    #informs user that plots are complete. only returned if no figures were produced
    else:
        return print('Plotting complete, process finished.')

#user inputs for testing
#these are not needed to run the code but will be left as proof of testing and so that testing conditions can be replicated precisely if need be

# nspace = int(input('Choose the number of spatial grid points to be used (function works for nspace = 30): '))
# ntime = int(input('Choose the number of time steps to be evolved (function works for ntime = 500): '))
# tau = float(input('Choose time step to be used (function works conditionally for |tau| < 1 when method =ftcs, unconditionally when method =crank): '))
# length = int(input('Choose width of solution (function works for length = 200): '))
#
# method_choice = str(input('Choose a solution method- FTCS, Crank-Nicholson: '))
# if 'ftcs' in method_choice.lower():
#     method = 'ftcs'
# elif 'crank' in method_choice.lower():
#     method = 'crank'
# else:
#     method = method_choice
#
# output = ['psi', 'prob']
# for i in range(2):
#     choices = ['psi', 'prob']
#     output_choice = str(input(f'Do you wish to make a plot for {choices[i]} (Y/N)? '))
#     if output_choice.lower() == 'y':
#         output[i] = choices[i]
#     else:
#         output[i] = 'none'
#
# save = [False, False]
# if output[0] == 'psi' or output[1] == 'prob':
#     for i in range(2):
#         choices = ['psi', 'prob']
#         save_choice = str(input(f'Do you wish to save the plot for {choices[i]} (Y/N)? '))
#         if save_choice.lower() == 'y':
#             save[i] = True
#         else:
#             save[i] = False
#
# file_name=['psi_plot', 'prob_plot']
# for i in range(2):
#     if save[i] == True:
#         file_choice = str(input(f'Enter desired file name: '))
#         file_name[i] = file_choice
#
# schrodinger_solution = sch_eqn(nspace, ntime, tau, method, length)
# plotting = sch_plot(schrodinger_solution, output, save, file_name)