#########################################################
###                                                   ###
###           Compute the Log Linear Solution         ###
###                                                   ###                     
#########################################################
"""Overview of the program:
    solve
"""
import numpy as np
from scipy import optimize

def obj_WC_LL(z, *argslistWC):
    x = z
  
    argslistWC = delta, gamma, psi, mu_c, rho, phi_x, sigma_bar, nu, phi_s, theta = argslistWC

    k1 = ( np.exp(x) / (np.exp(x) - 1) )
    k0 = ( np.log( (k1 - 1)**(1 - k1) * k1**k1) )

    A1 =  (1 - 1/psi)/(k1 - rho)
    A2 =  0.5* theta*( (1 - 1/psi)**2 + (A1 * phi_x)**2) / (k1 - nu)
    A0 =  (1/(k1 - 1)) * ( np.log(delta) + (1 - 1/psi)*mu_c + k0 + A2 * sigma_bar**2 * (1 - nu) + theta / 2*(A2 * phi_s)**(2 )  )
        
    diff = x - A0 - A2* sigma_bar**2

    return diff

def WC_LL(start_search, end_search ,param, max_iterations):

    delta = param['delta']
    gamma = param['gamma']
    psi = param['psi']

    # Consumption
    mu_c = param['mu_c']
    rho = param['rho']
    phi_x = param['phi_x'] 
    sigma_bar = param['sigma_bar']
    nu = param['nu'] 
    phi_s = param['phi_s']

    theta = param['theta']

    argslistWC = delta, gamma, psi, mu_c, rho, phi_x, sigma_bar, nu, phi_s, theta

    # find the mean price-dividend ratio
    z_opt = optimize.bisect(obj_WC_LL, start_search , end_search, maxiter = max_iterations ,args= argslistWC)

    # find parameters required to solve the model
    k1 = ( np.exp(z_opt) / (np.exp(z_opt) - 1) )
    k0 = ( np.log( (k1 - 1)**(1 - k1) * k1**k1) )

    A1 =  (1 - 1/psi)/(k1 - rho)
    A2 =  0.5* theta*( (1 - 1/psi)**2 + (A1 * phi_x)**2) / (k1 - nu)
    A0 =  (1/(k1 - 1)) * ( np.log(delta) + (1 - 1/psi)*mu_c + k0 + A2 * sigma_bar**2 * (1 - nu) + theta / 2*(A2 * phi_s)**(2 )  )

    return k1, k0, A1, A2, A0

def obj_pd_LL(z, *argslistPD):
    x = z

    delta, gamma, psi, mu_c, rho, phi_x, sigma_bar, nu, phi_s, mu_d, phi, phi_dc, phi_d, theta, k0,  k1, A0, A1, A2 = argslistPD 

    k1_m = (np.exp(x) / (np.exp(x) + 1))
    k0_m = (-np.log( (1 - k1_m)**(1 - k1_m) * k1_m**(k1_m) ) )

    A1_m = (phi - 1/psi) / (1 - rho * k1_m)
    A2_m = (0.5 * (phi_dc - gamma)**(2) + 0.5 * phi_d**(2) + (theta - 1) * A2 * (nu - k1)  + 0.5*( (theta - 1) *A1 + k1_m * A1_m )**(2) *phi_x**(2) )  / (1 - k1_m *nu)
    A0_m = (theta *np.log(delta) - gamma *mu_c + mu_d + (theta - 1) * ( k0 + A0 * (1 - k1) ) + k0_m + ( (theta - 1) * A2 + k1_m * A2_m) * sigma_bar**(2) * (1 - nu)  \
            + 0.5*( (theta - 1) * A2 + k1_m *A2_m)**(2) *phi_s**(2) ) / (1 - k1_m)

    diff = x - A0_m - A2_m * sigma_bar**(2)
        
    return diff

def pd_LL(start_search, end_search ,param, paramResults ,max_iterations):

    delta = param['delta']
    gamma = param['gamma']
    psi = param['psi']

    # Consumption
    mu_c = param['mu_c']
    rho = param['rho']
    phi_x = param['phi_x'] 
    sigma_bar = param['sigma_bar']
    nu = param['nu'] 
    phi_s = param['phi_s']

    # Dividends
    mu_d = param['mu_d']
    phi = param['phi'] 
    phi_dc = param['phi_dc']
    phi_d = param['phi_d']

    k0 = paramResults['k0'] 
    k1 = paramResults['k1']
    A0 = paramResults['A0']
    A1 = paramResults['A1']
    A2 = paramResults['A2']

    theta = (1 - gamma)/( 1 - 1/psi)

    argslistPD = delta, gamma, psi, mu_c, rho, phi_x, sigma_bar, nu, phi_s, mu_d, phi, phi_dc, phi_d, theta, k0,  k1, A0, A1, A2

    # find the mean price-dividend ratio
    z_opt = optimize.bisect(obj_pd_LL, start_search , end_search, maxiter = max_iterations, args= argslistPD)

    # find parameters required to solve the model
    k1_m = (np.exp(z_opt) / (np.exp(z_opt) + 1))
    k0_m = (-np.log( (1 - k1_m)**(1 - k1_m) * k1_m**(k1_m) ) )

    A1_m = (phi - 1/psi) / (1 - rho * k1_m)
    A2_m = (0.5 * (phi_dc - gamma)**(2) + 0.5 * phi_d**(2) + (theta - 1) * A2 * (nu - k1)  + 0.5*( (theta - 1) *A1 + k1_m * A1_m )**(2) *phi_x**(2) )  / (1 - k1_m *nu)
    A0_m = (theta *np.log(delta) - gamma *mu_c + mu_d + (theta - 1) * ( k0 + A0 * (1 - k1) ) + k0_m + ( (theta - 1) * A2 + k1_m * A2_m) * sigma_bar**(2) * (1 - nu)\
            + 0.5*( (theta - 1) * A2 + k1_m *A2_m)**(2) *phi_s**(2) ) / (1 - k1_m)

    return k0_m, k1_m, A0_m, A1_m, A2_m

def rf_coeff(param, paramResults):

    delta = param['delta']
    gamma = param['gamma']
    psi = param['psi']

    # Consumption
    mu_c = param['mu_c']
    rho = param['rho']
    phi_x = param['phi_x'] 
    sigma_bar = param['sigma_bar']
    nu = param['nu'] 
    phi_s = param['phi_s']

    theta = param['theta']

    k0 = paramResults['k0'] 
    k1 = paramResults['k1']
    A0 = paramResults['A0']
    A1 = paramResults['A1']
    A2 = paramResults['A2']

    theta = (1 - gamma)/( 1 - 1/psi)

    A1_f = theta -1 -(theta/psi) + (theta-1) * A1 *(rho - k1)
    A2_f = 0.5*(theta -1 - (theta/psi) )**2 + 0.5*( (theta -1) *A1 *phi_x)**2 + (theta -1) * A2*(nu -k1)
    A0_f = theta * np.log(delta) + (theta -1 - (theta/psi) ) *mu_c + (theta -1) * (A0 + A2 * sigma_bar**(2) *(1 - nu) + k0 - k1 *A0) + 0.5*( (theta -1) * A2 *phi_s)**(2)

    return A0_f, A1_f, A2_f
