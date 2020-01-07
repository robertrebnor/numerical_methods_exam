#########################################################
###                                                   ###
###           Compute the Low of Motions              ###
###                                                   ###                     
#########################################################
"""Overview of the program:
    

"""
import numpy as np

def Simul_States(years, NumOFsamples, param):
    """Simulate states for the model of Bansal and Yaron (2004)

    """
    #Unload the paramtervalues
    sigma_bar = param['sigma_bar']
    mu_c = param['mu_c']
    mu_d = param['mu_d']

    #Convert years to months
    T = years*12
    #Number of sample paths 
    NS = NumOFsamples

    x_simul = np.zeros((T+1, NS)) #State x  #(rows, col)
    s_simul = np.zeros((T+1, NS)) #State s
    cg_simul = np.zeros((T+1, NS)) #Consumption growth
    dg_simul = np.zeros((T+1, NS)) #Dividend growth

    #Set initial values to long run mean
    x_simul[0,:] = 0
    s_simul[0,:] = sigma_bar**(2)
    cg_simul[0,:] = mu_c
    dg_simul[0,:] = mu_d

    #Compute the shocks
    np.random.seed(7)

    shock_s = np.random.normal(0, 1, (T,NS) )
    shock_x = np.random.normal(0, 1, (T,NS) )
    shock_c = np.random.normal(0, 1, (T,NS) )
    shock_d = np.random.normal(0, 1, (T,NS) )

    eps = 0
    #Run simulations

    # Consumption
    rho = param['rho']
    phi_x = param['phi_x'] 
    nu = param['nu'] 
    phi_s = param['phi_s']

    # Dividends
    phi = param['phi'] 
    phi_dc = param['phi_dc']
    phi_d = param['phi_d']

    ##fix this for loops
    n = 0
    while n < NS:
        t = 0
        while t < T:    
            x_simul[t+1, n] = rho*x_simul[t,n] + phi_x*s_simul[t,n]**(0.5)*shock_x[t,n]
            s_simul[t+1, n] = max( sigma_bar**(2)*(1 - nu) + nu*s_simul[t,n] + phi_s*shock_s[t,n], eps )
            cg_simul[t+1, n] = mu_c + x_simul[t,n] + s_simul[t,n]**(0.5)*shock_c[t,n]
            dg_simul[t+1, n] = mu_d +  phi*x_simul[t,n] + phi_dc*s_simul[t,n]**(0.5)*shock_c[t,n] + phi_d*s_simul[t,n]**(0.5)*shock_d[t,n]
            t += 1
        n += 1

    return x_simul, s_simul, cg_simul, dg_simul