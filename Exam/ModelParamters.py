#ModelParamters

def ModelBY():
    param = dict()
    # Preferences
    param['delta']  = .998
    param['gamma'] = 10
    param['psi'] = 1.5
    # Consumption
    param['mu_c'] = .0015
    param['rho'] = .979
    param['phi_x'] = .044
    param['sigma_bar'] = 0.0078
    param['nu'] = .987
    param['phi_s'] = .0000023
    # Dividends
    param['mu_d'] = .0015
    param['phi'] = 3
    param['phi_dc'] = 0
    param['phi_d'] = 4.5
    param['theta'] = (1 - param['gamma'])/( 1 - 1/param['psi'])

    return param

def ModelBKY():
    param = dict()
    # Preferences
    param['delta']  = .9989
    param['gamma'] = 10
    param['psi'] = 1.5
    # Consumption
    param['mu_c'] = .0015
    param['rho'] = .975
    param['phi_x'] = .038
    param['sigma_bar'] = 0.0072
    param['nu'] = .999
    param['phi_s'] = .0000028
    # Dividends
    param['mu_d'] = .0015
    param['phi'] = 2.5
    param['phi_dc'] = 2.6
    param['phi_d'] = 5.96
    param['theta'] = (1 - param['gamma'])/( 1 - 1/param['psi'])

    return param


