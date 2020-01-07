#########################################################
###                                                   ###
###                  The Main Program                 ###
###                                                   ###                     
#########################################################
#"""Overview of the program:
#    1. Set up for the model / Set the paramters and values for the program 
#    2. Compute the log linear solution
#    3. Specify Settings for Global Solution
#    4. Compute Projection Solution
#"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pylab
from mpl_toolkits.mplot3d import Axes3D

import ModelParamters
import OrignialMatlabResults
import LogLinSolution
import LowOfMotion
import MyChebyshev
import MySolvers


#########################################################
###          0. Initialize the program                ###                 
#########################################################

##   0.1 Choose which model to estimate
#       ModelChoice = 1   gives Bansal and Yaron (2004)
#       ModelChoice = 2   gives Bansal, Kiku and Yaron (2012)

ModelChoice = 2

##   0.2 Choose if the orginal results from Matlab should be used
#           Since the random generator is different in Matlab and Python, you can here choose to load 
#           the simulated results from Matlab to get the same series as in the paper by PSW2018
#       reproduce_Matlab = 0 uses the series simulated in this program
#       reproduce_Matlab = 1 activates the series simulated in Matlab (default)
#       reproduce_Matlab = 2 activates the series simulated in Matlab and the results from the
#                            projection method in section 4.
#       Notice, in run simulations for annualized moments (section 5) option 1 and 2 is only available 
#               for ModelChoice = 2 amd years2 must be equal 5

reproduce_Matlab = 0

##  0.3 In section 2 the program solves the log-linearization of the model.
#           The Bisection method is used to solve the LL-models. 
#           To use this method values for the search interval [start_search,end_search]
#           and the max number of iterations must be set

start_search  = 2
end_search = 8000
max_iterations = 200

##  0.4 In section 3 the settings for the global solution is calcualted 
#           You have to define the number of years which data will be simulated
#           and the number of samples for each simulation you want.
#           years = 100000 and NumOFsamples = 1 is default from PSW2018

years = 100000  #100000
NumOFsamples = 1

##  0.5 In section 5 we want to run simulations for annualized moments
#           The result is presentet in a tabel, showing the global solution,
#           the log linear solution and the realtive error
#           To run this we need to define the numer of years which will be 
#           simulated and the number of samples paths.
#           The default values from PSW2018 is set up here.
years2 = 100000 #5
NumOFsamples2 = 1

#########################################################
###                                                   ###
###                   MAIN PROGRAM                    ###
###                                                   ###
#########################################################

#########################################################
###          1. Activate choosen set of parameters    ###                 
#########################################################

# Activate the choosen set of parameters 
if ModelChoice == 1:
    param = ModelParamters.ModelBY()
    print("Paramter values for BK2004 is activated")


if ModelChoice == 2:
    param = ModelParamters.ModelBKY()
    print("Paramter values for BKY2012 is activated")

# Preferences
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
theta = param['theta']

#########################################################
###          2. Compute Log Linear Solution           ###                 
#########################################################
#Key values for the root-finding, if you want to run this section separately
#start_search  = 2
#end_search = 8000
#max_iterations = 200

#   LL-model: Compute coefficients for log wealth-consumption ratio
k1, k0, A1, A2, A0 = LogLinSolution.WC_LL(start_search, end_search, param, max_iterations)

#   Print the results
print("k0", k0) 
print("k1", k1) 
print("A0", A0) 
print("A1", A1) 
print("A2", A2) 

#   Store the results in a dict
paramResults = dict()
paramResults['k0'] = k0
paramResults['k1'] = k1
paramResults['A0'] = A0
paramResults['A1'] = A1
paramResults['A2'] = A2

#   LL-model: Compute coefficients for log price-dividend ratio
k0_m, k1_m, A0_m, A1_m, A2_m = LogLinSolution.pd_LL(start_search, end_search, param, paramResults, max_iterations)

#   Print the results
print("k0_m", k0_m) 
print("k1_m", k1_m) 
print("A0_m", A0_m) 
print("A1_m", A1_m) 
print("A2_m", A2_m) 

#   LL-model: Compute coefficients for risk-free rate
A0_f, A1_f, A2_f = LogLinSolution.rf_coeff(param, paramResults)

#   Print the results
print("A0_f", A0_f) 
print("A1_f", A1_f) 
print("A2_f", A2_f) 

#   Store the results in a dict
paramResults['k0_m'] = k0_m
paramResults['k1_m'] = k1_m
paramResults['A0_m'] = A0_m
paramResults['A1_m'] = A1_m
paramResults['A2_m'] = A2_m
paramResults['A0_f'] = A0_f
paramResults['A1_f'] = A1_f
paramResults['A2_f'] = A2_f

############################################################
###          3.  Specify Settings for Global Solution    ###                 
############################################################
#Key values for the global solution, if you want to run this section separately
#years = 1 
#NumOFsamples = 1

#   Simulates the time series for x and s 
x_simul, s_simul, cg_simul, dg_simul = LowOfMotion.Simul_States(years, NumOFsamples, param) 

#   Compute the min and max value for x and s
xmin = np.min(x_simul)
xmax = np.max(x_simul)

smin = np.min(s_simul)
smin = max(1e-14,smin)
smax = np.max(s_simul)

#   If reproduce_Matlab is activated 
if reproduce_Matlab == 1 or 2:    
    if ModelChoice == 1:
        xmin, xmax, smin, smax = OrignialMatlabResults.Res_XSmaxmin(Getresult=True)
        print("Orginial results from BY2004:  xmin, xmax, smin and smax is loaded")
    if ModelChoice == 2:
        xmin, xmax, smin, smax = OrignialMatlabResults.Res_XSmaxmin(Getresult=True)
        print("Orginial results from BKY2012: xmin, xmax, smin and smax is loaded")

#   The approximation degree for x-state and s-state
degX = 6 
degS = 6 

#   The number of nodes for collocation projection
nX = degX + 1 
nS = degS + 1 

#   Compute Chebychev on [-1,1] 
zX = MyChebyshev.Chebyshev_NodesBasic(nX)
zS = MyChebyshev.Chebyshev_NodesBasic(nS)

#   Adjust nodes to approximation range
x = MyChebyshev.Chebyshev_NodesAdj(xmin, xmax, zX, nX) 
s = MyChebyshev.Chebyshev_NodesAdj(smin, smax, zS, nS) 

#   Compute Gauss-Hermite quadrature nodes and weights 
nGH = 5 # Number of Gauss-Hermite Nodes

[xi, wi] = np.polynomial.hermite.hermgauss(nGH)

#   Compute weights for 2D quadrature
wij = np.reshape(wi, (nGH,1)) *wi 

#   Adjust to state grid (dimension 1: x today; dimension 2: s today; dimension 3: shock x; dimension 4: shock s)
wij_full = np.repeat( np.repeat( np.reshape(wij, (1,1,nGH,nGH)), 7) , 7).reshape(5,5,7,7)

#########################################################
###          4.  Compute Projection Solution          ###                 
#########################################################

#   The set up state grid for the subsequent period (quadrature nodes): 
#       Dimension 1: x today 
#       Dimension 2: s today 
#       Dimension 3: shock x
#       Dimension 4: shock s
xprime = rho * np.tile( np.repeat(x,nS), (nGH, nGH) ).reshape(5,5,7,7) + 2**(0.5)*phi_x * np.tile( np.tile( np.transpose( s**(0.5)  ), nX ), (nGH, nGH) ).reshape(5,5,7,7)*\
    np.tile(np.repeat(np.repeat(xi, nX) ,nS), nGH).reshape(5,5,7,7) 

sprime = sigma_bar**(2)*(1 - nu) + nu * np.tile( np.tile( np.transpose( np.transpose(s)  ), nX ), (nGH, nGH) ).reshape(5,5,7,7) +\
    2**(0.5) * phi_s * np.repeat(np.repeat(np.repeat(xi, nX) ,nS), nGH).reshape(5,5,7,7)

#   Set up grid for consumption growth (cg) and dividend growth (dg):
#       Dimension 1: x today 
#       Dimension 2: s today 
#       Dimension 3: shock cg
#       Dimension 4: shock dg
cgprime = mu_c + np.tile( np.repeat(x,nS), (nGH, nGH) ).reshape(5,5,7,7) + 2**(0.5) * np.tile( np.tile( np.transpose( s**(0.5)  ), nX ), (nGH, nGH) ).reshape(5,5,7,7)*\
    np.tile(np.repeat(np.repeat(xi, nX) ,nS), nGH).reshape(5,5,7,7) 

dgprime = mu_d + phi*np.tile(np.repeat(x, nS),(nGH, nGH)).reshape(5,5,7,7)+\
    2**(1/2)* phi_dc *np.tile(np.tile(np.transpose(s**(1/2)), nX), (nGH, nGH)).reshape(5,5,7,7)*\
        np.tile(np.repeat(np.repeat(xi,nX), nS), nGH).reshape(5,5,7,7)+\
            2**(1/2)* phi_d*np.tile(np.tile(np.transpose(s**(1/2)), nX), (nGH, nGH)).reshape(5,5,7,7)*\
                np.repeat(np.repeat(np.repeat(xi,nX), nS), nGH).reshape(5,5,7,7)

#   Compute chebychev basis functions for state grid
[x_grid,s_grid]  = [ np.repeat(x, nX).reshape(7,7) ,np.repeat(s, nS).reshape(7,7) ] 

#   Basis functions for states today
Tx, Ts, Size_grid, N_grid = MyChebyshev.Cheby2D_Basis_Functions(x_grid, xmin, xmax, s_grid, smin, smax, degX, degS , Prime = False)

#   Basis functions for states tomorrow
TxPrime, TsPrime, Size_gridPrime, N_gridPrime = MyChebyshev.Cheby2D_Basis_Functions(xprime, xmin, xmax, sprime, smin, smax, degX, degS, Prime = True)

####    Compute log wealth-consumption ratio:
# Initial values for solution coefficients:
coefsWC0 = np.zeros((nX,nS))
coefsWC0[0,0] = A0
vars0 = coefsWC0.reshape(-1,1)

#   Set solver options
argumentslist = degS, degX, Tx, Ts, Size_grid, N_grid,TxPrime, TsPrime, Size_gridPrime, N_gridPrime, delta, theta, psi, wij_full, nGH, cgprime

#   Start solver
sol_wc_1 = fsolve(MySolvers.objective_WC, vars0, args= argumentslist , xtol = 1.0000e-06)

#   If reproduce_Matlab is activated 
if reproduce_Matlab == 2:
    if ModelChoice == 1:
        coefsWC = OrignialMatlabResults.Res_coefsWC_BY2004(Getresult=True)
        print("Orginial results from BK2004: coefsWC is loaded")
    if ModelChoice == 2:
        coefsWC = OrignialMatlabResults.Res_coefsWC(Getresult=True)
        print("Orginial results from BKY2012: coefsWC is loaded")
else:
    # The results from fslove in Python is used
    coefsWC = sol_wc_1

# Adjustment to get the right shape for coefsWC
i = 0
while i < len(coefsWC):
    vars0[i,0] = coefsWC[i] 
    i += 1
coefsWC = vars0

####    Extract Solution
#   Solution coefficients
coefsWC = coefsWC.reshape(degX+1, degS+1)
#   log wealth-consumption ratio at collocation nodes
wc = MyChebyshev.Cheby2D_Eval(coefsWC,Tx,Ts,Size_grid,N_grid, Prime = False) 
#   log wealth-consumption ratio in the subsequent period (evaluated at quadrature nodes)
wc_Prime = MyChebyshev.Cheby2D_Eval(coefsWC,TxPrime,TsPrime,Size_gridPrime,N_gridPrime,Prime = True) 

####    Compute log price-dividend Ratio:
# Initial values for solution coefficients:
coefsPD0 = np.zeros((nX,nS))
coefsPD0[0,0] = A0_m
vars0 = coefsPD0.reshape(-1,1)

#   Set solver options
argumentslist_PD = degS, degX, Tx, Ts, Size_grid, N_grid,TxPrime, TsPrime, Size_gridPrime, N_gridPrime, delta, theta, psi, wij_full, nGH, cgprime, dgprime, wc_Prime, wc

#   Start solver
sol_pd_1 = fsolve(MySolvers.objective_PD, vars0, args= argumentslist_PD , xtol = 1.0000e-06)

#   If reproduce_Matlab is activated 
if reproduce_Matlab == 2:
    if ModelChoice == 1:
        coefsPD = OrignialMatlabResults.Res_coefsPD_BY2004(Getresult=True)
        print("Orginial results from BK2004: coefsPD is loaded")
    if ModelChoice == 2:
        coefsPD = OrignialMatlabResults.Res_coefsPD(Getresult=True)
        print("Orginial results from BKY2012: coefsPD is loaded")
else:
    # The results from fslove in Python is used
    coefsPD = sol_pd_1

# Adjustment to get the right shape for coefsPD
i = 0
while i < len(coefsPD):
    vars0[i,0] = coefsPD[i] 
    i += 1
coefsPD = vars0

####    Extract Solution
#   Solution coefficients
coefsPD = coefsPD.reshape(degX+1, degS+1)
#   log wealth-consumption ratio at collocation nodes
pd = MyChebyshev.Cheby2D_Eval(coefsPD,Tx,Ts,Size_grid,N_grid, Prime = False) 
#   log wealth-consumption ratio in the subsequent period (evaluated at quadrature nodes)
pd_Prime = MyChebyshev.Cheby2D_Eval(coefsPD,TxPrime,TsPrime,Size_gridPrime,N_gridPrime,Prime = True) 

####    Compute Risk-Free Rate:
# Initial values for solution coefficients:
coefsPf0 = np.zeros((nX,nS))
coefsPf0[0,0] = A0_f
vars0 = coefsPf0.reshape(-1,1)

#   Set solver options
argumentslist_Pf = Tx, Ts, Size_grid, N_grid, delta, theta, psi, wij_full, nGH, cgprime, wc_Prime, wc

#   Start solver
sol_pf_1 = fsolve(MySolvers.objective_Pf, vars0, args= argumentslist_Pf , xtol = 1.0000e-06)

#   If reproduce_Matlab is activated 
if reproduce_Matlab == 2:
    if ModelChoice == 1:
        coefsPD = OrignialMatlabResults.Res_coefsPf_BY2004(Getresult=True)
        print("Orginial results from BK2004: coefsPD is loaded")
    if ModelChoice == 2:
        coefsPf = OrignialMatlabResults.Res_coefsPf(Getresult=True)
        print("Orginial results from BKY2012: coefsPD is loaded")
else:
    # The results from fslove in Python is used
    coefsPf = sol_pf_1

# Adjustment to get the right shape for coefsPD
i = 0
while i < len(coefsPf):
    vars0[i,0] = coefsPf[i] 
    i += 1
coefsPf = vars0

####    Extract Solution
#   Solution coefficients
coefsPf = coefsPf.reshape(degX+1, degS+1)
#   log wealth-consumption ratio at collocation nodes
pf = MyChebyshev.Cheby2D_Eval(coefsPf,Tx,Ts,Size_grid,N_grid, Prime = False) 

#### Compute log-linear solution at collocation nodes
wc_LL = A0 + A1*x_grid + A2*np.transpose( s_grid)
wc_Prime_LL = A0 + A1*xprime + A2*sprime

pd_LL = A0_m + A1_m*x_grid + A2_m*np.transpose( s_grid)
pd_Prime_LL = A0_m + A1_m*xprime + A2_m*sprime

pf_LL = A0_f + A1_f*x_grid + A2_f*np.transpose( s_grid)

print("Compute global solution: done")

#### Plot the results:

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(s,x,pd_LL)
surf2 = ax.plot_surface(s,x,pd)
ax.set_xlabel(r'$\sigma_{t}^{2}$')
ax.set_ylabel(r"$x_{t}$")
ax.set_zlabel(r"$p_{t}-d_{t}$")
plt.show()


######################################################
###                                                ###
###     5. Run Simulations for Annualized Moments  ###
###                                                ###
######################################################
print("Compute the long-run simulations")
# Number of simulated years
years = years2 
# Number of simulated months
T = years*12 
# Number of sample paths
NumOFsamples = NumOFsamples2

# Simulate states:
x_simul, s_simul, cg_simul, dg_simul = LowOfMotion.Simul_States(years, NumOFsamples, param) # Simulate long time series for x and s

#   If reproduce_Matlab is activated 
if reproduce_Matlab == 1 or 2:
    if ModelChoice == 2:
        x_simul_Org, s_simul_Org, cg_simul_Org, dg_simul_Org = OrignialMatlabResults.Res_Simsim(Getresult=True)
    
        i = 0
        while i < len(x_simul):
            x_simul[i] = x_simul_Org[i]
            s_simul[i] = s_simul_Org[i]
            cg_simul[i] = cg_simul_Org[i]
            dg_simul[i] = dg_simul_Org[i]
            i += 1
        print("Orginial results from BKY2012: x_simul, s_simul, cg_simul and dg_simul is loaded")

# Compute Asset Prices
pd_simul_LL = A0_m + A1_m *x_simul + A2_m *s_simul    # Log-linear solution for log price-dividend ratio
wc_simul_LL = A0 + A1 *x_simul + A2 *s_simul         # Log-linear solution for log wealth-consumpion ratio
pf_simul_LL = A0_f + A1_f *x_simul + A2_f *s_simul   # Log-linear solution for price of riskless asset ratio

pd_simul = np.zeros((T+1,NumOFsamples)) # Projection solution for log price-dividend ratio
wc_simul = np.zeros((T+1,NumOFsamples)) # Projection solution for log wealth-consumpion ratio
pf_simul = np.zeros((T+1,NumOFsamples)) # Projection solution for price of riskless asset ratio

ns = 0
while ns < NumOFsamples:
    t = 0
    while t < T+1:
        pd_simul[t,ns] = MyChebyshev.Cheby2D_1Node(coefsPD, x_simul[t,ns], xmin, xmax, s_simul[t,ns], smin, smax, degX, degS)
        wc_simul[t,ns] = MyChebyshev.Cheby2D_1Node(coefsWC, x_simul[t,ns], xmin, xmax, s_simul[t,ns], smin, smax, degX, degS)
        pf_simul[t,ns] = MyChebyshev.Cheby2D_1Node(coefsPf, x_simul[t,ns], xmin, xmax, s_simul[t,ns], smin, smax, degX, degS)
        t +=1
    ns +=1

PD_simul = np.exp(pd_simul)
Pf_simul = np.exp(pf_simul)
PD_simul_LL = np.exp(pd_simul_LL)
Pf_simul_LL = np.exp(pf_simul_LL)
WC_simul = np.exp(wc_simul)
WC_simul_LL = np.exp(wc_simul_LL)

# Compute monthly returns
tempRet_M1 = ( PD_simul[1:len(PD_simul-1),:] + 1) 
tempRet_M2 = ( PD_simul[0:len(PD_simul-2),:] )
tempRet_M3 = np.exp( dg_simul[1:len(dg_simul-1),:] )

Ret_M = np.zeros( (len(tempRet_M1),1) )
i = 0
while i < len(tempRet_M1):
    Ret_M[i] = tempRet_M1[i] / tempRet_M2[i] *tempRet_M3[i]
    i += 1

del tempRet_M1, tempRet_M2, tempRet_M3

tempRet_M_LL1 = ( PD_simul_LL[1:len(PD_simul-1),:] + 1) 
tempRet_M_LL2 = ( PD_simul_LL[0:len(PD_simul_LL-2),:] )
tempRet_M_LL3 = np.exp( dg_simul[1:len(dg_simul-1),:] )

Ret_M_LL = np.zeros( (len(tempRet_M_LL1),1) )
i = 0
while i < len(tempRet_M_LL1):
    Ret_M_LL[i] = tempRet_M_LL1[i] / tempRet_M_LL2[i] *tempRet_M_LL3[i]
    i += 1

del tempRet_M_LL1, tempRet_M_LL2, tempRet_M_LL3

# Copmute log returns
ret_m = np.log(Ret_M)
ret_rf = - np.log(Pf_simul[0:len(Pf_simul)-1,:])
ret_m_LL = k0_m + k1_m * pd_simul_LL[1:len(pd_simul_LL),:]  - pd_simul_LL[0:len(pd_simul_LL)-1,:] + dg_simul[1:len(dg_simul),:] 
ret_rf_LL = -np.log(Pf_simul_LL[0:len(Pf_simul_LL)-1,:])

# Call scrippt to compute annualized time series and statistics
import Annualized_Statistics

cg_simul_annual, dg_simul_annual, SimRes = Annualized_Statistics.AS_Results(years, NumOFsamples, ret_m, ret_rf, ret_m_LL, ret_rf_LL, dg_simul, cg_simul,PD_simul, WC_simul, PD_simul_LL, WC_simul_LL)

# Collect Results
print("Annualized Moments and Errors (Table 5):")
print(" ")
print("    E(pd)     std(p-d)  E(r_a-rf) E(rf)     std(r_a)  std(rf_a)")
tableResults = np.zeros((3,6))
tableResults[0,0] = SimRes['E_pd_LL'] 
tableResults[0,1] = SimRes['vol_pd_LL'] 
tableResults[0,2] = SimRes['EP_LL'] 
tableResults[0,3] = SimRes['E_rf_a_LL'] 
tableResults[0,4] = SimRes['Vol_ret_a_LL'] 
tableResults[0,5] = SimRes['Vol_ret_rf_a_LL'] 

tableResults[1,0] = SimRes['E_pd'] 
tableResults[1,1] = SimRes['vol_pd'] 
tableResults[1,2] = SimRes['EP'] 
tableResults[1,3] = SimRes['E_rf_a'] 
tableResults[1,4] = SimRes['Vol_ret_a'] 
tableResults[1,5] = SimRes['Vol_ret_rf_a'] 

j = 0 
while j <6:
    tableResults[2,j] = np.absolute( tableResults[0,j] / tableResults[1,j] -1 )
    j +=1

tableResults
print("First line log linear solution; second line global solution; third line: relative error")


######################################################
###                                                ###
###     6. Compute Euler Errors                    ###
###                                                ###
######################################################

print("Compute euler errors: start ")

n_eval = 100 #500 # number of uniform nodes in each dimension
x_eval = np.linspace(xmin, xmax, n_eval).reshape(-1,1)  # evaluation nodes for x
s_eval = np.linspace(smin ,smax, n_eval).reshape(-1,1) # evaluation nodes for s

[x_grid_eval, s_grid_eval] = [ np.repeat(x_eval, n_eval).reshape(n_eval,n_eval) ,np.repeat(s_eval, n_eval).reshape(n_eval,n_eval) ] # rectangular state grid of evaluation nodes
s_grid_eval = np.transpose(s_grid_eval)

# Set up state grid for the subsequent period (quadrature nodes):
xprime_eval = rho * np.tile( np.repeat(x_eval,n_eval), (nGH, nGH) ).reshape(nGH,nGH,n_eval,n_eval) + 2**(0.5)*phi_x * np.tile( np.tile( np.transpose( s_eval**(0.5)  ), n_eval ), (nGH, nGH) ).reshape(nGH,nGH,n_eval,n_eval) *\
    np.tile(np.repeat(np.repeat(xi, n_eval) ,n_eval), nGH).reshape(nGH,nGH,n_eval,n_eval) 

sprime_eval = sigma_bar**(2)*(1 - nu) + nu * np.tile( np.tile( np.transpose( np.transpose(s_eval)  ), n_eval ), (nGH, nGH) ).reshape(nGH,nGH,n_eval,n_eval) +\
    2**(0.5) * phi_s * np.repeat(np.repeat(np.repeat(xi, n_eval) ,n_eval), nGH).reshape(nGH,nGH,n_eval,n_eval) 

cgprime_eval = mu_c + np.tile( np.repeat(x_eval,n_eval), (nGH, nGH) ).reshape(nGH,nGH,n_eval,n_eval) + 2**(0.5) * np.tile( np.tile( np.transpose( s_eval**(0.5)  ), n_eval ), (nGH, nGH) ).reshape(nGH,nGH,n_eval,n_eval) *\
    np.tile(np.repeat(np.repeat(xi, n_eval) ,n_eval), nGH).reshape(nGH,nGH,n_eval,n_eval) 

dgprime_eval = mu_d + phi*np.tile(np.repeat(x_eval, n_eval),(nGH, nGH)).reshape(nGH,nGH,n_eval,n_eval)+\
    2**(1/2)* phi_dc *np.tile(np.tile(np.transpose(s_eval**(1/2)), n_eval), (nGH, nGH)).reshape(nGH,nGH,n_eval,n_eval)*\
        np.tile(np.repeat(np.repeat(xi,n_eval), n_eval), nGH).reshape(nGH,nGH,n_eval,n_eval)+\
            2**(1/2)* phi_d*np.tile(np.tile(np.transpose(s_eval**(1/2)), n_eval), (nGH, nGH)).reshape(nGH,nGH,n_eval,n_eval)*\
                np.repeat(np.repeat(np.repeat(xi,n_eval), n_eval), nGH).reshape(nGH,nGH,n_eval,n_eval)


# Log wealth-consumption and price-dividend ratio at evaluation nodes (global solution)
wc_eval = MyChebyshev.Cheby2D(coefsWC, x_grid_eval, xmin, xmax, s_grid_eval ,smin, smax, degX, degS, Prime = False)
pd_eval = MyChebyshev.Cheby2D(coefsPD, x_grid_eval, xmin, xmax, s_grid_eval ,smin, smax, degX, degS, Prime = False)

wc_prime_eval = MyChebyshev.Cheby2D(coefsWC, xprime_eval, xmin, xmax, sprime_eval, smin, smax, degX, degS, Prime = True)
pd_prime_eval = MyChebyshev.Cheby2D(coefsPD, xprime_eval, xmin, xmax, sprime_eval, smin, smax, degX, degS, Prime = True)

# Log wealth-consumption and price-dividend ratio at evaluation nodes (log-linear solution)
wc_eval_LL = A0 + A1 * x_grid_eval + A2 * s_grid_eval
pd_eval_LL = A0_m + A1_m * x_grid_eval + A2_m * s_grid_eval

wc_prime_eval_LL = A0 + A1 * xprime_eval + A2 * sprime_eval
pd_prime_eval_LL = A0_m + A1_m * xprime_eval + A2_m * sprime_eval

# Gauss-Hermite weights
wij_eval = np.tile( (wi.reshape(-1,1) *wi ).reshape(nGH, nGH, 1, 1 ), (1,1,n_eval,n_eval) )

"""
#I get in trouble here:

# Copmute euler errors in pricing equation of wealth and equities (global solution)
wc_prime_eval =wc_prime_eval.reshape(5,5,100,100) 
euler_error_wc_proj = 

1 - delta**(theta) * np.pi**(-1) 
###
np.sum( np.sum( wij_eval * ( np.exp(wc_prime_eval) / (np.tile( np.exp(wc_eval) ,(nGH, nGH, 1,1) ) -1) )**(theta) ,out (5, 1,100,100) ) ,axis=0).shape

sum (sum( wij_eval * ( np.exp(wc_prime_eval) / (np.tile( np.exp(wc_eval) ,(nGH, nGH, 1,1) ) -1) )**(theta)))

wij_eval #OK
np.exp(wc_prime_eval) #OK
(np.tile( np.exp(wc_eval) ,(nGH, nGH, 1,1) ) -1)

1 - 
par.delta^par.theta * pi^(-1) * sum(sum( wij_eval .* (exp(wc_prime_eval)./(repmat(exp(wc_eval),1,1,nGH,nGH)-1)).^(par.theta),3),4) .* (pi^(-1) .* sum(sum(wij_eval.* exp((par.theta-par.theta/par.psi).*cgprime_eval),3),4));

print('so far')
"""


