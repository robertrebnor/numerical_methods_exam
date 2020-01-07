import numpy as np
import MyChebyshev

def objective_WC(coef, *argumentslist):

    degS, degX, Tx, Ts, Size_grid, N_grid,TxPrime, TsPrime, Size_gridPrime, N_gridPrime, delta, theta, psi, wij_full, nGH, cgprime = argumentslist

    coefsWC_test = coef.reshape( degS+1, degX+1)

    # Wealth-Consumption Ratio Today
    wc = np.exp( MyChebyshev.Cheby2D_Eval(coefsWC_test,Tx,Ts,Size_grid,N_grid, Prime = False) )
  
    # Wealth-Consumption Ratio Tomorrow
    wc_prime = np.exp ( MyChebyshev.Cheby2D_Eval(coefsWC_test,TxPrime,TsPrime,Size_gridPrime,N_gridPrime,Prime = True) )
  
    # Asset Pricing Equation for Wealth-Consumption Ratio
    F = 1 - delta**(theta) * ( 1/np.pi ) *  np.sum( np.sum( wij_full * ( wc_prime/(np.tile(wc, (nGH,nGH,1,1) ) -1)  )**(theta) ,axis=0) ,axis=0)*( 1/np.pi ) * np.sum( np.sum( wij_full * np.exp( (theta - theta/psi)*cgprime),axis=0), axis=0)
  
    Fvec = np.transpose(F).reshape(-1,1)
    F = Fvec.flatten()

    return F


def objective_PD(coef, *argumentslist_PD):

    degS, degX, Tx, Ts, Size_grid, N_grid,TxPrime, TsPrime, Size_gridPrime, N_gridPrime, delta, theta, psi, wij_full, nGH, cgprime, dgprime, WC_Prime, WC = argumentslist_PD

    WC_Prime = np.exp(WC_Prime)
    WC = np.exp(WC)

    coefsPD_test = coef.reshape( degS+1, degX+1)

    # Price-Dividend Ratio Today
    PD = np.exp( MyChebyshev.Cheby2D_Eval(coefsPD_test,Tx,Ts,Size_grid,N_grid, Prime = False) )
  
    # Price-Dividend Ratio Tomorrow
    PD_Prime = np.exp ( MyChebyshev.Cheby2D_Eval(coefsPD_test,TxPrime,TsPrime,Size_gridPrime,N_gridPrime,Prime = True) )
  
    # Asset Pricing Equation for Price-Dividend Ratio
    F = 1 - delta**(theta) * ( 1/np.pi ) *  np.sum( np.sum( wij_full * ( WC_Prime/(np.tile(WC, (nGH,nGH,1,1) ) -1)  )**(theta -1) * ( (PD_Prime + 1)/ (np.tile(PD, (nGH,nGH,1,1)) )) ,axis=0) ,axis=0)\
        *( 1/np.pi ) * np.sum( np.sum( wij_full * np.exp( (theta - 1 - theta/psi)*cgprime + dgprime ),axis=0), axis=0)
   
    Fvec = np.transpose(F).reshape(-1,1)
    F = Fvec.flatten()

    return F

def objective_Pf(coef, *argumentslist_Pf):

    Tx, Ts, Size_grid, N_grid, delta, theta, psi, wij_full, nGH, cgprime, WC_Prime, WC = argumentslist_Pf

    WC_Prime = np.exp(WC_Prime)
    WC = np.exp(WC)

    coefsPf = coef.reshape( WC.shape[0], WC.shape[1])

    # Price of risk-free asset Today
    Pf = np.exp( MyChebyshev.Cheby2D_Eval(coefsPf,Tx,Ts,Size_grid,N_grid, Prime = False) )

    # Asset Pricing Equation for Risk-Free Rate
    F = Pf - delta**(theta) * ( 1/np.pi ) *  np.sum( np.sum( wij_full * ( WC_Prime/(np.tile(WC, (nGH,nGH,1,1) ) -1)  )**(theta-1) ,axis=0) ,axis=0)*( 1/np.pi ) * np.sum( np.sum( wij_full * np.exp( (theta - 1 - theta/psi)*cgprime),axis=0), axis=0)

    Fvec = np.transpose(F).reshape(-1,1)
    F = Fvec.flatten()

   
    Fvec = np.transpose(F).reshape(-1,1)
    F = Fvec.flatten()

    return F





