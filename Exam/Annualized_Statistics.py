#Annualized_Statistics
import numpy as np

def AS_Results(years, NumOFsamples, ret_m, ret_rf, ret_m_LL, ret_rf_LL, dg_simul, cg_simul,PD_simul, WC_simul, PD_simul_LL, WC_simul_LL ):
    """Script to compute annualized time series and statistics
    """
    # Annualized log returns
    j = 0
    ret_a = np.zeros( (years,NumOFsamples) )
    ret_rf_a = np.zeros( (years,NumOFsamples) )
    ret_a_LL = np.zeros( (years,NumOFsamples) )
    ret_rf_a_LL = np.zeros( (years,NumOFsamples) )
    while j < NumOFsamples:
        ret_a = ( np.sum(ret_m[:,j].reshape(12,years, order='F'), axis= 0) ).reshape(-1,1) 
        ret_rf_a = ( np.sum(ret_rf[:,j].reshape(12,years, order='F'), axis= 0) ).reshape(-1,1) 
        ret_a_LL = ( np.sum(ret_m_LL[:,j].reshape(12,years, order='F'), axis= 0) ).reshape(-1,1) 
        ret_rf_a_LL = ( np.sum(ret_rf_LL[:,j].reshape(12,years, order='F'), axis= 0) ).reshape(-1,1) 

        j +=1
    
    # Annualized gross returns
    Ret_A = np.exp(ret_a)-1 
    Ret_rf_A = np.exp(ret_rf_a)-1 

    Ret_A_LL = np.exp(ret_a_LL)-1 
    Ret_rf_A_LL = np.exp(ret_rf_a_LL)-1 

    # Annualized log market over risk free return
    rp_a = ret_a - ret_rf_a 
    rp_a_LL = ret_a_LL - ret_rf_a_LL 

    # Copmute annualized wealth-consumption and price-dividend ratio:

    #% Explanation:
    # The annualized price-dividend Ratio is computed by multiplying the
    # monthly price-dividend Ratio in the last month of the year by dividends
    # in the last month of the year. The result is then divided by the sum over all dividends 
    # of the corresponding year. That is what the following lines do using
    # growth rates instead of levels.


    AnnualDG = np.zeros((12,NumOFsamples))
    AnnualCG = np.zeros((12,NumOFsamples))
    SumGrowthd = np.zeros((years,NumOFsamples))
    SumGrowthc = np.zeros((years,NumOFsamples))
    SumCumGrowthd = np.zeros((years,NumOFsamples))
    SumCumGrowthc = np.zeros((years,NumOFsamples))

    i = 0
    while i < years:
        j = 0
        while j < NumOFsamples:
                #Extract monthly log dividend growth value for a given year
            AnnualDG = dg_simul[i*12+1: i*12+12+1 ,j ] .reshape(-1,1) 
            AnnualCG = cg_simul[i*12+1: i*12+12+1 ,j ] .reshape(-1,1)  
        
            # Annual Dividend Growth (not in logs)
            SumGrowthd[i,j] = np.exp(np.sum(AnnualDG)) 
            SumGrowthc[i,j] = np.exp(np.sum(AnnualCG)) 
            
            # Sum Over all Cumulated Monthly Dividend Growths
            SumCumGrowthd[i,j] = np.sum(np.exp(np.cumsum(AnnualDG))) 
            SumCumGrowthc[i,j] = np.sum(np.exp(np.cumsum(AnnualCG)))
            j +=1
        i +=1
 

    # Global Solution
    i = 0
    PD_simul_annual = np.zeros((years,NumOFsamples))
    WC_simul_annual = np.zeros((years,NumOFsamples))
    while i < years:
        j = 0
        while j < NumOFsamples:
            PD_simul_annual[i,j] = PD_simul[ i*12+12, j] *SumGrowthd[i,j] /SumCumGrowthd[i,j]
            WC_simul_annual[i,j] = WC_simul[ i*12+12, j] *SumGrowthc[i,j]/SumCumGrowthc[i,j]
            j +=1
        i +=1

    pd_simul_annual = np.log(PD_simul_annual)
    wc_simul_annual = np.log(WC_simul_annual)

    # Log-Linear Solution
    i = 0
    PD_simul_annual_LL = np.zeros((years,NumOFsamples))
    WC_simul_annual_LL = np.zeros((years,NumOFsamples))
    while i < years:
        j = 0
        while j < NumOFsamples:
            PD_simul_annual_LL[i,j] = PD_simul_LL[ i*12+12, j] *SumGrowthd[i,j] /SumCumGrowthd[i,j]
            WC_simul_annual_LL[i,j] = WC_simul_LL[ i*12+12, j] *SumGrowthc[i,j]/SumCumGrowthc[i,j]
            j +=1
        i +=1

    pd_simul_annual_LL = np.log(PD_simul_annual_LL) 
    wc_simul_annual_LL = np.log(WC_simul_annual_LL) 


    # Annualized log consumption and dividend growth
    cg_simul_annual = np.zeros((years,NumOFsamples))
    dg_simul_annual = np.zeros((years,NumOFsamples))

    j= 0
    while j < NumOFsamples:
        cg_simul_annual[0,j] = None
        dg_simul_annual[0,j] = None
        j +=1

    i = 0
    while i < years-1:
        j = 0
        while j < NumOFsamples:
            cg_simul_annual[i+1,j] = np.log(SumCumGrowthc[i+1,j] / SumCumGrowthc[i,j] * SumGrowthc[i,j] )
            dg_simul_annual[i+1,j] = np.log(SumCumGrowthd[i+1,j] / SumCumGrowthd[i,j] * SumGrowthd[i,j] )
            j +=1
        i +=1

    del AnnualDG, AnnualCG, SumGrowthd, SumGrowthc, SumCumGrowthd, SumCumGrowthc

    SimRes = dict()
   
    SimRes['E_pd']  = np.median( np.mean(pd_simul_annual) ) # expected value of the annualized price-dividend ratio (global solution)
    SimRes['vol_pd'] = np.median( np.std(pd_simul_annual) ) # volatility of the annualized price-dividend ratio (global solution)

    SimRes['E_wc']  =  np.median( np.mean(wc_simul_annual) )     #expected value of the annualized price-dividend ratio (global solution
    SimRes['vol_wc']  =  np.median( np.std(wc_simul_annual) )      #volatility of the annualized wealth-consumption ratio (global solution)

    SimRes['E_pd_LL']  = np.median( np.mean(pd_simul_annual_LL) )       # expected value of the annualized price-dividend ratio (log-linear solution)
    SimRes['vol_pd_LL']  = np.median( np.std(pd_simul_annual_LL) )      # volatility of the annualized price-dividend ratio (log-linear solution)

    SimRes['E_wc_LL']  = np.median( np.mean(wc_simul_annual_LL) )       # expected value of the annualized price-dividend ratio (log-linear solution)
    SimRes['vol_pd_LL']  =  np.median( np.std(wc_simul_annual_LL) )     # volatility of the annualized wealth-consumption ratio (log-linear solution)

    SimRes['E_ret_a']  = np.median( np.mean(Ret_A) )       # expected annual gross market return (global solution)
    SimRes['E_rf_a']  =  np.median( np.mean(Ret_rf_A) )      # expected annual gross risk-free return (global solution)

    SimRes['E_ret_a_LL']  = np.median( np.mean(Ret_A_LL) )        # expected annual gross market return (log-linear solution)
    SimRes['E_rf_a_LL']  = np.median( np.mean(Ret_rf_A_LL) )        # expected annual gross risk-free return (log-linear solution)

    SimRes['EP']  =   np.median( np.mean(Ret_A) - np.mean(Ret_rf_A) )  # expected annual gross market over risk-free return (global solution)
    SimRes['EP_LL']  = np.median( np.mean(Ret_A_LL) - np.mean(Ret_rf_A_LL) )      # expected annual gross market over risk-free return (log-linear solution)

    SimRes['Vol_ret_a']  = np.median( np.std(Ret_A) )       # volatility of annual gross market return (global solution)
    SimRes['Vol_ret_rf_a']  = np.median( np.std(Ret_rf_A) )       # volatility of annual gross risk-free return (global solution)

    SimRes['Vol_ret_a_LL']  =  np.median( np.std(Ret_A_LL) )       # volatility of gross market return (log-linear solution)
    SimRes['Vol_ret_rf_a_LL']  =  np.median( np.std(Ret_rf_A_LL) )       # volatility of annual gross risk-free return (log-linear solution)

    return cg_simul_annual, dg_simul_annual, SimRes
