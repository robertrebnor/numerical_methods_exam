#########################################################
###                                                   ###
###           Compute Chebyshev nodes                 ###
###                                                   ###                     
#########################################################
"""Overview of the program:
    
"""
import numpy as np 

# Chebyshev nodes
def Chebyshev_NodesBasic(n_nodes):
    """ Calculates the Chebyshev nodes on the interval [-1,1]

    Input:
    ------
    n_nodes :  number
        The numer of nodes on the interval [-1,1]

    Output
    ------
    The Chebyshev nodes on the interval [-1,1]
    """
    z_k = []
    k = 1
    while k <= n_nodes:
        cal_node = 0
        cal_node = - np.cos( ( (2*k -1) / (2*n_nodes) )*np.pi  )
        z_k.append(cal_node)
        k = k +1
    z_k= np.array(z_k)
    return z_k

def Chebyshev_NodesAdj(xLow, xHigh, z_k, n_nodes):
    """ Adjusts the Chebyshev nodes to the interval [xLow, xHigh]

    Input:
    ------
    xLow: number
        The lowest x-value on the interval 
    xHigh: number
        The highest x-value in the interval
    z_k: array
        The Chebyshev nodes from the interval [-1,1]
    n_nodes :  number
        The numer of nodes in the interval

    Output
    ------
    The Chebyshev nodes adjusted to the interval [xLow, xHigh]
    """
    x_k = []
    k2 = 1
    while k2 <= n_nodes:
        adjusted_node = (z_k[k2-1] +1) * ( (xHigh - xLow) / 2 ) + xLow
        x_k.append(adjusted_node)
        k2 = k2 +1
    x_k= np.array(x_k)
    return x_k

def Cheby2D_Basis_Functions(X1, X1min, X1max, X2, X2min, X2max, degX1, degX2, Prime = False):
    """Compute 2D basis functions for the chebychev polynomials for each of the grid points in [X1_ij,X2_ij]
    """
    Size_Data = X1.shape
    Size_Data2 = X2.shape

    if Size_Data != Size_Data2:
        print('Size of Arrrays X1, X2 and X2 must be the same')

    if Prime == True:
        X1 = X1.reshape(5,5,-1,1, order ='F')
        X1 = X1.reshape(-1,1)
    if Prime == False:
        X1 = np.transpose(X1).reshape(-1,1)

    if Prime == True:
        X2 = X2.reshape(5,5,-1,1, order ='F')
        X2 = X2.reshape(-1,1)
    if Prime == False:
        X2 = X2.reshape(-1,1)
    
    N_Data = X1.shape[0]

    zX1 = (X1 - X1min)*2 / (X1max-X1min) - 1 
    zX2 = (X2 - X2min)*2 / (X2max-X2min) - 1 

    TX1 = np.zeros( ( degX1 +1, N_Data, degX2 +1) ) 
    TX2 = np.zeros( ( degX1 +1, N_Data, degX2 +1) )

    zX1_full = np.tile(zX1.reshape(1,-1), degX2+1).reshape(degX1+1, N_Data, 1) 
    zX2_full = np.tile(zX2, degX2+1) 
    TX1[:,:,0] = 1
    TX1[:,:,1] = zX1[:,0]
 
    i = 1
    while i < degX1:
        TX1[:,:,i+1] = 2* zX1[:,0] * TX1[:,:,i]  - TX1[:,:,i-1] 
        i +=1

    TX2[0,:,:] = 1 
    TX2[1,:,:] = zX2_full

    j = 1
    while j < degX2:
        TX2[j+1,:,:] = 2*zX2_full  * TX2[j,:,:]  - TX2[j-1,:,:]
        j +=1

    return  TX1, TX2, Size_Data , N_Data

def Cheby2D_Eval(coefs, TX1, TX2, Size_Data, N_data, Prime = False):
    """ Evaluate 2D chebychev polynomial given the coefficients coefs and the
        basis functions TX1 and TX2
    """

    if Prime == False:
        degX1 = coefs.shape[0] -1
        degX2 = coefs.shape[1] -1

        tempShape = TX1* TX2* np.tile( np.transpose(coefs).reshape(degX1 + 1, 1, degX2 + 1), (1, N_data,1) ) 

        tempShape2 = np.zeros((degX1+1,N_data,1))

        i = 0
        while i < (degX2+1):
            j = 0
            while j < N_data:
                tempShape2[i,j,0] = np.sum(tempShape[i,j,:], axis=0) 
                j += 1
            i +=1

        tempShapeFinal = np.zeros((N_data,1))
        i = 0
        while i < N_data:
            j = 0
            while j < (degX1 +1):
                tempShapeFinal[i,0] += tempShape2[j,i,0]
                j += 1
            i +=1

        v  = np.transpose( tempShapeFinal.reshape(Size_Data) )

    if Prime == True:
        degX1 = coefs.shape[0] -1
        degX2 = coefs.shape[1] -1

        tempShape = TX1* TX2* np.tile( np.transpose(coefs).reshape(degX1 + 1, 1, degX2 + 1), (1, N_data,1) ) 

        tempShape2 = np.zeros((degX1+1,N_data,1))

        i = 0
        while i < (degX2+1):
            j = 0
            while j < N_data:
                tempShape2[i,j,0] = np.sum(tempShape[i,j,:], axis=0) 
                j += 1
            i +=1

        tempShapeFinal = np.zeros((N_data,1))
        i = 0
        while i < N_data:
            j = 0
            while j < (degX1 +1):
                tempShapeFinal[i,0] += tempShape2[j,i,0]
                j += 1
            i +=1

        v  =  np.reshape( tempShapeFinal, Size_Data , order = 'C') 

        k = 0
        while k <5:
            i = 0
            while i < 5:
                v[k,i,:,:] = np.transpose(v[k,i,:,:])
                i +=1
            k +=1

    return v


def Cheby2D_1Node(coefs, X1, X1min, X1max, X2, X2min, X2max, degX1, degX2):
    """Evaluate 2D chebychev polynomial given the coefficients coefs at the node [X1, X2]
    """

    zX1 = (X1 - X1min)*2 / (X1max-X1min) - 1 
    zX2 = (X2 - X2min)*2 / (X2max-X2min) - 1 


    TX1 = np.zeros( ( degX1 +1, degX2 +1) ) 
    TX2 = np.zeros( ( degX1 +1, degX2 +1) ) 

    TX1[0,:] = 1
    TX1[1,:] = zX1

    j = 1 
    while j < (degX1):
        TX1[j+1, :] = 2 * zX1 * TX1[j,:] - TX1[j-1,:]
        j +=1

    TX2[:,0] = 1
    TX2[:,1] = zX2

    j = 1 
    while j < (degX2):
        TX2[:, j+1] = 2 * zX2 * TX2[:,j] - TX2[:,j-1]
        j +=1

    v = np.sum(  np.sum(TX1*TX2*coefs , axis=0) )

    return v

def Cheby2D(coefs, X1, X1min, X1max, X2, X2min, X2max, degX1, degX2, Prime = False):
    """Compute 2D basis functions for the chebychev polynomials for each of the grid points in [X1_ij,X2_ij]
    """
 
    Size_Data = X1.shape
    Size_Data2 = X2.shape

    if Size_Data != Size_Data2:
        print('Size of Arrrays X1, X2 and X2 must be the same')

    if Prime == True:
        X1 = X1.reshape(5,5,-1,1, order ='F')
        X1 = X1.reshape(-1,1)
    if Prime == False:
        X1 = np.transpose(X1).reshape(-1,1)

    if Prime == True:
        X2 = X2.reshape(5,5,-1,1, order ='F')
        X2 = X2.reshape(-1,1)
    if Prime == False:
        X2 = X2.reshape(-1,1)
    
    N_Data = X1.shape[0]

    zX1 = (X1 - X1min)*2 / (X1max-X1min) - 1 
    zX2 = (X2 - X2min)*2 / (X2max-X2min) - 1 

    TX1 = np.zeros( ( degX1 +1, N_Data, degX2 +1) ) 
    TX2 = np.zeros( ( degX1 +1, N_Data, degX2 +1) ) 

    zX1_full = np.tile(zX1.reshape(1,-1), degX2+1).reshape(degX1+1, N_Data, 1) 
    zX2_full = np.tile(zX2, degX2+1) 

    TX1[:,:,0] = 1
    TX1[:,:,1] = zX1[:,0]

    i = 1
    while i < degX1:
        TX1[:,:,i+1] = 2* zX1[:,0] * TX1[:,:,i]  - TX1[:,:,i-1] 
        i +=1

    TX2[0,:,:] = 1 
    TX2[1,:,:] = zX2_full

    j = 1
    while j < degX2:
        TX2[j+1,:,:] = 2*zX2_full  * TX2[j,:,:]  - TX2[j-1,:,:]
        j +=1

    v = Cheby2D_Eval(coefs, TX1, TX2, Size_Data, N_Data, Prime = False)

    return  v