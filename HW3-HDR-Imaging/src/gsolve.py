import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse

# this is the gsolve function. It has been completed and provided. You're welcome.
# Written by Yunhao Li from Northwestern Comp Photo Lab.
def gsolve(Z, B, l):
    """
    This function will plot the curve of the solved G function and the measured pixels.
    
    Don't worry. We have implemented this function for you. It should work right out of the box
    it the correct arguments are passed into
    
    Input:
    Z - Measured Brightness
    B - Log Exposure Times
    l - lambda
    Output:
    g - the gSolve
    lE - Log Erradiance of the image
    """
    Z = Z.astype(np.int64)
    n = 256
    w = np.ones((n,1)) / n
    m = Z.shape[0]
    p = Z.shape[1]
    A = np.zeros((m*p+n+1,n+m))
    b = np.zeros((A.shape[0],1))
    k = 0
    for i in range(m):
        for j in range(p):
            wij = w[Z[i,j]]
            A[k,Z[i,j]] = wij
            A[k,n+i] = -wij
            b[k,0] = wij * B[j]
            k += 1

    A[k,128] = 1
    k = k + 1
    for i in range(n-2):
        A[k,i] = l*w[i+1]
        A[k,i+1] = -2 * l * w[i+1]
        A[k,i+2] = l * w[i+1]
        k = k + 1
    x = np.linalg.lstsq(A,b,rcond=None)
    x = x[0]
    g = x[0:n]
    lE = x[n:x.shape[0]]
    return g.squeeze(),lE.squeeze()

def plotCurves(solveG, LE, logexpTime, zValues,mylambda):
    """
    This function will plot the curve of the solved G function and the measured pixels. You don't need to return anything in this function.
    
    You might want to implement the function "plotCurve" first for one specific color channel
    before you go on an plot this for all 3 channels
    
    Input
    solveG: A (256,1) array. Solved G function generated in the previous section.
    LE: Log Erradiance of the image.
    logexpTime: (k,) array, k is the number of input images. Log exposure time.
    zValues: m*n array. m is the number of sampling points, and n is the number of input images. Z value generated in the previous section. 
             Please note that in this function, we only take z value in ONLY ONE CHANNEL.
    title: A string. Title of the plot.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colormap = ["Red", "Green", "Blue"]
    plt.suptitle('Camera Response Curve for lambda = ' + str(mylambda))
    for c in range(3):
        zValuesChannel = zValues[:, :, c]
        LEChannel = LE[:, c]
        solveGChannel = solveG[:, c]
        x = np.zeros_like(zValuesChannel, dtype=float)
        outputG = np.zeros_like(zValuesChannel, dtype=float)
    
        #x values are the log radiance[i] + log exposure time[j]
        for i in range(logexpTime.shape[0]):
            for j in range(LEChannel.shape[0]):
                x[j, i] = LEChannel[j] + logexpTime[i]
        
        intensties = np.arange(0, 256)
        axs[c].plot(solveGChannel, intensties, label='G(Z)', color='red',linewidth=2.5)

        axs[c].scatter(x, zValuesChannel, label='Sample Points')
        axs[c].axis([np.min(x), np.max(x), 0, 255])
        axs[c].title.set_text('Channel ' + colormap[c])


    plt.show()
   









        

def plotCurve(solveG, LE, logexpTime, zValues, title):
    """
    Plots the curve of the solved G function and the measured pixels.

    Plotted Data should be in the form:
        x = Log Radiance + log exposure (same units as g(Z))
        y = Z values
        
    This function will plot the curve of the solved G function and the measured pixels. You don't need to return anything in this function.
    Input
    solveG: A (256,1) array. Solved G function generated in the previous section.
    LE: Log Erradiance of the image.
    logexpTime: (k,) array, k is the number of input images. Log exposure time.
    zValues: m*n array. m is the number of sampling points, and n is the number of input images. Z value generated in the previous section. 
    
    Please note that in this function, we only take z value in ONLY ONE CHANNEL.
    
    title: A string. Title of the plot.
    """
    
    #note, solve G is G(z)
    x = np.zeros_like(zValues, dtype=float)
    outputG = np.zeros_like(zValues, dtype=float)
    #x values are the log radiance[i] + log exposure time[j]
    for i in range(logexpTime.shape[0]):
        for j in range(LE.shape[0]):
            x[j, i] = LE[j] + logexpTime[i]
            
    # print(solveG)
    # print(x.shape)
    plt.scatter(x, zValues, label='Sample Points')
    intensties = np.arange(0, 256)
    plt.plot(solveG, intensties, label='G(Z)', color='red',linewidth=5)

    plt.title(title)
