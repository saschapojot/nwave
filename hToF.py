from consts import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
inFileName = "./preciseHPrime/hp" + sng + "d" + str(d) + "p" + str(p) + ".csv"
inDat = pd.read_csv(inFileName)
rowN = 0
v0Val = inDat.iloc[rowN, 1]

x0=10
#max grid number
N=10000
#step length
ds=x0/N

def F(h, v):
    rst = (d / p - 2 / p - 1 / p ** 2) * h + (-d + 2 + 2 / p) * v - h ** (2 * p + 1)
    return rst

def RK4(ha, va):
    '''
    One step 4th order Runge-Kutta
    :param ha:
    :param va:
    :return:
    '''
    M0 = +ds * F(ha, va)
    M1 = +ds * F(ha + 1 / 2 * ds * va, va + 1 / 2 * M0)
    M2 = +ds * F(ha + 1 / 2 * ds * va + 1 / 4 * ds * M0, va + 1 / 2 * M1)
    M3 = +ds * F(ha + ds * va + 1 / 2 * ds * M1, va + M2)
    hNext = ha + ds * va + 1 / 6 * ds * (M0 + M1 + M2)
    vNext = va + 1 / 6 * (M0 + 2 * M1 + 2 * M2 + M3)
    return [hNext, vNext]

hvValsAll=[]
hvValsAll.append([0,v0Val])
#n=0,1,...,N-1
for n in range(0,N):
    hvCurr=hvValsAll[-1]
    hCurr=hvCurr[0]
    vCurr=hvCurr[1]
    hNext, vNext=RK4(hCurr,vCurr)
    hvValsAll.append([hNext,vNext])


omega=(d-1)/2-1/p
sVals=[ds*a for a in range(0,N+1)]
xVals=[np.exp(-s) for s in sVals]
hVals=[elem[0] for elem in hvValsAll]
fVals=[np.exp(sVals[a]*omega)*hVals[a] for a in range(0,len(sVals)) ]
plt.figure()
plt.plot(xVals,fVals)
plt.xscale("log")
plt.show()
plt.close()