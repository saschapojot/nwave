from consts import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# this script solves h value
# solves h backward
# starting from
x0 = 10
# grid number
N = 10000

# step length
ds = x0 / N


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
    M0 = -ds * F(ha, va)
    M1 = -ds * F(ha - 1 / 2 * ds * va, va + 1 / 2 * M0)
    M2 = -ds * F(ha - 1 / 2 * ds * va - 1 / 4 * ds * M0, va + 1 / 2 * M1)
    M3 = -ds * F(ha - ds * va - 1 / 2 * ds * M1, va + M2)
    hNext = ha - ds * va - 1 / 6 * ds * (M0 + M1 + M2)
    vNext = va + 1 / 6 * (M0 + 2 * M1 + 2 * M2 + M3)
    return [hNext, vNext]

#array of vecs,[[hVal,vVal]]
hvVals=[]
#append initial value
hvVals.append([eta,-(d*p-2*p-1)/p*eta])
for nTmp in range(0,3*N):
    hvCurr=hvVals[-1]
    hCurr=hvCurr[0]
    vCurr=hvCurr[1]
    hNext,vNext=RK4(hCurr,vCurr)
    hvVals.append([hNext,vNext])

outTabFileName="./hvals/hv"+sng+"d"+str(d)+"p"+str(p)+".csv"
outTabDat=pd.DataFrame(columns=["h","v"])
for rowDat in  hvVals:
    ha=rowDat[0]
    va=rowDat[1]
    outTabDat=outTabDat.append({"h":ha,"v":va},ignore_index=True)

outTabDat.to_csv(outTabFileName,index=False)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax1=plt.gca()
plt.title("$p=$"+str(p)+", $d=$"+str(d)+r", $\eta=$"+str(eta))
plt.plot(outTabDat.iloc[:,0],outTabDat.iloc[:,1])

outFigName="./hvFigs/fig"+sng+"d"+str(d)+"p"+str(p)+".png"
plt.savefig(outFigName)
plt.close()