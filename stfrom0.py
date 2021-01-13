from consts import *
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
#ca ne marche pas!!!!!!!!!!!!!!!!!!!
row=0
inFFileName="./fvals/fmrow"+str(row)+sng+"d"+str(d)+"p"+str(p)+".csv"

iFMat=pd.read_csv(inFFileName)
startRowNum=410
restRowNum=len(iFMat)-startRowNum

if restRowNum%2==0:
    startRowNum+=1


subIFMat=iFMat.iloc[startRowNum:,:]

xVals=np.array(subIFMat.iloc[:,0])
fVals=np.array(subIFMat.iloc[:,1])
nPointTot=len(xVals)


def F(a,va,ma,lmd):
    xa=xVals[a]
    fa=fVals[a]
    return -(2*xa+2*lmd)/xa**2*ma-7*xa**2*fa**6*va


#a=0,2,...,nPointTot-3
def RK4(a,va,ma,lmd):
    stepAta=xVals[a+2]-xVals[a]
    K0=ma
    M0=stepAta*F(a,va,ma,lmd)

    K1=stepAta*(ma+1/2*M0)
    M1=stepAta*F(a+1,va+1/2*K0,ma+1/2*M0,lmd)

    K2=stepAta*(ma+1/2*M1)
    M2=stepAta*F(a+1,va+1/2*K1,ma+1/2*M1,lmd)

    K3=stepAta*(ma+M2)
    M3=stepAta*F(a+2,va+K2,ma+M2,lmd)

    vaNext=va+1/6*(K0+2*K1+2*K2+K3)
    maNext=ma+1/6*(M0+2*M1+2*M2+M3)

    return [vaNext,maNext]


def shootFrom0(vars):
    lmdReEst=vars[0]
    lmdImEst=vars[1]
    lmdEst=lmdReEst+1j*lmdImEst

    x0=xVals[0]
    v0Est=np.exp(2*lmdEst/x0)
    m0Est=v0Est*(-2*lmdEst/x0**2)

    solAll=[]
    solAll.append([v0Est,m0Est])
    for a in range(0,nPointTot-2,2):
        vCurr,mCurr=solAll[-1]
        vNext,mNext=RK4(a,vCurr,mCurr,lmdEst)
        solAll.append([vNext,mNext])

    vmLast=solAll[-1]
    vLast=vmLast[0]

    vLastRe=np.real(vLast)
    vLastIm=np.imag(vLast)

    return [vLastRe,vLastIm]


lmdReEst=-0.7
lmdImEst=0.7
inVars=[lmdReEst,lmdImEst]
lmdRe, lmdIm=fsolve(shootFrom0,(lmdReEst,lmdImEst))

print(lmdRe)
print(lmdIm)


