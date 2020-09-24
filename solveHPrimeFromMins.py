from consts import *
import numpy as np
import pandas as pd
# this script uses RK4 and binary search to solve h'(0)

inFileName = "./hminVals/min" + sng + "d" + str(d) + "p" + str(p) + ".csv"
stopVal=1e-12
maxIter=100000
def F(h, v):
    rst = (d / p - 2 / p - 1 / p ** 2) * h + (-d + 2 + 2 / p) * v - h ** (2 * p + 1)
    return rst


def RK4(hOut, vOut, stepS):
    M0 = stepS * F(hOut, vOut)
    M1 = stepS * F(hOut + 1 / 2 * stepS * vOut, vOut + 1 / 2 * M0)
    M2 = stepS * F(hOut + 1 / 2 * stepS * vOut + 1 / 4 * stepS * M0, vOut + 1 / 2 * M1)
    M3 = stepS * F(hOut + stepS * vOut + 1 / 2 * stepS * M1, vOut + M2)
    hNext = hOut + stepS * vOut + 1 / 6 * stepS * (M0 + M1 + M2)
    vNext = vOut + 1 / 6 * (M0 + 2 * M1 + 2 * M2 + M3)
    return [hNext, vNext]


def binarySearch(ha,va,stepS):
    '''

    :param ha:
    :param va:
    :param stepS>0
    :return: h'(0) close to given ha,h'a=va
    '''
    if va>0:
        num=0
        if ha>0:
            hNext,vNext=RK4(ha,va,-stepS)
        else:
            hNext,vNext=RK4(ha,va,stepS)
        while np.abs(ha)>stopVal and num<maxIter:
            num+=1
            if ha>0 and hNext>0:
                ha=hNext
                va=vNext
                hNext,vNext=RK4(ha,va,-stepS)
            elif ha<0 and hNext<0:
                ha=hNext
                va=vNext
                hNext,vNext=RK4(ha,va,stepS)
            elif ha>0 and hNext<0:
                ha=hNext
                va=vNext
                stepS/=2
                hNext,vNext=RK4(ha,va,stepS)
            elif ha<0 and hNext>0:
                ha=hNext
                va=vNext
                stepS/=2
                hNext,vNext=RK4(ha,va,-stepS)
        return hNext,vNext
    elif va<0:
        num=0
        if ha>0:
            hNext,vNext=RK4(ha,va,stepS)
        else:
            hNext,vNext=RK4(ha,va,-stepS)
        while np.abs(ha)>stopVal and num<maxIter:
            num+=1
            if ha>0 and hNext>0:
                ha=hNext
                va=vNext
                hNext,vNext=RK4(ha,va,stepS)
            elif ha<0 and hNext<0:
                ha=hNext
                va=vNext
                hNext,vNext=RK4(ha,va,-stepS)
            elif ha>0 and hNext<0:
                ha=hNext
                va=vNext
                stepS/=2
                hNext,vNext=RK4(ha,va,-stepS)
            elif ha<0 and hNext>0:
                ha=hNext
                va=vNext
                stepS/=2
                hNext,vNext=RK4(ha,va,stepS)

        return hNext,vNext

ds=0.1
inDat=pd.read_csv(inFileName)
nRow=len(inDat)

outFileName="./preciseHPrime/hp"+ sng + "d" + str(d) + "p" + str(p) + ".csv"
outDat=pd.DataFrame(columns=["h","v"])
for n in range(0,nRow):
    ha=inDat.iloc[n,0]
    va=inDat.iloc[n,1]
    hNext,vNext=binarySearch(ha,va,ds)
    outDat=outDat.append({"h":hNext,"v":vNext},ignore_index=True)
outDat.to_csv(outFileName,index=False)

