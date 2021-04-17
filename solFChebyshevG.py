from consts import *
import numpy as np
import pandas as pd


inFileName = "./preciseHPrime/hp" + sng + "d" + str(d) + "p" + str(p) + ".csv"
inDat = pd.read_csv(inFileName)
rowN = 0
hp0Val = inDat.iloc[rowN, 1]
alpha = 1 / 2 * (p * (d - 1) - 2)
xExpon = 2 * alpha - 2
N = 128

yVals = [-np.cos(np.pi * j / (2 * N)) for j in range(2 * N, -1, -1)]
xVals = [yElem / 2 + 1 / 2 for yElem in yVals]
#delete x=0
del xVals[-1]



def P(x, f, m):
    if d > 3:
        rst = 1 / (4 * x ** 2) * (d - 1) * (d - 3) * f - 2 / x * m - x ** xExpon * f ** (2 * p + 1)
        return rst
    else:
        rst = -2 / x * m - x ** xExpon * f ** (2 * p + 1)
        return rst


def RK4(xa, xNext, fa, ma):
    '''
    One step RK4 to solve f
    :param a: current point number, 2N, 2N-1,...,2
    :return:
    '''
    dx = xNext - xa
    Q0 = dx * P(xa, fa, ma)
    Q1 = dx * P(xa + 1 / 2 * dx, fa + 1 / 2 * dx * ma, ma + 1 / 2 * Q0)
    Q2 = dx * P(xa + 1 / 2 * dx, fa + 1 / 2 * dx * ma + 1 / 4 * dx * Q0, ma + 1 / 2 * Q1)
    Q3 = dx * P(xa + dx, fa + dx * ma + 1 / 2 * dx * Q1, ma + Q2)
    fNext = fa + dx * ma + dx / 6 * (Q0 + Q1 + Q2)
    mNext = ma + 1 / 6 * (Q0 + 2 * Q1 + 2 * Q2 + Q3)
    return [fNext, mNext]

def RK4Sec(xSecStart,xSecEnd,fa,ma):
    '''
    execute single step RK4 on [xSecStart, xSecEnd]
    :param xSecStart:
    :param xSecEnd:
    :param fa:
    :param ma:
    :return:
    '''
    nSecTmp=int(np.ceil(np.abs(xSecStart-xSecEnd)/gridPointNum))
    dxTmp=(xSecStart-xSecEnd)/nSecTmp

    #init x
    xa=xSecStart
    for j in range(0,nSecTmp):
        #j=0,1,...,nSecTmp-1
        xNext=xa+dxTmp
        [fa,ma]=RK4(xa,xNext,fa,ma)
        xa=xNext
    return [fa,ma]


#total number of x points
xfmVals=[]
xfmVals.append([xVals[0],0,-hp0Val])
xTotLen=len(xVals)
for j in range(0,xTotLen-1):
    xCurr=xVals[j]
    xNext=xVals[j+1]
    fCurr=xfmVals[j][1]
    mCurr=xfmVals[j][2]
    [fCurr,mCurr]=RK4(xCurr,xNext,fCurr,mCurr)
    xfmVals.append([xNext,fCurr,mCurr])

fVals=[elem[1] for elem in xfmVals]
outFileName="./gvals/yg"+"row"+str(rowN)+sng + "d" + str(d) + "p" + str(p) +"N"+str(N)+ ".csv"
#outFigName="./gFigs/fig"+"row"+str(rowN)+sng + "d" + str(d) + "p" + str(p) + ".png"

outDat=pd.DataFrame(columns=["y","g"])
for j in range(len(xfmVals)-1,-1,-1):
    xTmp=xfmVals[j][0]
    fTmp=xfmVals[j][1]
    yTmp=2*xTmp-1
    outDat=outDat.append({"y":yTmp,"g":fTmp},ignore_index=True)

outDat.to_csv(outFileName,index=False)