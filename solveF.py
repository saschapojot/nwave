from consts import *

import pandas as pd
import matplotlib.pyplot as plt

inFileName = "./preciseHPrime/hp" + sng + "d" + str(d) + "p" + str(p) + ".csv"
inDat = pd.read_csv(inFileName)
rowN = 2
hp0Val = inDat.iloc[rowN, 1]
alpha = 1 / 2 * (p * (d - 1) - 2)
xExpon = 2 * alpha - 2
N = 10000
dx = 1 / N


def P(x, f, m):
    if d > 3:
        rst = 1 / (4 * x ** 2) * (d - 1) * (d - 3) * f - 2 / x * m - x ** xExpon * f ** (2 * p + 1)
        return rst
    else:
        rst = -2 / x * m - x ** xExpon * f ** (2 * p + 1)
        return rst


def RK4(xa, fa, ma):
    '''
    One step RK4 to solve f
    :param xa:
    :param fa:
    :param ma:
    :return:
    '''
    Q0 = -dx * P(xa, fa, ma)
    Q1 = -dx * P(xa - 1 / 2 * dx, fa - 1 / 2 * dx * ma, ma + 1 / 2 * Q0)
    Q2 = -dx * P(xa - 1 / 2 * dx, fa - 1 / 2 * dx * ma - 1 / 4 * dx * Q0, ma + 1 / 2 * Q1)
    Q3 = -dx * P(xa - dx, fa - dx * ma - 1 / 2 * dx * Q1, ma + Q2)
    xNext = xa - dx
    fNext = fa - dx * ma - dx / 6 * (Q0 + Q1 + Q2)
    mNext = ma + 1 / 6 * (Q0 + 2 * Q1 + 2 * Q2 + Q3)
    return [xNext, fNext, mNext]

xfmVals=[]
xN=1
fN=0
mN=-hp0Val
xfmVals.append([xN,fN,mN])
for a in range(N,1,-1):
    xa,fa,ma=xfmVals[-1]
    xNext,fNext,mNext=RK4(xa,fa,ma)
    xfmVals.append([xNext,fNext,mNext])

itemNum=len(xfmVals)
outDat=pd.DataFrame(columns=["x","f","m"])
for n in range(itemNum-1,-1,-1):
    xa,fa,ma=xfmVals[n]
    outDat=outDat.append({"x":xa,"f":fa,"m":ma},ignore_index=True)


outFileName="./fvals/fm"+"row"+str(rowN)+sng + "d" + str(d) + "p" + str(p) + ".csv"
outFigName="./fFigs/fig"+"row"+str(rowN)+sng + "d" + str(d) + "p" + str(p) + ".png"
outDat.to_csv(outFileName,index=False)
plt.figure()
plt.plot(outDat.iloc[:,0],outDat.iloc[:,1])
plt.xscale("log")
plt.ylabel("$f$")
plt.title("$d=$"+str(d)+", $p=$"+str(p)+r", $\eta=$"+str(eta)+", row = "+str(rowN))
plt.savefig(outFigName)
plt.close()