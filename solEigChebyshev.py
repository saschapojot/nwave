from consts import *
import numpy as np
import pandas as pd

rowN = 0
N = 32
inFileName = "./gvals/yg" + "row" + str(rowN) + sng + "d" + str(d) + "p" + str(p) +"N"+str(N) + ".csv"
inDat = pd.read_csv(inFileName)

yVals = list(inDat.iloc[:, 0])
gVals = list(inDat.iloc[:, 1])

yVals.insert(0, -1)
gVals.insert(0, np.nan)

#A, B are (2N-1) * (2N-1) matrices
A = np.zeros((2 * N - 1, 2 * N - 1), dtype=float)
B = np.zeros((2 * N - 1, 2 * N - 1), dtype=float)

#TValMat is (2N+1) * (2N+1) matrix
TValMat = np.zeros((2 * N + 1, 2 * N + 1), dtype=float)
for q in range(0, 2 * N + 1):
    for r in range(0, 2 * N + 1):
        TValMat[q, r] = np.cos(q * np.arccos(yVals[r]))

# assemble matrix A

# assemble A_{0q}
for q in range(0,2*N-1):
    #assemble A0qI
    A0qI=sum([(yVals[r]+1)**2*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**2*TValMat[q,2*N]*np.pi/(4*N)
    #assemble A0qIIj
    A0qII=0
    for j in range(1,N):
        A0qIIj=sum([(yVals[r]+1)**2*TValMat[2*j,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
            +(yVals[2*N]+1)**2*TValMat[2*j,2*N]*TValMat[q,2*N]*np.pi/(4*N)
        A0qII+=(N**2-j**2)*A0qIIj
    #assemble A0qIIIj
    A0qIII=0
    for j in range(1,N):
        A0qIIIj=sum([(yVals[r]+1)*TValMat[2*j,r]*TValMat[q,r]for r in range(1,2*N)])*np.pi/(2*N)\
            +(yVals[2*N]+1)*TValMat[2*j,2*N]*TValMat[q,2*N]*np.pi/(4*N)
        A0qIII+=A0qIIIj
    #assemble A0qIV
    A0qIV=sum([(yVals[r]+1)**4*gVals[r]**6*TValMat[q,r]for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**4*gVals[2*N]**6*TValMat[q,2*N]*np.pi/(4*N)
    #assemble A0qV
    A0qV=sum([(yVals[r]+1)**4*gVals[r]**6*TValMat[2*N,r]*TValMat[q,r]for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**4*gVals[2*N]**6*TValMat[2*N,2*N]*TValMat[q,2*N]*np.pi/(4*N)
    A[0,q]=-4*N**3*A0qI-8*N*A0qII-8*N*A0qIII+7/16*A0qIV-7/16*A0qV
    #A0q assembled
    #assemble A1q
    #assemble A1qI
    A1qI=0
    for j in range(0,N-1):
        A1qIj=sum([(yVals[r]+1)**2*TValMat[2*j,r]*TValMat[1,r]for r in range(1,2*N)])*np.pi/(2*N)\
            +(yVals[2*N]+1)**2*TValMat[2*j,2*N]*TValMat[1,2*N]*np.pi/(4*N)
        A1qI+=(4*(N-1)**2+4*(N-1)-4*j**2-4*j)*A1qIj
    #assemble A1qII
    A1qII=sum([(yVals[r]+1)*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)*TValMat[q,2*N]*np.pi/(4*N)
    #assemble A1qIV
    A1qIV=0
    for j in range(1,N):
        A1qIVj=sum([(yVals[r]+1)*TValMat[2*j,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
            +(yVals[2*N]+1)*TValMat[2*j,2*N]*TValMat[q,2*N]*np.pi/(4*N)
        A1qIV+=A1qIVj
    #assemble A1qV
    A1qV=sum([(yVals[r]+1)**4*gVals[r]**6*TValMat[1,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**4*gVals[2*N]**6*TValMat[1,2*N]*TValMat[q,2*N]*np.pi/(4*N)
    #assemble A1qVI
    A1qVI=sum([(yVals[r]+1)**4*gVals[r]**6*TValMat[2*N-1,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**4*gVals[2*N]**6*TValMat[2*N-1,2*N]*TValMat[q,2*N]*np.pi/(4*N)
    A[1,q]=-(2*(N-1)+1)*A1qI+2*A1qII-(4*(N-1)+2)*(A1qII+2*A1qIV)+7/16*A1qV-7/16*A1qVI
    #A1q assembled
    #assemble A2q
    #assemble A2qI
    A2qI=sum([(yVals[r]+1)**2*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**2*TValMat[q,2*N]*np.pi/(4*N)
   #assemble A2qIIj
    A2qII=0
    for j in range(1,N):
       A2qIIj=sum([(yVals[r]+1)**2*TValMat[2*j,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
           +(yVals[2*N]+1)**2*TValMat[2*j,2*N]*TValMat[q,2*N]*np.pi/(4*N)
       A2qII+=(N**2-j**2)*A2qIIj
    #assemble A2qIII
    A2qIII=sum([(yVals[r]+1)*TValMat[1,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)*TValMat[1,2*N]*TValMat[q,2*N]*np.pi/(4*N)
    #assemble A2qIVj
    A2qIV=0
    for j in range(0,N):
        A2qIVj=sum([(yVals[r]+1)*TValMat[2*j+1,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
            +(yVals[2*N]+1)*TValMat[2*j+1,2*N]*TValMat[q,2*N]*np.pi/(4*N)
        A2qIV+=A2qIVj
    #assemble A2qV
    A2qV=sum([(yVals[r]+1)**4*gVals[r]**6*TValMat[2,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**4*gVals[2*N]**6*TValMat[2,2*N]*TValMat[q,2*N]*np.pi/(4*N)
    #assemble A2qVI
    A2qVI=sum([(yVals[r]+1)**4*gVals[r]**6*TValMat[2*N,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +(yVals[2*N]+1)**4*gVals[2*N]**6*TValMat[2*N,2*N]*TValMat[q,2*N]*np.pi/(4*N)
    A[2,q]=4*A2qI-(4*N**3*A2qI+8*N*A2qII)+8*A2qIII-8*N*A2qIV+7/16*A2qV-7/16*A2qVI
    #A2q assembled
print(A[2,:])