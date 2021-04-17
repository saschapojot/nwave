from consts import *
import numpy as np
import pandas as pd
import scipy.linalg as slin
rowN = 0
N = 128
inFileName = "./gvals/yg" + "row" + str(rowN) + sng + "d" + str(d) + "p" + str(p) + "N" + str(N) + ".csv"
inDat = pd.read_csv(inFileName)

yVals = list(inDat.iloc[:, 0])
gVals = list(inDat.iloc[:, 1])

yVals.insert(0, -1)
gVals.insert(0, np.nan)

# A, B are (2N-1) * (2N-1) matrices
A = np.zeros((2 * N - 1, 2 * N - 1), dtype=float)
B = np.zeros((2 * N - 1, 2 * N - 1), dtype=float)

# TValMat is (2N+1) * (2N+1) matrix
TValMat = np.zeros((2 * N + 1, 2 * N + 1), dtype=float)
for q in range(0, 2 * N + 1):
    for r in range(0, 2 * N + 1):
        TValMat[q, r] = np.cos(q * np.arccos(yVals[r]))

# assemble matrix A

# assemble A_{0q}
for q in range(0, 2 * N - 1):
    # assemble A0qI
    A0qI = sum([(yVals[r] + 1) ** 2 * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
           + (yVals[2 * N] + 1) ** 2 * TValMat[q, 2 * N] * np.pi / (4 * N)
    # assemble A0qIIj
    A0qII = 0
    for j in range(1, N):
        A0qIIj = sum([(yVals[r] + 1) ** 2 * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                    2 * N) \
                 + (yVals[2 * N] + 1) ** 2 * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
        A0qII += (N ** 2 - j ** 2) * A0qIIj
    # assemble A0qIIIj
    A0qIII = 0
    for j in range(1, N):
        A0qIIIj = sum([(yVals[r] + 1) * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
                  + (yVals[2 * N] + 1) * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
        A0qIII += A0qIIIj
    # assemble A0qIV
    A0qIV = sum([(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
            + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[q, 2 * N] * np.pi / (4 * N)
    # assemble A0qV
    A0qV = sum(
        [(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2 * N, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                       2 * N) \
           + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2 * N, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
    A[0, q] = -4 * N ** 3 * A0qI - 8 * N * A0qII - 8 * N * A0qIII + 7 / 16 * A0qIV - 7 / 16 * A0qV
    # A0q assembled
    # assemble A1q
    # assemble A1qI
    A1qI = 0
    for j in range(0, N - 1):
        A1qIj = sum([(yVals[r] + 1) ** 2 * TValMat[2 * j, r] * TValMat[1, r] for r in range(1, 2 * N)]) * np.pi / (
                    2 * N) \
                + (yVals[2 * N] + 1) ** 2 * TValMat[2 * j, 2 * N] * TValMat[1, 2 * N] * np.pi / (4 * N)
        A1qI += (4 * (N - 1) ** 2 + 4 * (N - 1) - 4 * j ** 2 - 4 * j) * A1qIj
    # assemble A1qII
    A1qII = sum([(yVals[r] + 1) * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
            + (yVals[2 * N] + 1) * TValMat[q, 2 * N] * np.pi / (4 * N)
    # assemble A1qIV
    A1qIV = 0
    for j in range(1, N):
        A1qIVj = sum([(yVals[r] + 1) * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
                 + (yVals[2 * N] + 1) * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
        A1qIV += A1qIVj
    # assemble A1qV
    A1qV = sum(
        [(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[1, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                       2 * N) \
           + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[1, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
    # assemble A1qVI
    A1qVI = sum([(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2 * N - 1, r] * TValMat[q, r] for r in
                 range(1, 2 * N)]) * np.pi / (2 * N) \
            + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2 * N - 1, 2 * N] * TValMat[q, 2 * N] * np.pi / (
                        4 * N)
    A[1, q] = -(2 * (N - 1) + 1) * A1qI + 2 * A1qII - (4 * (N - 1) + 2) * (
                A1qII + 2 * A1qIV) + 7 / 16 * A1qV - 7 / 16 * A1qVI
    # A1q assembled
    # assemble A2q
    # assemble A2qI
    A2qI = sum([(yVals[r] + 1) ** 2 * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
           + (yVals[2 * N] + 1) ** 2 * TValMat[q, 2 * N] * np.pi / (4 * N)
    # assemble A2qIIj
    A2qII = 0
    for j in range(1, N):
        A2qIIj = sum([(yVals[r] + 1) ** 2 * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                    2 * N) \
                 + (yVals[2 * N] + 1) ** 2 * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
        A2qII += (N ** 2 - j ** 2) * A2qIIj
    # assemble A2qIII
    A2qIII = sum([(yVals[r] + 1) * TValMat[1, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
             + (yVals[2 * N] + 1) * TValMat[1, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
    # assemble A2qIVj
    A2qIV = 0
    for j in range(0, N):
        A2qIVj = sum([(yVals[r] + 1) * TValMat[2 * j + 1, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                    2 * N) \
                 + (yVals[2 * N] + 1) * TValMat[2 * j + 1, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
        A2qIV += A2qIVj
    # assemble A2qV
    A2qV = sum(
        [(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                       2 * N) \
           + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
    # assemble A2qVI
    A2qVI = sum(
        [(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2 * N, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                        2 * N) \
            + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2 * N, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
    A[2, q] = 4 * A2qI - (
                4 * N ** 3 * A2qI + 8 * N * A2qII) + 8 * A2qIII - 8 * N * A2qIV + 7 / 16 * A2qV - 7 / 16 * A2qVI
    # A2q assembled
    # assemble A2kq, k=2,3,...,N-1

    for k in range(2, N):
        # assemble A2kqI
        A2kqI = sum([(yVals[r] + 1) ** 2 * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
                + (yVals[2 * N] + 1) ** 2 * TValMat[q, 2 * N] * np.pi / (4 * N)
        # assemble A2kqIIj
        A2kqII = 0
        for j in range(1, k - 1):
            A2kqIIj = sum(
                [(yVals[r] + 1) ** 2 * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
                      + (yVals[2 * N] + 1) ** 2 * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
            A2kqII += (k ** 2 - j ** 2) * A2kqIIj
        # assemble A2kqIIIj
        A2kqIII = 0
        for j in range(1, N):
            A2kqIIIj = sum(
                [(yVals[r] + 1) ** 2 * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
                       + (yVals[2 * N] + 1) ** 2 * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
            A2kqIII += (N ** 2 - j ** 2) * A2kqIIIj
        # assemble A2kqIVj
        A2kqIV = 0
        for j in range(1, k):
            A2kqIVj = sum([(yVals[r] + 1) * TValMat[2 * j + 1, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                        2 * N) \
                      + (yVals[2 * N] + 1) * TValMat[2 * j + 1, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
            A2kqIV += A2kqIVj
        # assemble A2kqVj
        A2kqV = 0
        for j in range(0, N):
            A2kqVj = sum([(yVals[r] + 1) * TValMat[2 * j + 1, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                        2 * N) \
                     + (yVals[2 * N] + 1) * TValMat[2 * j + 1, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
            A2kqV += A2kqVj
        # assemble A2kqVI
        A2kqVI = sum([(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2 * k, r] * TValMat[q, r] for r in
                      range(1, 2 * N)]) * np.pi / (2 * N) \
                 + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2 * k, 2 * N] * TValMat[q, 2 * N] * np.pi / (
                             4 * N)
        # assemble A2kqVII
        A2kqVII = sum([(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2 * N, r] * TValMat[q, r] for r in
                       range(1, 2 * N)]) * np.pi / (2 * N) \
                  + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2 * N, 2 * N] * TValMat[q, 2 * N] * np.pi / (
                              4 * N)
        A[2 * k, q] = 4 * (
                    k ** 3 - N ** 3) * A2kqI + 8 * k * A2kqII - 8 * N * A2kqIII + 8 * k * A2kqIV - 8 * N * A2kqV + 7 / 16 * A2kqVI - 7 / 16 * A2kqVII
    # A2kq assembled
    # assemble A2kp1q, k=1,2,...,N-2
    for k in range(1, N - 1):
        # assemble A2kp1qIj
        A2kp1qI = 0
        for j in range(0, k):
            A2kp1qIj = sum(
                [(yVals[r] + 1) ** 2 * TValMat[2 * j + 1, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                                   2 * N) \
                       + (yVals[2 * N] + 1) ** 2 * TValMat[2 * j + 1, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
            A2kp1qI += (4 * k ** 2 + 4 * k - 4 * j ** 2 - 4 * j) * A2kp1qIj
        # assemble A2kp1qIIj
        A2kp1qII = 0
        for j in range(0, N - 1):
            A2kp1qIIj = sum(
                [(yVals[r] + 1) ** 2 * TValMat[2 * j + 1, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                                    2 * N) \
                        + (yVals[2 * N] + 1) ** 2 * TValMat[2 * j + 1, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
            A2kp1qII += (4 * (N - 1) ** 2 + 4 * (N - 1) - 4 * j ** 2 - 4 * j) * A2kp1qIIj
        # assemble A2kp1qIII
        A2kp1qIII = sum([(yVals[r] + 1) * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
                    + (yVals[2 * N] + 1) * TValMat[q, 2 * N] * np.pi / (4 * N)
        # assemble A2kp1qIVj
        A2kp1qIV = 0
        for j in range(1, k + 1):
            A2kp1qIVj = sum([(yVals[r] + 1) * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                        2 * N) \
                        + (yVals[2 * N] + 1) * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
            A2kp1qIV += A2kp1qIVj
        # assemble A2kp1qV
        A2kp1qV = sum([(yVals[r] + 1) * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (2 * N) \
                  + (yVals[2 * N] + 1) * TValMat[q, 2 * N] * np.pi / (4 * N)
        # assemble A2kp1qVIj
        A2kp1qVI = 0
        for j in range(1, N):
            A2kp1qVIj = sum([(yVals[r] + 1) * TValMat[2 * j, r] * TValMat[q, r] for r in range(1, 2 * N)]) * np.pi / (
                        2 * N) \
                        + (yVals[2 * N] + 1) * TValMat[2 * j, 2 * N] * TValMat[q, 2 * N] * np.pi / (4 * N)
        # assemble A2kp1qVII
        A2kp1qVII = sum([(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2 * k + 1, r] * TValMat[q, r] for r in
                         range(1, 2 * N)]) * np.pi / (2 * N) \
                    + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2 * k + 1, 2 * N] * TValMat[
                        q, 2 * N] * np.pi / (4 * N)
        # assemble A2kp1qVIII
        A2kp1qVIII = sum([(yVals[r] + 1) ** 4 * gVals[r] ** 6 * TValMat[2 * N - 1, r] * TValMat[q, r] for r in
                          range(1, 2 * N)]) * np.pi / (2 * N) \
                     + (yVals[2 * N] + 1) ** 4 * gVals[2 * N] ** 6 * TValMat[2 * N - 1, 2 * N] * TValMat[
                         q, 2 * N] * np.pi / (4 * N)
        A[2 * k + 1, q] = (2 * k + 1) * A2kp1qI - (2 * (N - 1) + 1) * A2kp1qII + (4 * k + 2) * (
                    A2kp1qIII + 2 * A2kp1qIV) - (4 * (N - 1) + 2) * (A2kp1qV + 2 * A2kp1qVI) \
                          + 7 / 16 * A2kp1qVII - 7 / 16 * A2kp1qVIII

#     # A assembled

#assemble matrix B
for q in range(0,2*N-1):
    #assemble B0qIj
    B0qI=0
    for j in range(0,N):
        B0qIj=TValMat[2*j+1,0]*TValMat[q,0]*np.pi/(4*N)+sum(TValMat[2*j+1,r]*TValMat[q,r] for r in range(1,2*N))*np.pi/(2*N)\
            +TValMat[2*j+1,2*N]*TValMat[q,2*N]*np.pi/(4*N)
        B0qI+=B0qIj
    B[0,q]=16*N*B0qI
    #B0q assembled
    #assemble B1q
    B1qI=TValMat[q,0]*np.pi/(4*N)+sum([TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
        +TValMat[q,2*N]*np.pi/(4*N)
    B1qII=0
    for j in range(1,N):
        B1qIIj=TValMat[2*j,0]*TValMat[q,0]*np.pi/(4*N)+sum([TValMat[2*j,r]*TValMat[q,r]for r in range(1,2*N)])*np.pi/(2*N)\
            +TValMat[2*j,2*N]*TValMat[q,2*N]*np.pi/(4*N)
        B1qII+=B1qIIj
    B[1,q]=-4*B1qI+(16*(N-1)+8)*(1/2*B1qI+B1qII)
    #B1q assembled

    #assemble B2kq, k=1,2,...,N-1
    for k in range(1,N):
        B2kqI=0
        B2kqII=0
        #assemble B2kqIj
        for j in range(0,k):
            B2kqIj=TValMat[2*j+1,0]*TValMat[q,0]*np.pi/(4*N)+sum([TValMat[2*j+1,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
                +TValMat[2*j+1,2*N]*TValMat[q,2*N]*np.pi/(4*N)
            B2kqI+=B2kqIj
        for j in range(0,N):
            B2kqIIj=TValMat[2*j+1,0]*TValMat[q,0]*np.pi/(4*N)+sum([TValMat[2*j+1,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
                +TValMat[2*j+1,2*N]*TValMat[q,2*N]*np.pi/(4*N)
            B2kqII+=B2kqIIj
        B[2*k,q]=-16*k*B2kqI+16*N*B2kqII
    #B2kq assembled

    #assemble B2k+1q, k=1,2,...,N-2
    for k in range(1,N-1):

        B2kp1qII=0
        B2kp1qIII=0
        #assemble B2kp1qI
        B2kp1qI=TValMat[q,0]*np.pi/(4*N)+sum([TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
            +TValMat[q,2*N]*np.pi/(4*N)
        #assemble B2kp1qIIj
        for j in range(1,k+1):
            B2kp1qIIj=TValMat[2*j,0]*TValMat[q,0]*np.pi/(4*N)+sum([TValMat[2*j,r]*TValMat[q,r] for r in range(1,2*N)])*np.pi/(2*N)\
                +TValMat[2*j,2*N]*TValMat[q,2*N]*np.pi/(4*N)
            B2kp1qII+=B2kp1qIIj
        #assemble B2kp1qIIIj
        for j in range(1,N):
            B2kp1qIIIj=TValMat[2*j,0]*TValMat[q,0]*np.pi/(4*N)+sum([TValMat[2*j,r]*TValMat[q,r]for r in range(1,2*N)])*np.pi/(2*N)\
                +TValMat[2*j,2*N]*TValMat[q,2*N]*np.pi/(4*N)
            B2kp1qIII+=B2kp1qIIIj
        B[2*k+1,q]=-(16*k+8)*(1/2*B2kp1qI+B2kp1qII)+(16*(N-1)+8)*(1/2*B2kp1qI+B2kp1qIII)
    #B assembled


#solve Auhat=lambda B uhat
lAll,vecAll=slin.eig(A,B)
lSorted=sorted(lAll,key=np.abs)
print(lSorted)
dtfr=pd.DataFrame(data=lSorted)



