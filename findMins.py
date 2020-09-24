from consts import *
import pandas as pd
import numpy as np
#this script finds minimums of |h| to solve h'(0)

inDatFileName="./hvals/hv"+sng+"d"+str(d)+"p"+str(p)+".csv"
inDatVals=pd.read_csv(inDatFileName)

hAbs=np.abs(inDatVals.iloc[:,0])

def findMins(vec):
    '''

    :param vec: abs of an array
    :return: min pos of elems in vec, except front and back
    '''
    elemNum=len(vec)
    diffVec=[]
    for n in range(1,elemNum):
        diffVec.append(vec[n]-vec[n-1])
    outPos=[]
    for n in range(1,len(diffVec)):
        if diffVec[n-1]<0 and diffVec[n]>=0:
            outPos.append(n)
    return outPos

outFileName="./hminVals/min"+sng+"d"+str(d)+"p"+str(p)+".csv"
hMinInd=findMins(hAbs)
outDat=pd.DataFrame(columns=["h","v"])
for ind in hMinInd:
    hVal=inDatVals.iloc[ind,0]
    vVal=inDatVals.iloc[ind,1]
    outDat=outDat.append({"h":hVal,"v":vVal},ignore_index=True)
outDat.to_csv(outFileName,index=False)
