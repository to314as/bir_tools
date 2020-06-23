import numpy as np
import math


def makeSpiral(N,nRounds,PowCoeff,phiBase,m):
#Simple VD Spiral

    if(nRounds<4):
        phiBase=0;

    t=np.arange(0,1,1/N);
    r=t**PowCoeff;
    phi=phiBase+2*np.pi*t*nRounds;

    C=r*np.exp(1j*phi)*m;

    return np.array([np.real(C),np.imag(C)])

def PointsInCircum(r,n=100):
    return np.array([(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n)])

def makeRadial(N,spikes,s):
#Simple Radial points
    coords=[]
    for i in range(1,N//spikes):
        coords.append(PointsInCircum(s/(N//spikes)*i,n=spikes)+180)
    return np.array(coords).reshape(-1,2)