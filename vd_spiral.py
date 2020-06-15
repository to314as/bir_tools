import numpy as np

def makeSpiral(N,nRounds,PowCoeff,phiBase,m):
#Simple VD Spiral

    if(nRounds<4):
        phiBase=0;

    t=np.arange(0,1,1/N);
    r=t**PowCoeff;
    phi=phiBase+2*np.pi*t*nRounds;

    C=r*np.exp(1j*phi)*m;

    return np.array([np.real(C),np.imag(C)])