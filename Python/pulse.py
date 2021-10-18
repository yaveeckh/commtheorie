
import numpy as np
import math

def pulse(t,T,alpha):
    een = (1-alpha)*np.sinc(t*(1-alpha)/T)
    twee = (alpha)*np.cos(math.pi*(t/T-0.25))*np.sinc(alpha*t/T-0.25)
    drie = (alpha)*np.cos(math.pi*(t/T+0.25))*np.sinc(alpha*t/T+0.25)
    y = 1/np.sqrt(T)*(een+twee+drie);
    return y
