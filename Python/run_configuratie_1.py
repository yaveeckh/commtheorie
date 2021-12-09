from numpy.lib.index_tricks import IndexExpression
from kwantisatie import Kwantisatie
from broncodering import Broncodering
from kanaalcodering import Kanaalcodering
from moddet import ModDet
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy
import time
import random
import math
from scipy.fft import fft, fftfreq

from playsound import playsound

def run_kwantisatie():
    obj = Kwantisatie(0)

    # Lloyd-Max KWANTISATOR

    opt_kwant = obj.bepaal_Lloyd_Max_kwantisator(2**6)
    r_opt = opt_kwant[3]
    q_opt = opt_kwant[4]
    gekwantiseerd_opt = obj.kwantiseer(r_opt, q_opt)
    
    return (r_opt,q_opt,gekwantiseerd_opt)

    
def run_broncodering():
    obj = Broncodering()
    
    print('Kwantisatie\n')
    r, q, bronsymbolen = run_kwantisatie()
    r = r.tolist()
    q = q.tolist()

    print('Vaste-lengte\n')
    data_encoded = obj.vaste_lengte_encodeer(bronsymbolen, q)
    data_encoded_str = ''
    for bitstring in data_encoded:
        for bit in bitstring:
            data_encoded_str += bit

    data_encoded_lijst = []
    for bit in data_encoded_str:
        data_encoded_lijst.append(bit)

    return data_encoded_lijst

def run_kanaalcodering():
    return 1

def run_moddet():
    return 1

warnings.simplefilter('ignore') # ignore warnings of integral


#run_kwantisatie()
#run_broncodering()
#run_kanaalcodering()
#run_moddet()

