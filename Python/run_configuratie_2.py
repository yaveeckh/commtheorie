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
    obj_2 = Kwantisatie()
    
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
        data_encoded_lijst.append(int(bit))

    bitlist_kanaal, M = run_kanaalcodering(data_encoded_lijst)
    data_decoded = obj.vaste_lengte_decodeer(bitlist_kanaal, q)
    obj_2.save_and_play_music(np.array(data_decoded), "Configuratie_2.wav", 0)

    GKA = 0
    for i in range(len(data_decoded)):
        GKA += (bronsymbolen[i] - data_decoded[i])**2
    GKA /= len(data_decoded)
    return M/len(bronsymbolen), GKA

def run_kanaalcodering(bitlist): 
    obj = Kanaalcodering()
    bitlist_grouped = np.reshape(bitlist, (len(bitlist)//10, 10))
    
    print("Kanaal codering")
    bits_encoded = obj.kanaalencodering_1(bitlist_grouped)
    
    bitlist_moddet = run_moddet(bits_encoded.flatten())

    bitlist_moddet_grouped = np.reshape(bitlist_moddet, (len(bitlist_moddet)//14, 14))
    
    print("kanaal decodering")
    bits_decoded = obj.kanaaldecodering_1(bitlist_moddet_grouped)[0]

    return (bits_decoded.flatten(), len(bits_encoded.flatten()))


def run_moddet(bitlist):
    obj = ModDet()

    #variables:
    constellatie = 'BPSK'
    T = 10**(-6)
    Ns = 6
    f0 = 2*10**6
    alpha = 0.5
    Lf = 10
    N0 = 0.37
    sigma = math.sqrt(N0*Ns/2/T)
    hch = 1
    theta = math.pi / 16

    bitarray_out = obj.modulation_detection(bitlist, constellatie, T, Ns, f0, alpha, Lf, N0, hch, theta)

    return bitarray_out

warnings.simplefilter('ignore') # ignore warnings of integral


print(run_broncodering())

