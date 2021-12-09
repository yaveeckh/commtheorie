from os import R_OK
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

    bitlist_moddet = run_kanaalcodering(data_encoded_lijst)
    data_decoded = obj.vaste_lengte_decodeer(bitlist_moddet, q)

    obj_2.save_and_play_music(np.array(data_decoded), "Configuratie_2.wav", 1)

    return data_decoded

def run_kanaalcodering(bitlist): 
    obj = Kanaalcodering()
    bitlist_grouped = np.reshape(bitlist, (len(bitlist)//10, 10))
    
    print("Kanaal codering")
    encoded = obj.kanaalencodering_1(bitlist_grouped)
    
    bitlist_moddet = run_moddet(encoded.flatten())

    encoded_ch = np.reshape(bitlist_moddet, (len(bitlist_moddet)//14, 14))
    
    #print("kanaal decodering")
    #bits_decoded = obj.kanaaldecodering_1(bitlist_moddet_grouped)[0]


    T = 0
    T_max = 8

    decoded = obj.kanaaldecodering_1(encoded_ch, True)
    decoded_bits = decoded[0]
    decoded_corrected = copy.deepcopy(decoded[0])
    decoded_fouten = decoded[1]
    
    print("Decoded")
    if(decoded_fouten != []):
        
        while(T < T_max and len(decoded_fouten) > 0):
            ARQ = False if T == T_max-1 else True 
            print(f'---Retransmissie {T}---')
            retrans_pack = np.array([encoded[i] for i in decoded_fouten])
            
            r_moddet = run_moddet(retrans_pack.flatten())
            retransmitted = np.reshape(r_moddet, (len(r_moddet)//14, 14))

            d = obj.kanaaldecodering_1(retransmitted, ARQ=True)
            decoded_retransmitted = d[0]
            fouten_retransmitted = d[1]
            nieuwe_fouten = [decoded_fouten[i] for i in fouten_retransmitted]
            # print(f'oude fouten: {decoded_fouten}')
            # print(f'nieuwe fouten: {nieuwe_fouten}')
                
            if T < T_max-1:
                to_remove = []
                for index, row in enumerate(decoded_fouten):
                    if row not in nieuwe_fouten:
                        #print(f'{row}, {decoded_retransmitted[index]}, {bitlist_grouped[row]}')
                        decoded_corrected[row] = decoded_retransmitted[index]
                        to_remove.append(row)
                
                for i in to_remove: decoded_fouten.remove(i)
                
                T += 1
            else:
                #print("Volledige decodering!")

                for i, row in enumerate(decoded_fouten):
                    decoded_corrected[row] = decoded_retransmitted[i]
                T += 1
            print("--------")


    return decoded_corrected.flatten()


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


run_broncodering()

