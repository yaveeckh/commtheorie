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
    bronsymbolen_vast = copy.deepcopy(bronsymbolen)
    r = r.tolist()
    q = q.tolist()


    print('rel_freq')
    alfabet_scalair = q
    rel_freq = [0 for _ in range(len(alfabet_scalair))]
    aantal_symbolen = 0
    while len(bronsymbolen) > 1:
        aantal_symbolen += 1
        index = alfabet_scalair.index(bronsymbolen[0])
        rel_freq[index] += 1
        del bronsymbolen[0]

    for index, element in enumerate(rel_freq):
        rel_freq[index] = element / aantal_symbolen

    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie, '\n')
    

    print('Codetabel + dictionary')
    index_lijst = [i + 1 for i in range(len(alfabet_scalair))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('gem_len = ', gem_len, '\n')

    macrosymbolen = [alfabet_scalair.index(sym) + 1 for sym in bronsymbolen_vast]
    print('Huffman_encodeer\n')
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += str(datapoint)
    
    
    data_binair_lijst = []
    for bit in data_binair_str:
        data_binair_lijst.append(int(bit))
    
    bitlist_kanaal, M = run_kanaalcodering(data_binair_lijst)

    print('Binair -> macro')
    data_macro = obj.Huffman_decodeer(bitlist_kanaal , np.array(codetabel), np.array(index_lijst))

    print('Macro -> Bron')
    data_decoded = [alfabet_scalair[sym - 1] for sym in data_macro]
    obj_2.save_and_play_music(np.array(data_decoded), "Configuratie_4.wav", 0)

    GKA = 0
    for i in range(len(data_decoded)):
         if i <= len(bronsymbolen_vast):
            GKA += (bronsymbolen_vast[i] - data_decoded[i])**2
    GKA /= len(data_decoded)
    return M/len(bronsymbolen_vast), GKA

def run_kanaalcodering(bitlist): 
    obj = Kanaalcodering()

    bitlist_2 = bitlist[:-(len(bitlist)%10)]
    bitlist_grouped = np.reshape(bitlist_2, (len(bitlist_2)//10, 10))
    
    M = 0

    print("Kanaal codering")
    encoded = obj.kanaalencodering_1(bitlist_grouped)
    bitlist_moddet = run_moddet(encoded.flatten())
    M += len(bitlist_moddet)
    encoded_ch = np.reshape(bitlist_moddet, (len(bitlist_moddet)//14, 14))

    print("Kanaal decodering")
    decoded = obj.kanaaldecodering_1(encoded_ch, True)
    decoded_bits = decoded[0]
    decoded_corrected = copy.deepcopy(decoded[0])
    decoded_fouten = decoded[1]
    
    T = 0
    T_max = 8
    print("Retransmissies")
    if(decoded_fouten != []):
        
        while(T < T_max and len(decoded_fouten) > 0):
            ARQ = False if T == T_max-1 else True 

            retrans_pack = np.array([encoded[i] for i in decoded_fouten])
            r_moddet = run_moddet(retrans_pack.flatten())
            M += len(retrans_pack.flatten())
            retransmitted = np.reshape(r_moddet, (len(r_moddet)//14, 14))

            r_decoded = obj.kanaaldecodering_1(retransmitted, ARQ=True)
            r_bits = r_decoded[0]
            r_fouten = r_decoded[1]
            nieuwe_fouten = [decoded_fouten[i] for i in r_fouten]
                
            if T < T_max-1:
                to_remove = []
                for index, row in enumerate(decoded_fouten):
                    if row not in nieuwe_fouten:
                        decoded_corrected[row] = r_bits[index]
                        to_remove.append(row)      
                for i in to_remove: decoded_fouten.remove(i)
            else:
                for i, row in enumerate(decoded_fouten):
                    decoded_corrected[row] = r_bits[i]
            T += 1
    return (decoded_corrected.flatten(), M)


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

