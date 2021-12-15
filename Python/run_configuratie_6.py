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

    print('Bron -> Macro')
    alfabet_scalair = q
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet_scalair)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    

    print('Codetabel + dictionary')
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('gem_len = ', gem_len, '\n')


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
    data_decoded = obj.vector_naar_scalair(data_macro, alfabet_scalair)
    obj_2.save_and_play_music(np.array(data_decoded), "Configuratie_6.wav", 0)


    GKA = 0
    for i in range(len(data_decoded)):
         if i <= len(bronsymbolen_vast):
            GKA += (bronsymbolen_vast[i] - data_decoded[i])**2
    GKA /= len(data_decoded)
    return M/len(bronsymbolen_vast), GKA

def run_kanaalcodering(bitlist):
        
    M = 0
    g_x = [1,1,0,0,1,1,0,1,1]
    bitlist = np.array(bitlist, np.uint8)

    if(len(bitlist) %2 != 0):
        bitlist = bitlist[:-(len(bitlist)%2)]

    obj = Kanaalcodering()
    bitlist = np.reshape(bitlist, (len(bitlist)//2, 2))
    print("encoding")
    encoded = obj.kanaalencodering_2(bitlist, g_x)
    
    print("kanaal")
    #Simuleer kanaal door bits te veranderen
    bitlist_moddet = run_moddet(encoded.flatten())
    M += len(bitlist_moddet)
    print("kanaal done")

    encoded_ch = np.reshape(bitlist_moddet, (len(bitlist_moddet)//14, 14))
    
    print("decoding")
    #Decodeer wat door het kanaal komt
    decoded = obj.kanaaldecodering_2(encoded_ch, g_x)
    decoded_fouten = decoded[1]

    #Kopieer decoded voor een gecorrigeerde array
    decoded_corrected = np.array(copy.deepcopy(decoded[0]), np.uint8)   

    T = 0
    T_max = 5  
    if(decoded_fouten != []):
        
        while(T < T_max and len(decoded_fouten) > 0):

            #Retransmit
            retrans_pack = np.array([encoded[i] for i in decoded_fouten])
            
            r_moddet = run_moddet(retrans_pack.flatten())
            M += len(retrans_pack.flatten())
            retransmitted = np.reshape(r_moddet, (len(r_moddet)//14, 14))

            #Decode
            r_decoded = obj.kanaaldecodering_2(retransmitted, g_x)
            r_bits = r_decoded[0]
            r_fouten = r_decoded[1]

            nieuwe_fouten = [decoded_fouten[i] for i in r_fouten]

            to_remove = []
            for index, row in enumerate(decoded_fouten):
                if row not in nieuwe_fouten:
                    decoded_corrected[row] = r_bits[index]
                    to_remove.append(row)
            
            for i in to_remove: decoded_fouten.remove(i)
            
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