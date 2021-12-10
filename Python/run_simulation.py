from numpy.lib.index_tricks import IndexExpression, MGridClass
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
from scipy import special

from playsound import playsound

def run_kwantisatie():
    obj = Kwantisatie(0)

    
    print('Generating plot: optimal SQR in function of alpha..')
    alpha = np.array([i for i in range(2,9)])
    y = np.array([obj.bepaal_optimale_lineaire_kwantisator(2**i)[2] for i in range(2,9)])
    y2 = np.array([obj.bepaal_compansie_kwantisator(2**i)[1] for i in range(2,9)])
    y3 = np.array([obj.bepaal_Lloyd_Max_kwantisator(2**i)[1] for i in range(2,7)])
    winst = [0 for _ in range(0,6)] 
    for i in range(0,6):
        winst[i] = y[i+1] - y[i]
    print(winst)
    plt.plot(alpha, y)
    plt.plot(alpha, y2)
    plt.plot(alpha[:5], y3)
    plt.xlabel("Alpha")
    plt.ylabel("SQR [dB]")
    plt.savefig('SQR.png')
    plt.close()
    print('Done!')
    

    return 1
    # Lloyd-Max KWANTISATOR

    # opt_kwant = obj.bepaal_Lloyd_Max_kwantisator(2**6)
    # r_opt = opt_kwant[3]
    # q_opt = opt_kwant[4]
    # gekwantiseerd_opt = obj.kwantiseer(r_opt, q_opt)
    
    # return (r_opt,q_opt,gekwantiseerd_opt)

    
def run_broncodering():
    obj = Broncodering()
    
    # Bron -> Macro
    alfabet_scalair = ['5','8']
    stream = '5555555558585885858588888'
    print(stream)
    bronsymbolen = []
    for symbol in stream:
        bronsymbolen.append(symbol)
    bronsymbolen_vast = copy.deepcopy(bronsymbolen)
    rel_freq = [0 for _ in range(len(alfabet_scalair))]
    aantal_symbolen = 0
    while len(bronsymbolen) > 1:
        aantal_symbolen += 1
        index = alfabet_scalair.index(bronsymbolen[0])
        rel_freq[index] += 1
        del bronsymbolen[0]
    
    for index, element in enumerate(rel_freq):
        rel_freq[index] = element / aantal_symbolen
    print(rel_freq)
    index_lijst = [i + 1 for i in range(len(alfabet_scalair))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('dictionary = ', dictionary)
    print('codetabel = ',codetabel)


    print('Huffman_encodeer\n')
    print(bronsymbolen_vast)
    bronsym = [int(sym) for sym in bronsymbolen_vast]
    macrosymbolen = [alfabet_scalair.index(str(sym)) + 1 for sym in bronsym]

    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += str(datapoint)
    print('data_binair = ', data_binair)
    data_binair_lijst = []
    for bit in data_binair_str:
        data_binair_lijst.append(int(bit))
    
    print('Binair -> macro')
    data_macro = obj.Huffman_decodeer(data_binair_lijst, np.array(codetabel), np.array(index_lijst))
    print('data_macro = ', data_macro)

    data_decoded = [alfabet_scalair[sym -1] for sym in data_macro]
    print(data_decoded)
    
    return 1

def run_kanaalcodering():
    return 1

def run_moddet():
    #variables:
    T = 10**(-6)
    Ns = 6
    f0 = 2*10**6
    alpha = 0.5
    Lf = 10
    N0 = 0.37
    sigma = math.sqrt(N0*Ns/2/T)
    hch = 1
    theta = math.pi / 16

    obj = ModDet()
    bitstring = bin(random.randint(0,2**20))[2:].zfill(20)
    bitvector_in = []
    for bit in bitstring:
        bitvector_in.append(int(bit))
    if(len(bitvector_in)%2 == 0):
        slice = len(bitvector_in)
    else:
        slice = -1

    print('TESTING MODDET:')
    print('---------------')
    print("BPSK: ")
    a = obj.mapper(bitvector_in, 'BPSK')
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, sigma, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
    rdown = obj.decimatie(rdemod, Ns, Lf)
    u = obj.maak_decisie_variabele(rdown, hch, theta)
    a_estim = obj.decisie(u, 'BPSK')
    bitvector_out = obj.demapper(a_estim, 'BPSK')
    if bitvector_in == bitvector_out: print('OK') 
    else: print('NOT OK')

    print('4QAM: ')
    a = obj.mapper(bitvector_in, '4QAM')
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, sigma, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
    rdown = obj.decimatie(rdemod, Ns, Lf)
    u = obj.maak_decisie_variabele(rdown, hch, theta)
    a_estim = obj.decisie(u, '4QAM')
    bitvector_out = obj.demapper(a_estim, '4QAM')
    if bitvector_in[:slice] == bitvector_out: print('OK') 
    else: print('NOT OK')

    print('4PAM: ')
    a = obj.mapper(bitvector_in, '4PAM')
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, sigma, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
    rdown = obj.decimatie(rdemod, Ns, Lf)
    u = obj.maak_decisie_variabele(rdown, hch, theta)
    a_estim = obj.decisie(u, '4PAM')
    bitvector_out = obj.demapper(a_estim, '4PAM')
    if bitvector_in[:slice] == bitvector_out: print('OK') 
    else: print('NOT OK')

    print('4PSK: ')
    a = obj.mapper(bitvector_in, '4PSK')
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, sigma, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
    rdown = obj.decimatie(rdemod, Ns, Lf)
    u = obj.maak_decisie_variabele(rdown, hch, theta)
    a_estim = obj.decisie(u, '4PSK')
    bitvector_out = obj.demapper(a_estim, '4PSK')
    if bitvector_in[:slice] == bitvector_out: print('OK') 
    else: print('NOT OK')


    return 

warnings.simplefilter('ignore') # ignore warnings of integral


run_kwantisatie()
#run_broncodering()
#run_kanaalcodering()
#run_moddet()
