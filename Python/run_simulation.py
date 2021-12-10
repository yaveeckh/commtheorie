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
    N0 = 0.0005
    sigma = math.sqrt(N0*Ns/2/T)
    hch = 1
    theta = math.pi / 16

    obj = ModDet()
    bitstring = bin(random.randint(0,2**1000))[2:].zfill(1000)
    bitvector_in = []
    for bit in bitstring:
        bitvector_in.append(int(bit))
    if(len(bitvector_in)%2 == 0):
        slice = len(bitvector_in)
    else:
        slice = -1
    
    ######### PLOT |P(f)| ##########
    
    def plot_puls(alhpa, filename):
        t_theorie = np.linspace(-1000*Lf*T, 1000*Lf*T, 2*Lf*Ns*1000 + 1)
        t_vector = np.linspace(-Lf*T, Lf*T, 2*Lf*Ns + 1)
        afgeknot = obj.pulse(t_vector, T, alhpa)
        afgeknot_fourier = fft(afgeknot)*T/Ns
        afgeknot_x = fftfreq(2*Lf*Ns + 1, d=T/Ns)

        theorie = obj.pulse(t_theorie, T, alpha)
        theorie_fourier = fft(theorie)*T/Ns
        theorie_x = fftfreq(2*1000*Lf*Ns +1, d=T/Ns)

        plt.plot(afgeknot_x, np.abs(afgeknot_fourier))
        plt.plot(theorie_x, np.abs(theorie_fourier))
        
        plt.savefig(filename)
        plt.close()
        return


    # plot_puls(0.05)
    # plot_puls(0.5)
    plot_puls(0.95, "moddet/puls/alpha_95.png")
    return
    
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
    u = obj.maak_decisie_variabele(rdown, hch, theta, 1, "moddet/4QAM_scatter.png")
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


#run_kwantisatie()
run_broncodering()
#run_kanaalcodering()
#run_moddet()
