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

from playsound import playsound

def run_kwantisatie():
    obj = Kwantisatie(0)
    
    ########################
    # LINEAIRE KWANTISATOR #
    ########################

    # Maak een figuur van de optimale SQR voor lineaire kwantisatie in dB 
    # in functie van alpha = [2,...,8], waarbij M = 2**alpha
    """ 
    print('Generating plot: optimal SQR in function of alpha..')
    alpha = np.array([i for i in range(2,9)])
    y = np.array([obj.bepaal_optimale_lineaire_kwantisator(2**i)[2] for i in range(2,9)])
    winst = [0 for _ in range(0,6)] 
    for i in range(0,6):
        winst[i] = y[i+1] - y[i]
    print(winst)
    plt.plot(alpha, y)
    plt.savefig('SQR.png')
    plt.close()
    print('Done!')
    """
    # Plot nu opnieuw de distributie fU (u) waarbij de bekomen 
    # kwantisatiedrempels en reconstructieniveaus duidelijk zijn aangegeven.
    
    # opt_lin_kwant = obj.bepaal_optimale_lineaire_kwantisator(2**6, True)
    # r_opt_lin = opt_lin_kwant[4]
    # q_opt_lin = opt_lin_kwant[5]
    # gekwantiseerd_lin = obj.kwantiseer(r_opt_lin, q_opt_lin)
    

    """
    print('Generating plot: fU(u)')
    plt.figure(figsize=(20,10))
    for i in range(0, 2**6):
        plt.axvline(q_opt_lin[i], 0, 0.1, color = 'k', lw = 0.5)
        plt.axvline(r_opt_lin[i], 0, 0.2, color = 'r', lw = 0.5)
    obj.plot_distributie('fu_uniform.png')
    print('Done!')
    """

    #########################
    # COMPANSIE KWANTISATOR #
    #########################
    
    # compansie_kwant = obj.bepaal_compansie_kwantisator(2**6)
    # r_compansie = compansie_kwant[3]
    # q_compansie = compansie_kwant[4]
    # gekwantiseerd_compansie = obj.kwantiseer(r_compansie, q_compansie)

    """
    print('Generating plot: fU(u)')
    plt.figure(figsize=(20,10))
    for i in range(0, 2**6):
        plt.axvline(q_compansie[i], 0, 0.1, color = 'k', lw = 0.5)
        plt.axvline(r_compansie[i], 0, 0.2, color = 'r', lw = 0.5)
    plt.axvline(r_compansie[2**6], 0, 0.2, color = 'r', lw = 0.5)
    obj.plot_distributie('fu_compansie.png')
    print('Done!')
    """

    
    #########################
    # Lloyd-Max KWANTISATOR #
    #########################

    opt_kwant = obj.bepaal_Lloyd_Max_kwantisator(2**6)
    r_opt = opt_kwant[3]
    q_opt = opt_kwant[4]
    gekwantiseerd_opt = obj.kwantiseer(r_opt, q_opt)

    """
    print('Generating plot: fU(u)')
    plt.figure(figsize=(20,10))
    for i in range(0, 2**6):
        plt.axvline(q_opt[i], 0, 0.1, color = 'k', lw = 0.5)
        plt.axvline(r_opt[i], 0, 0.2, color = 'r', lw = 0.5)
    plt.axvline(r_opt[2**6], 0, 0.2, color = 'r', lw = 0.5)
    obj.plot_distributie('fu_opt.png')
    print('Done!')
    """

    ###########################

    # Sla de gekwantiseerde fragmenten ook op
    #obj.save_and_play_music(obj.kwantiseer(r_opt_lin, q_opt_lin), "uniform.wav", 0)
    #obj.save_and_play_music(obj.kwantiseer(r_compansie, q_compansie), "compansie.wav", 0)
    #obj.save_and_play_music(np.array(obj.kwantiseer(r_opt, q_opt)), "LM.wav", 0)

    return (r_opt,q_opt,gekwantiseerd_opt)

    
def run_broncodering():
    obj = Broncodering()
    
    print('Kwantisatie')
    r, q, bronsymbolen = run_kwantisatie()
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
    print('gem_len = ', gem_len)


    print('Macro -> binair')
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += datapoint

    return data_binair_str


    

def run_kanaalcodering():
    return 1

def run_moddet():
    #variables:
    T = 10**(-6)
    Ns = 6
    f0 = 2*10**6
    alpha = 0.5
    Lf = 10
    N0 = 0.05
    sigma = math.sqrt(N0*Ns/2/T)
    hch = 1
    theta = math.pi / 16

    obj = ModDet()
    bitstring = bin(random.randint(0,2**1111))[2:].zfill(1111)
    bitvector_in = []
    for bit in bitstring:
        bitvector_in.append(int(bit))
    if(len(bitvector_in)%2 == 0):
        slice = 0
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


#run_kwantisatie()
run_broncodering()
#run_kanaalcodering()
#run_moddet()
