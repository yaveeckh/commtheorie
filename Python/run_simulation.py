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
    print('LINEAIR QUANTISATION')
    print('-----------------------')

    # Maak een figuur van de optimale SQR voor lineaire kwantisatie in dB 
    # in functie van α = [2,...,8], waarbij M = 2**α
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
    
    opt_lin_kwant = obj.bepaal_optimale_lineaire_kwantisator(2**6, True)
    r_opt_lin = opt_lin_kwant[4]
    q_opt_lin = opt_lin_kwant[5]
    

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
    print('COMPANSION QUANTISATION')
    print('-----------------------')
    
    compansie_kwant = obj.bepaal_compansie_kwantisator(2**6)
    r_compansie = compansie_kwant[3]
    q_compansie = compansie_kwant[4]
    gekwantiseerd = obj.kwantiseer(r_compansie, q_compansie)

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


    ###########################

    # Sla de gekwantiseerde fragmenten ook op: ’uniform.wav’, ’LM.wav’ en ’compansie.wav’
    #bj.save_and_play_music(obj.kwantiseer(r_opt_lin, q_opt_lin), "uniform.wav", 0)
    #obj.save_and_play_music(obj.kwantiseer(r_compansie, q_compansie), "compansie.wav", 0)

    return (r_compansie,q_compansie,gekwantiseerd)
    
def run_broncodering():
    obj = Broncodering()

    """
    # TEST

    # Bron -> Macro
    # ['1', '1', '1', '2', '1', '3',...] -> ['11', '12', '13',...] -> ['0', '1', '2',...]
    alfabet = ['1','2']
    stream = '11122122122121222222'
    bronsymbolen = []
    for symbol in stream:
        bronsymbolen.append(symbol)
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet)
    print('macrosymbolen = ', macrosymbolen)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    #print('entropie = ', entropie)
    
    # Codetabel
    # ['0', '1', '2',...] -> [11001, 1111, 000]
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('dictionary = ', dictionary)
    #print('gem_len = ', gem_len)
    print('codetabel = ',codetabel)

    # Macro -> binair : KLOPT
    #data_binair = [obj.Huffman_encodeer(macrosymbolen[i], dictionary) for i in range(len(macrosymbolen))]
    #data_binair = [x[0] for x in data_binair]
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    print('data_binair = ', data_binair)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += datapoint
    # Binair -> macro : KLOPT NIET
    #data_decoded = [obj.Huffman_decodeer([int(char) for char in data_binair[i]], np.array(codetabel), np.array(index_lijst)) for i in range(len(data_binair))]
    #data_decoded = [x[0] for x in data_decoded if x != []]
    data_decoded = obj.Huffman_decodeer(data_binair_str, np.array(codetabel), np.array(index_lijst))
    print('data_decoded = ', data_decoded, '\n')


    # Macro -> Bron
    # ['0', '1', '2',...] -> ['1', '1', '1', '2', '1', '3',...]
    bronsymbolen_nadien = obj.vector_naar_scalair(data_decoded, alfabet)
    print('bronsymbolen = ', bronsymbolen_nadien)

    # Vaste-lengte
    encoded = obj.vaste_lengte_encodeer(stream, alfabet)
    print('bronsymbolen_encoded = ', encoded)
    decoded = obj.vaste_lengte_decodeer(encoded, alfabet)
    print('bronsymbolen_decoded = ', decoded)
    """

    #############################################################################################
    
    start = time.time()
    # op kwantisatie - compansie
    r_compansie, q_compansie, bronsymbolen = run_kwantisatie()
    #print('bronsymbolen = ', bronsymbolen)
    print('q_compansie = ', q_compansie, '\n')
    bronsymbolen_vast = copy.deepcopy(bronsymbolen)
    


    print('BRONCODERING')
    print('----------------------- \n')
    print('Bron -> Macro')
    alfabet_scalair = q_compansie
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet_scalair)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    #print('macrosymbolen = ', macrosymbolen, '\n')
    print('entropie = ', entropie)
    stop = time.time()
    print('Time: kwantisatie = ', stop - start, '\n')
    


    print('Codetabel + dictionary')
    start_1 = time.time()
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    #print('dictionary = ', dictionary, '\n')
    print('gem_len = ', gem_len)
    #print('codetabel = ',codetabel, '\n')
    stop_1 = time.time()
    print('Time: maak_codetabel_Huffman = ', stop_1 - start_1, '\n')


    print('Macro -> binair')
    start_2 = time.time()
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += datapoint
    print('data_binair = ', data_binair, '\n')
    stop_2 = time.time()
    print('Time: Huffman_encodeer = ', stop_2 - start_2, '\n')
    

    print('Binair -> macro')
    start_3 = time.time()
    data_macro = obj.Huffman_decodeer(data_binair_str, np.array(codetabel), np.array(index_lijst))
    print('data_macro = ', data_macro, '\n')
    stop_3 = time.time()
    print('Time: Huffman_decodeer = ', stop_3 - start_3, '\n')

    
    print('Macro -> Bron')
    start_4 = time.time()
    data_bron = obj.vector_naar_scalair(data_macro, alfabet_scalair)
    print('data_bron = ', data_bron, '\n')
    stop_4 = time.time()
    print('Time: vector_naar_scalair = ', stop_4 - start_4, '\n')
    

    print('Vaste-lengte')
    start_5 = time.time()
    encoded_vast = obj.vaste_lengte_encodeer(bronsymbolen_vast, alfabet_scalair)
    print('bronsymbolen_encoded = ', encoded_vast)
    decoded_vast = obj.vaste_lengte_decodeer(encoded_vast, alfabet_scalair)
    print('bronsymbolen_decoded = ', decoded_vast)
    stop_5 = time.time()
    print('Time: vaste_lengte_encodeer + decodeer = ', stop_5 - start_5, '\n')
    

    return 1
    

def run_kanaalcodering():
    return 1

def run_moddet():
    obj = ModDet()
    bitstring = bin(random.randint(0,2**16))[2:].zfill(16)
    bitvector = []
    for bit in bitstring:
        bitvector.append(int(bit))
    
    print('testing for:', bitvector)
    """
    if (obj.demapper(obj.mapper(bitvector,'BPSK'), 'BPSK') == bitvector): print('BPSK works!')
    if (obj.demapper(obj.mapper(bitvector,'4QAM'), '4QAM') == bitvector): print('4QAM works!')
    if (obj.demapper(obj.mapper(bitvector,'4PAM'), '4PAM') == bitvector): print('4PAM works!')
    if (obj.demapper(obj.mapper(bitvector,'4PSK'), '4PSK') == bitvector): print('4PSK works!')

    samples = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    print('Testing Channel (adding white noise):')
    print(obj.kanaal(samples, 0.01, 1)) """

    print('Testing modulation:')    
    a = obj.mapper(bitvector, '4QAM')
    print(a)
    s = obj.moduleer(a, 10**(-6), 6, 2*10**6, 0.5, 10)
    r = obj.kanaal(s, math.sqrt(0.75*6/2/(10**(-6))), 1)
    rdemod = obj.demoduleer(r, 10**(-6), 6, 2*10**6, 0.5, 10, math.pi/16)
    rdown = obj.decimatie(rdemod, 6, 10)
    u = obj.maak_decisie_variabele(rdown, 1, math.pi/16)
    a_estim = obj.decisie(u, '4QAM')
    print(a_estim)
    return 

warnings.simplefilter('ignore') # ignore warnings of integral


#run_kwantisatie()
#run_broncodering()
#run_kanaalcodering()
run_moddet()
