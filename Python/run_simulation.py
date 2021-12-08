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
    """
    opt_lin_kwant = obj.bepaal_optimale_lineaire_kwantisator(2**6, True)
    r_opt_lin = opt_lin_kwant[4]
    q_opt_lin = opt_lin_kwant[5]
    gekwantiseerd_lin = obj.kwantiseer(r_opt_lin, q_opt_lin)
    """

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
    
    compansie_kwant = obj.bepaal_compansie_kwantisator(2**6)
    r_compansie = compansie_kwant[3]
    q_compansie = compansie_kwant[4]
    gekwantiseerd_compansie = obj.kwantiseer(r_compansie, q_compansie)

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

    """
    #########################
    # Lloyd-Max KWANTISATOR #
    #########################

    opt_kwant = obj.bepaal_Lloyd_Max_kwantisator(2**6)
    r_opt = opt_kwant[3]
    q_opt = opt_kwant[4]
    gekwantiseerd_opt = obj.kwantiseer(r_opt, q_opt)

    
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
    #obj.save_and_play_music(obj.kwantiseer(r_opt, q_opt), "LM.wav", 0)

    #return (r_opt,q_opt,gekwantiseerd_opt)
    return (r_compansie,q_compansie,gekwantiseerd_compansie)
    #return (r_opt_lin,q_opt_lin,gekwantiseerd_lin)

    
def run_broncodering():
    obj = Broncodering()
    
    print('Kwantisatie')
    start = time.time()
    r, q, bronsymbolen = run_kwantisatie()
    bronsymbolen_vast = copy.deepcopy(bronsymbolen)
    stop = time.time()
    print('Time: kwantisatie = ', stop - start, '\n')


    start_0 = time.time()
    print('Bron -> Macro')
    alfabet_scalair = q
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet_scalair)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    stop_0 = time.time()
    print('Time: scalair_naar_vector = ', stop_0 - start_0, '\n')
    

    print('Codetabel + dictionary')
    start_1 = time.time()
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('gem_len = ', gem_len)
    stop_1 = time.time()
    print('Time: maak_codetabel_Huffman = ', stop_1 - start_1, '\n')


    print('Macro -> binair')
    start_2 = time.time()
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += datapoint
    stop_2 = time.time()
    print('Time: Huffman_encodeer = ', stop_2 - start_2, '\n')
    with open('data_compansie.txt', 'w') as file_out:
        file_out.write('\n'.join(data_binair))


    print('Binair -> macro')
    start_3 = time.time()
    data_macro = obj.Huffman_decodeer(data_binair_str, np.array(codetabel), np.array(index_lijst))
    stop_3 = time.time()
    print('Time: Huffman_decodeer = ', stop_3 - start_3, '\n')

    
    print('Macro -> Bron')
    start_4 = time.time()
    data_bron = obj.vector_naar_scalair(data_macro, alfabet_scalair)
    stop_4 = time.time()
    print('Time: vector_naar_scalair = ', stop_4 - start_4, '\n')
    data_bron_str = []
    for data_point in data_bron:
        data_bron_str.append(str(data_point))
    with open('Macro>Bron.txt', 'w') as file_out:
        file_out.write('\n'.join(data_bron_str))


    print('Vaste-lengte')
    start_5 = time.time()
    encoded_vast = obj.vaste_lengte_encodeer(bronsymbolen_vast, alfabet_scalair)
    decoded_vast = obj.vaste_lengte_decodeer(encoded_vast, alfabet_scalair)
    stop_5 = time.time()
    print('Time: vaste_lengte_encodeer + decodeer = ', stop_5 - start_5, '\n')

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
