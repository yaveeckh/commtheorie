from numpy.lib.index_tricks import IndexExpression
from kwantisatie import Kwantisatie
from broncodering import Broncodering
from kanaalcodering import Kanaalcodering
from moddet import ModDet
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy

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
    print(macrosymbolen)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    
    # Codetabel
    # ['0', '1', '2',...] -> [11001, 1111, 000]
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('dictionary = ', dictionary)
    print('gem_len = ', gem_len)
    print('codetabel = ',codetabel)

    # Macro -> binair : KLOPT
    data_binair = [obj.Huffman_encodeer(macrosymbolen[i], dictionary) for i in range(len(macrosymbolen))]
    data_binair = [x[0] for x in data_binair]
    print('data_binair = ', data_binair)

    # Binair -> macro : KLOPT NIET
    data_decoded = [obj.Huffman_decodeer([int(char) for char in data_binair[i]], np.array(codetabel), np.array(index_lijst)) for i in range(len(data_binair))]
    #data_decoded = [x[0] for x in data_decoded if x != []]
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

    
    # op kwantisatie - compansie
    r_compansie, q_compansie, bronsymbolen = run_kwantisatie()
    #print('bronsymbolen = ', bronsymbolen)
    print('q_compansie = ', q_compansie, '\n')
    bronsymbolen_vast = copy.deepcopy(bronsymbolen)
    
    print('BRONCODERING')
    print('-----------------------')
    # Bron -> Macro
    # ['1', '1', '1', '2', '1', '3',...] -> ['11', '12', '13',...] -> ['0', '1', '2',...]
    alfabet_scalair = q_compansie
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet_scalair)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    #print('macrosymbolen = ', macrosymbolen, '\n')
    

    # Codetabel
    # dicitonary ('0' : 11001, '1' : 1111, '2' : 000,...])
    print('check_0')
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    #print('dictionary = ', dictionary, '\n')
    print('gem_len = ', gem_len)
    #print('codetabel = ',codetabel, '\n')


    # Macro -> binair
    print('check_1')
    data_binair = [obj.Huffman_encodeer(macrosymbolen[i], dictionary) for i in range(len(macrosymbolen))]
    data_binair = [x[0] for x in data_binair]
    #print('data_binair = ', data_binair, '\n')



    # Binair -> macro
    print('check_2')
    data_macro = [obj.Huffman_decodeer([int(char) for char in data_binair[i]], np.array(codetabel), np.array(index_lijst)) for i in range(len(data_binair))]
    #print('data_macro = ', data_macro, '\n')


    # Macro -> Bron
    # ['0', '1', '2',...] -> ['1', '1', '1', '2', '1', '3',...]
    print('check_3')
    data_bron = obj.vector_naar_scalair(data_macro, alfabet_scalair)
    #print('data_bron = ', data_bron, '\n')
    

    # Vaste-lengte
    print('check_4')
    encoded_vast = obj.vaste_lengte_encodeer(bronsymbolen_vast, alfabet_scalair)
    #print('bronsymbolen_encoded = ', encoded_vast)
    decoded_vast = obj.vaste_lengte_decodeer(encoded_vast, alfabet_scalair)
    #print('bronsymbolen_decoded = ', decoded_vast)

    return 1
    

def run_kanaalcodering():
    return 1

def run_moddet():
    return 1

warnings.simplefilter('ignore') # ignore warnings of integral


#run_kwantisatie()
run_broncodering()
#run_kanaalcodering()
#run_moddet()
