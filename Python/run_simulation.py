
from kwantisatie import Kwantisatie
from broncodering import Broncodering
from kanaalcodering import Kanaalcodering
from moddet import ModDet
import numpy as np
import matplotlib.pyplot as plt
import warnings

from playsound import playsound




def run_kwantisatie():
    obj = Kwantisatie(0)
    
    ########################
    # LINEAIRE KWANTISATOR #
    ########################
    print('LINEAIR QUANTISATION')
    print('--------------------')

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
    
    print('Generating plot: fU(u)')
    opt_lin_kwant = obj.bepaal_optimale_lineaire_kwantisator(2**6, True)
    r_opt_lin = opt_lin_kwant[4]
    q_opt_lin = opt_lin_kwant[5]
    

    """
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
    
    print('Generating plot: fU(u)')
    compansie_kwant = obj.bepaal_compansie_kwantisator(2**6)
    r_compansie = compansie_kwant[3]
    q_compansie = compansie_kwant[4]

    """
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

    print('Done')
    
    return 1
    
def run_broncodering():
    obj = Broncodering()

    """ 
    # TEST maak_codetabel_Huffman
    rel_freq = [11/24, 4/24, 4/24, 2/24, 2/24, 1/24]
    alfabet = ['1','2','3','4','5','6']
    print(obj.maak_codetabel_Huffman(rel_freq, alfabet))
    return 1
    """

    
    # Bron -> Macro
    alfabet = ['1','2','3','4','5','6']
    stream = '13131314151621222324252631323334353641424344454651525354555661626364656612323456214314345621635412653443524536124651426511122314314141515116166654546656545363264623441264612554353453451243564361435424123425423515632456154323242562453556346235125432415'
    bronsymbolen = []
    for symbol in stream:
        bronsymbolen.append(symbol)
    sc_to_vect = obj.scalair_naar_vector(bronsymbolen, alfabet)
    print(sc_to_vect)
    entropie = 0.0
    for kans in sc_to_vect[2]:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    
    # Codetabel
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(sc_to_vect[2], sc_to_vect[1])
    print('dictionary = ', dictionary)
    print('gem_len = ', gem_len)
    print('codetabel = ',codetabel)

    # Macro -> Bron
    bronsymbolen_nadien = obj.vector_naar_scalair(sc_to_vect[0], sc_to_vect[1])
    print('bronsymbolen = ', bronsymbolen_nadien)
    
    

def run_kanaalcodering():
    return 1

def run_moddet():
    return 1

warnings.simplefilter('ignore') # ignore warnings of integral

#run_kwantisatie()
run_broncodering()
#run_kanaalcodering()
#run_moddet()
