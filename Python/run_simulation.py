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

    return (q_compansie)
    
def run_broncodering():
    obj = Broncodering()

    """ TEST opgave voorbeeld
    # Bron -> Macro
    # ['1', '1', '1', '2', '1', '3',...] -> ['11', '12', '13',...] -> ['0', '1', '2',...]
    alfabet = ['1','2','3','4','5','6']
    stream = '131313141516212223242526313233343536414243444546515253545556616263646566'
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
    # ['0', '1', '2',...] -> [11001, 1111, 000]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(sc_to_vect[2], sc_to_vect[1])
    print('dictionary = ', dictionary)
    print('gem_len = ', gem_len)
    print('codetabel = ',codetabel)

    # Macro -> Bron
    # ['0', '1', '2',...] -> ['1', '1', '1', '2', '1', '3',...]
    bronsymbolen_nadien = obj.vector_naar_scalair(sc_to_vect[0], alfabet)
    print('bronsymbolen = ', bronsymbolen_nadien)

    # Vaste-lengte
    encoded = obj.vaste_lengte_encodeer(stream, alfabet)
    print(encoded)
    decoded = obj.vaste_lengte_decodeer(encoded, alfabet)
    print('bronsymbolen_decoded = ', decoded)
    """

    #############################################################################################


    """ TEST op kwantisatie - compansie
    q_compansie = run_kwantisatie()
    # Bron -> Macro
    # ['1', '1', '1', '2', '1', '3',...] -> ['11', '12', '13',...] -> ['0', '1', '2',...]
    alfabet = q_compansie
    bronsymbolen = [-0.4424896240234375, -0.35686492919921875, -0.3108863830566406, -0.27846336364746094, -0.2536945343017578, -0.23332881927490234, -0.21568775177001953, -0.19983863830566406, -0.18565940856933594, -0.1729879379272461, -0.16143512725830078, -0.15072059631347656, -0.14066028594970703, -0.1311511993408203, -0.12212038040161133, -0.1135258674621582, -0.10532999038696289, -0.09747505187988281, -0.08991289138793945, -0.08258628845214844, -0.0754704475402832, -0.0685272216796875, -0.06174039840698242, -0.05509614944458008, -0.0485692024230957, -0.04214334487915039, -0.035791873931884766, -0.02950429916381836, -0.023265361785888672, -0.017072933399077506, -0.010893821716308594, -0.004744529724121094, 0.0014069080352783203, 0.0075647830963134766, 0.013753175735473633, 0.019980430603027344, 0.026259422302246094, 0.032600227242970195, 0.03902554512023926, 0.04555559158325195, 0.05222177505493164, 0.05904436111450195, 0.0660521339938275, 0.07326650619506836, 0.08071517944335938, 0.08841371536254883, 0.09637022018432617, 0.10458707809448242, 0.1130952685113065, 0.12191173102535419, 0.1310882568359375, 0.14077281951904297, 0.15110301971435547, 0.16219520568847656, 0.17421531677246094, 0.1872243881225586, 0.2014622688293457, 0.21708250045776367, 0.2345428466796875, 0.2544822692871094, 0.27840328216552734, 0.3101734478858872, 0.356815708413115, 0.4490060806274414]
    sc_to_vect = obj.scalair_naar_vector(bronsymbolen, alfabet)
    entropie = 0.0
    for kans in sc_to_vect[2]:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    
    # Codetabel
    # ['0', '1', '2',...] -> [11001, 1111, 000]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(sc_to_vect[2], sc_to_vect[1])
    #print('dictionary = ', dictionary)
    print('gem_len = ', gem_len)
    #print('codetabel = ',codetabel)

    # Macro -> Bron
    # ['0', '1', '2',...] -> ['1', '1', '1', '2', '1', '3',...]
    bronsymbolen_nadien = obj.vector_naar_scalair(sc_to_vect[0], alfabet)
    print('bronsymbolen = ', bronsymbolen_nadien)
    """

    return 1
    

def run_kanaalcodering():
    return 1

def run_moddet():
    return 1

warnings.simplefilter('ignore') # ignore warnings of integral


#run_kwantisatie()
#run_broncodering()
#run_kanaalcodering()
#run_moddet()
