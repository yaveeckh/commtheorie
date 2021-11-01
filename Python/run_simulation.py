
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
    
    # Maak een figuur van de optimale SQR in dB 
    #   in functie van α = [2,...,8], waarbij M = 2**α
    delta = np.array([i for i in range(2,9)])
    y = np.array([obj.bepaal_optimale_lineaire_kwantisator(2**i)[2] for i in range(2,9)])
    winst = [0 for _ in range(0,6)] 
    for i in range(0,6):
        winst[i] = y[i+1] - y[i]
    print(max(winst))
    plt.plot(delta,y)
    plt.savefig('SQR.png')

    #Eerst wordt het originele geluidsfragment beperkt in grootte 
    #   door elke monsterwaarde te kwantiseren met slechts M = 26 = 64 waarden tussen 1 en -1

    return 1
    
def run_broncodering():
    return 1

def run_kanaalcodering():
    return 1

def run_moddet():
    return 1

warnings.simplefilter('ignore') # ignore warnings of integral


run_kwantisatie()
#run_broncodering()
#run_kanaalcodering()
#run_moddet()
