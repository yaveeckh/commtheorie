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
    obj.plot_distributie()
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
