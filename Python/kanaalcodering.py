import numpy as np
import numpy.matlib
import math
from scipy import signal

class Kanaalcodering():
    def __init__():
        pass
    
    # functie die de encoder van de uitwendige code implementeert
    def encodeer_uitwendig(bitstring):
        # bitstring : vecter met ongecodeerde bits
        bitstring_vec = np.array(bitstring)
        # generator matrix
        G = np.array([[1,1,0,0,0,0,0,0,1,0,0,0,0,0], [0,1,0,0,0,1,0,0,1,0,1,0,0,0], [0,0,1,0,0,1,1,0,0,0,0,0,0,0], [0,1,0,1,0,1,1,0,0,0,0,0,0,0], [0,1,0,0,1,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,1,0,1,0,0,1,0,0], [0,0,0,0,0,1,0,0,1,0,0,0,1,0], [0,1,0,0,0,1,1,1,1,0,0,0,0,0], [0,1,0,0,0,0,1,0,0,0,0,0,0,1], [0,1,0,0,0,0,1,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0,1,1,0,0,0,0]])
        
        # Implementeer vanaf hier
        
        bitenc = np.matmul(bitstring_vec, G)
                
        # bitenc : vector met gecodeerde bits
        return bitenc
    
    # functie die de decoder van de uitwendige code implementeert
    def decodeer_uitwendig(bitstring):
        # bitstring : vector met gecodeerde bits
        
        H = np.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0 ],[0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])
        # Implementeer vanaf hier
        
        
        # bitdec : vector met gedecodeerde bits bij volledige foutcorrectie
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders        
        return(bitdec,bool_fout)
        
    # functie die de encoder van de inwendige code implementeert
    def encodeer_inwendig(bitstring,g_x):
        # bitstring : vector met ongecodeerde bits
        # g_x : CRC-veelterm
        
        # Implementeer vanaf hier

        
        # bitenc : vector met gecodeerde bits
        return bitenc
    
    # functie die de decoder van de inwendige code implementeert
    def decodeer_inwendig(bitstring,g_x):
        # bitstring : vector met gecodeerde bits
        # g_x : CRC-veelterm
        
        # Implementeer vanaf hier
        
            
        # bitenc : vector met gedecodeerde bits
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders
        return (bitdec,bool_fout)
