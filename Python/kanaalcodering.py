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
        
        G = np.array([
            [1,1,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,1,0,0,0,1,0,0,1,0,1,0,0,0],
            [0,0,1,0,0,1,1,0,0,0,0,0,0,0],
            [0,1,0,1,0,1,1,0,0,0,0,0,0,0],
            [0,1,0,0,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,1,0,0,1,0,0],
            [0,0,0,0,0,1,0,0,1,0,0,0,1,0],
            [0,1,0,0,0,1,1,1,1,0,0,0,0,0],
            [0,1,0,0,0,0,1,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,1,0,1,1,0,0,0,0]
        ])
        # Implementeer vanaf hier
        
        bitenc = np.fmod(np.matmul(bitstring_vec, G),2)
                
        # bitenc : vector met gecodeerde bits
        return bitenc
    
    # functie die de decoder van de uitwendige code implementeert
    def decodeer_uitwendig(bitstring):
        # bitstring : vector met gecodeerde bits
        bitstring_vec = np.array(bitstring)
        bool_fout = 0
        bitdec = np.zeros((1,10))

        H = np.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0 ],[0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])
        
        # Implementeer vanaf hier
        s = np.matmul(bitstring_vec, np.transpose(H))

        if (s == np.zeros((1, 4))):
            bool_fout = 0
        else: bool_fout = 1
        
        #syndroom tabel

        syn = np.array([[0,0,0,0],[1,0,0,1],[1,0,0,0],[0,1,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,0],[0,0,1,0],[1,1,1,1],[0,0,0,1],[0,0,1,1],[1,1,0,1],[0,1,1,1],[0,1,0,1],[1,0,1,0],[1,0,1,1]])
        err = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,1,0,0,0,0,0,0,0]])

        if (s == syn[0]):
            bool_fout = 0
            bitdec = bitstring_vec
        else:
            bool_fout = 1
            e = err[np.where(s = syn)]
            bitdec = np.mdo((bitstring_vec + e))
                

        # bitdec : vector met gedecodeerde bits bij volledige foutcorrectie
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders        
        return(bitdec,bool_fout)
        
    # functie die de encoder van de inwendige code implementeert
    def encodeer_inwendig(bitstring,g_x):
        # bitstring : vector met ongecodeerde bits
        # g_x : CRC-veelterm
        # Implementeer vanaf hier
        shift = [1] + [0] * (len(g_x) - 1)
        b_shifted = np.abs(np.fmod(np.polymul(bitstring, shift),2))
        remainder = np.abs(np.fmod(np.polydiv(b_shifted, g_x)[1],2))
        bitenc = np.polyadd(b_shifted, remainder)
        # bitenc : vector met gecodeerde bits
        return bitenc

    # functie die de decoder van de inwendige code implementeert
    def decodeer_inwendig(bitstring,g_x):
        # bitstring : vector met gecodeerde bits
        # g_x : CRC-veelterm
        # Implementeer vanaf hier
        
        rem = np.abs(np.fmod(np.polydiv(bitstring, g_x)[1], 2))
        bool_fout = False if np.all((rem == 0)) else True
        bitdec = bitstring[:len(g_x)-1]
        # bitenc : vector met gedecodeerde bits
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders
        return (bitdec,bool_fout)


print("----Test 1: Inwendig a---")
g_x = [1,1,0,1,0,1]
a = Kanaalcodering.encodeer_inwendig([1,1,0,1,1],g_x)
print(a)
print(Kanaalcodering.decodeer_inwendig(a, g_x))

print("----Test 2: Inwendig b---")
g_x = [1,1,0,0,1,1,0,1,1]
a = Kanaalcodering.encodeer_inwendig([1,1,0,1,1],g_x)
print(a)
print(Kanaalcodering.decodeer_inwendig(a, [1,1,0,1,0,1]))

print("---Test 3: Blokcode ---")
a = Kanaalcodering.encodeer_uitwendig([1,0,1,0,1,1,0,1,1,0])
print(a)
print(Kanaalcodering.decodeer_uitwendig(a))