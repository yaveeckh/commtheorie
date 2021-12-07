import numpy as np
from numpy.ma.core import bitwise_xor
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
        
        bitenc = np.mod(np.matmul(bitstring_vec, G),2)
                
        # bitenc : vector met gecodeerde bits
        return bitenc
    
    # functie die de decoder van de uitwendige code implementeert
    def decodeer_uitwendig(bitstring):
        # # bitstring : vector met gecodeerde bits
        # bitstring_vec = np.array(bitstring)
        # bool_fout = 0
        # bitdec = np.zeros((1,10))

        H = np.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0 ],[0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])
        
        # # Implementeer vanaf hier
        
        bitstring_vec = np.array(bitstring)
        s = np.mod(np.matmul(bitstring_vec,np.transpose(H)),2)
       
        bool_fout = False if (np.all(s == 0)) else True
        
        syn = np.array([
            [0,0,0,0],
            [1,0,0,1],
            [1,0,0,0],
            [0,1,1,0],
            [1,1,1,0],
            [1,1,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [1,1,1,1],
            [0,0,0,1],
            [0,0,1,1],
            [1,1,0,1],
            [0,1,1,1],
            [0,1,0,1],
            [1,0,1,0],
            [1,0,1,1]
        ])


        err = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,1,0,0,0,0,0,0,0]
        ])

        e = np.zeros(14)
        r_c = np.zeros(14)

        if (bool_fout):
            index = 0
            for i in range(16):
                if np.all(s == syn[i]):
                    index = i
            
            e = err[index]
            r_c = np.bitwise_xor(bitstring_vec, e)
        else:
            r_c = bitstring_vec

        idx = [0,10,2,3,4,11,12,7,13,9]
        bitdec = r_c[idx]
        # bitdec : vector met gedecodeerde bits bij volledige foutcorrectie
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders        
        return(bitdec,bool_fout)
        
    # functie die de encoder van de inwendige code implementeert
    def encodeer_inwendig(bitstring,g_x):
        # bitstring : vector met ongecodeerde bits
        # g_x : CRC-veelterm
        # Implementeer vanaf hier
        shift = [1] + [0] * (len(g_x) - 1)
        b_shifted = np.mod(np.polymul(bitstring, shift),2)
        remainder = np.mod(np.polydiv(b_shifted, g_x)[1],2)
        bitenc = np.polyadd(b_shifted, remainder)
        # bitenc : vector met gecodeerde bits
        return bitenc

    # functie die de decoder van de inwendige code implementeert
    def decodeer_inwendig(bitstring,g_x):
        # bitstring : vector met gecodeerde bits
        # g_x : CRC-veelterm
        # Implementeer vanaf hier
        
        rem = np.mod(np.polydiv(bitstring, g_x)[1], 2)
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
a = Kanaalcodering.encodeer_uitwendig([1,0,0,0,1,1,0,1,1,0])
print(a)
b = Kanaalcodering.decodeer_uitwendig(a)
print(b)
    