import matplotlib
import numpy as np
from numpy.lib.function_base import kaiser
from numpy.ma.core import bitwise_xor
import numpy.matlib
import math
from numpy.typing import _128Bit
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib as mplt
import copy
import sys
np.set_printoptions(threshold=sys.maxsize)

class Kanaalcodering():
    def __init__(self):
        pass
    
    # functie die de encoder van de uitwendige code implementeert
    def encodeer_uitwendig(self, bitstring):
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
    def decodeer_uitwendig(self, bitstring, volledige_decodering = False):
        # # bitstring : vector met gecodeerde bits
        # bitstring_vec = np.array(bitstring)
        # bool_fout = 0
        # bitdec = np.zeros((1,10))

        H = np.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0 ],[0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])
        
        # # Implementeer vanaf hier
        bitstring_vec = np.array(bitstring)
        s = np.mod(np.matmul(bitstring_vec,np.transpose(H)),2)
        #print(s)
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

        bitdec = np.array([])
        e = np.zeros(14)
        r_c = np.zeros(14)

        if (bool_fout and volledige_decodering):
            index = 0
            for i in range(16):
                if np.all(s == syn[i]):
                    index = i
            
            e = err[index]
            r_c = np.mod(bitstring_vec+e,2)
        else:
            r_c = bitstring_vec

        idx = [0,10,2,3,4,11,12,7,13,9]
        bitdec = r_c[idx]
        # bitdec : vector met gedecodeerde bits bij volledige foutcorrectie
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders        
        return(bitdec,bool_fout)
        
    # functie die de encoder van de inwendige code implementeert
    def encodeer_inwendig(self, bitstring,g_x):
        # bitstring : vector met ongecodeerde bits
        # g_x : CRC-veelterm
        # Implementeer vanaf hier

        b_shifted = np.append(bitstring, np.array([0] * (len(g_x) - 1)))
        remainder = np.mod(np.polydiv(b_shifted, g_x)[1],2)
        bitenc = np.polyadd(b_shifted, remainder)
        
        # bitenc : vector met gecodeerde bits
        return bitenc

    # functie die de decoder van de inwendige code implementeert
    def decodeer_inwendig(self, bitstring,g_x):
        # bitstring : vector met gecodeerde bits
        # g_x : CRC-veelterm
        # Implementeer vanaf hier
        
        rem = np.mod(np.polydiv(bitstring, g_x)[1], 2)
        bool_fout = False if np.all((rem == 0)) else True
        bitdec = bitstring[:len(bitstring)-len(g_x)+1]
        # bitenc : vector met gedecodeerde bits
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders
        return (bitdec,bool_fout)

    def kanaalencodering_1(self, bit_vec):
        bit_vec_encoded = np.array([self.encodeer_uitwendig(group) for group in bit_vec])
        return bit_vec_encoded
    

    def kanaaldecodering_1(self, bit_vec, ARQ = False):
        
        fouten = []
        bit_vec_decoded = []

        for i in range(len(bit_vec)):
            decodeer_fout = []
            decoded_bits = []
            if(ARQ == False):
                decoded_group = self.decodeer_uitwendig(bit_vec[i], True)
                decodeer_fout = decoded_group[1]
                decoded_bits = decoded_group[0]
                bit_vec_decoded += [list(decoded_bits)]
            else:
                decoded_group = self.decodeer_uitwendig(bit_vec[i], False)
                decoded_bits = decoded_group[0]
                decodeer_fout = decoded_group[1]
                bit_vec_decoded += [list(decoded_bits)]
                if(decodeer_fout):
                    fouten.append(i)
                    
        return np.array(bit_vec_decoded), fouten


    def kanaal_simulatie(self, bit_vec, p):
        a = [0,1]
        f = np.random.choice(a, (len(bit_vec),14), p=[1-p, p])
        bit_vec_altered = np.mod(np.add(bit_vec, f),2)
        return bit_vec_altered


    def kanaalencodering_2(self, bit_vec, g_x):
        crc = np.array([self.encodeer_inwendig(group, g_x) for group in bit_vec])
        blk = np.array([self.encodeer_uitwendig(group) for group in crc])
        return blk
    
    def kanaaldecodering_2(self, bit_vec, g_x):
        fouten = []
        blk_decoded = []
        crc_decoded = []
        for i in range(len(bit_vec)):
            blk_decoded += [list(self.decodeer_uitwendig(bit_vec[i], True)[0])]
        
        for i in range(len(blk_decoded)):
            decoded_inwendig = self.decodeer_inwendig(blk_decoded[i],g_x)
            crc_decoded += [decoded_inwendig[0]]
            
            bool_fout = decoded_inwendig[1]

            if bool_fout:
                fouten.append(i)
        
<<<<<<< HEAD
        return np.array(crc_decoded), fouten
=======
        return np.array(crc_decoded), fouten


    
        
# print("----Test 1: Inwendig a---") 
# g_x = [1,1,0,1,0,1]
# a = Kanaalcodering.encodeer_inwendig([1,1,0,1,1],g_x)
# print(a)
# print(Kanaalcodering.decodeer_inwendig(a, g_x))

# print("----Test 2: Inwendig b---")
# g_x = [1,1,0,0,1,1,0,1,1]
# a = Kanaalcodering.encodeer_inwendig([1,1,0,1,1],g_x)
# print(a)
# print(Kanaalcodering.decodeer_inwendig(a, [1,1,0,1,0,1]))

# print("---Test 3: Blokcode ---")
# a = Kanaalcodering.encodeer_uitwendig([1,0,0,0,1,1,0,1,1,0])
# print(a)
# b = Kanaalcodering.decodeer_uitwendig(a)
# print(b)

#obj = Kanaalcodering()
# data_binair =  ['11', '10', '01', '00', '10', '01', '01', '00', '00', '00']
# a =  obj.kanaalencodering_1(data_binair)
# print(a)

# b = obj.kanaaldecodering_1(a)
# print(b)



#c = obj.plot_s_1()
#c = obj.plot_s_2()
>>>>>>> 9351b71f5e74536feabd9a721c908f916de0373f
