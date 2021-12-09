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
        #print(bitstring)
        b_shifted = np.append(bitstring, np.array([0] * (len(g_x) - 1)))
        print(b_shifted)
        # print(shift)
        # b_shifted = np.mod(np.polymul(bitstring, shift),2)
        #print(b_shifted)
        remainder = np.mod(np.polydiv(b_shifted, g_x)[1],2)
        #print(remainder)
        bitenc = np.polyadd(b_shifted, remainder)
        #print(bitenc)
        # bitenc : vector met gecodeerde bits
        return bitenc

    # functie die de decoder van de inwendige code implementeert
    def decodeer_inwendig(self, bitstring,g_x):
        # bitstring : vector met gecodeerde bits
        # g_x : CRC-veelterm
        # Implementeer vanaf hier
        
        rem = np.mod(np.polydiv(bitstring, g_x)[1], 2)
        bool_fout = False if np.all((rem == 0)) else True
        bitdec = bitstring[:len(g_x)-1]
        # bitenc : vector met gedecodeerde bits
        # bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders
        return (bitdec,bool_fout)

    def bron_enckanaal(bitstring):
        bit_vec = np.array([])
        for bits in bitstring:
            for bit in bits:
                bit_vec = np.append(bit_vec, int(bit))
        bit_vec = bitstring

        n = int(np.ceil(len(bit_vec)/10))
        n_last_group = len(bit_vec) % 10
        
        if(n_last_group != 0):
            bit_vec = np.append(bit_vec, np.array((10-n_last_group)*[0]))

        bit_vec = np.resize(bit_vec, (n,10))
        return bit_vec

    def kanaalencodering_1(self, bit_vec):
        bit_vec_encoded = np.array([self.encodeer_uitwendig(group) for group in bit_vec])
        return bit_vec_encoded
    

    def kanaal_kanaaldec(bit_vec):
        n = int(len(bit_vec)/14)
        bit_vec = np.resize(bit_vec, (n,14))

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

        #bit_vec_decoded = np.array([self.decodeer_uitwendig(group)[0] for group in bit_vec]).flatten()

        return np.array(bit_vec_decoded), fouten


    def kanaal_simulatie(self, bit_vec, p):
        a = [0,1]
        f = np.random.choice(a, (len(bit_vec),14), p=[1-p, p])
        bit_vec_altered = np.mod(np.add(bit_vec, f),2)
        return bit_vec_altered

    def simulation_1(self, p, n = 1000000 ,T_max = 6):

        T = 0

        rand = np.random.randint(0,2, n)
        rand_grouped = np.resize(rand, (int(len(rand)/10),10))

        encoded = self.kanaalencodering_1(rand_grouped)
        
        encoded_ch = self.kanaal_simulatie(encoded,p)
        decoded = self.kanaaldecodering_1(encoded_ch, True)
        decoded_bits = decoded[0]
        decoded_corrected = copy.deepcopy(decoded[0])
        decoded_fouten = decoded[1]
        print(f'Initiele fouten: {decoded_fouten}')
        print(len(decoded_fouten))
        if(decoded_fouten != []):
            
            while(T < T_max and len(decoded_fouten) > 0):
                ARQ = False if T == T_max-1 else True 
                #print(f'---Retransmissie {T}---')
                retrans_pack = [encoded[i] for i in decoded_fouten]
                retransmitted = self.kanaal_simulatie(retrans_pack,p)
                d = self.kanaaldecodering_1(retransmitted, ARQ=True)
                decoded_retransmitted = d[0]
                fouten_retransmitted = d[1]
                nieuwe_fouten = [decoded_fouten[i] for i in fouten_retransmitted]
                #print(f'oude fouten: {decoded_fouten}')
                #print(f'nieuwe fouten: {nieuwe_fouten}')
                    
                if T < T_max-1:
                    to_remove = []
                    for index, row in enumerate(decoded_fouten):
                        if row not in nieuwe_fouten:
                            #print(f'{row}, {decoded_retransmitted[index]}, {rand_grouped[row]}')
                            decoded_corrected[row] = decoded_retransmitted[index]
                            to_remove.append(row)
                    
                    for i in to_remove: decoded_fouten.remove(i)
                    
                    T += 1
                else:
                    #print("Volledige decodering!")

                    for i, row in enumerate(decoded_fouten):
                        decoded_corrected[row] = decoded_retransmitted[i]
                    T += 1
                #print("--------")

        fout = False if np.all(rand_grouped == decoded_corrected) else True
        print(fout)

        fouten = 0
        for idx,i in enumerate(rand_grouped):
            if np.any(rand_grouped[idx] != decoded_corrected[idx]):
                fouten += 1
        pe = fouten/(n/10)

        return pe

    def simulation_2(self, p, n = 1000000 ,T_max = 2, g_x=[1,1,0,1,0,1]):
        T = 0

        rand = np.random.randint(0,2, n)
        rand_grouped = np.resize(rand, (int(len(rand)/5),5))

        encoded = self.kanaalencodering_2(rand_grouped, g_x)

        #Simuleer kanaal door bits te veranderen
        encoded_ch = self.kanaal_simulatie(encoded,p)

        #Decodeer wat door het kanaal komt
        decoded = self.kanaaldecodering_2(encoded_ch, g_x)
        decoded_bits = decoded[0]
        decoded_fouten = decoded[1]


        #Kopieer decoded voor een gecorrigeerde array
        decoded_corrected = copy.deepcopy(decoded[0])

        print(f'Initiele fouten: {decoded_fouten}')

        if(decoded_fouten != []):
            
            while(T < T_max and len(decoded_fouten) > 0):
                print(f'---Retransmissie {T}---')
                
                #Retransmit
                retrans_pack = [encoded[i] for i in decoded_fouten]
                retransmitted = self.kanaal_simulatie(retrans_pack,p)

                #Decode
                d = self.kanaaldecodering_2(retransmitted, g_x)
                decoded_retransmitted = d[0]
                fouten_retransmitted = d[1]

                nieuwe_fouten = [decoded_fouten[i] for i in fouten_retransmitted]
                print(f'oude fouten: {decoded_fouten}')
                print(f'nieuwe fouten: {nieuwe_fouten}')
                    
                if T < T_max:
                    to_remove = []
                    for index, row in enumerate(decoded_fouten):
                        if row not in nieuwe_fouten:
                            print(f'{row}, {decoded_retransmitted[index]}, {rand_grouped[row]}')
                            decoded_corrected[row] = decoded_retransmitted[index]
                            to_remove.append(row)
                    
                    for i in to_remove: decoded_fouten.remove(i)
                    
                    T += 1

        fout = False if np.all(rand_grouped == decoded_corrected) else True
        print(fout)

    def simulation_3(self, p, n = 1000000 ,T_max = 2, g_x=[1,0,0,0,1,1,0,1,1,0]):
        T = 0

        rand = np.random.randint(0,2, n)
        rand_grouped = np.resize(rand, (int(len(rand)/2),2))
        print(rand_grouped)
        encoded = self.kanaalencodering_2(rand_grouped, g_x)

        #Simuleer kanaal door bits te veranderen
        encoded_ch = self.kanaal_simulatie(encoded,p)

        #Decodeer wat door het kanaal komt
        decoded = self.kanaaldecodering_2(encoded_ch, g_x)
        decoded_bits = decoded[0]
        decoded_fouten = decoded[1]


        #Kopieer decoded voor een gecorrigeerde array
        decoded_corrected = copy.deepcopy(decoded[0])

        print(f'Initiele fouten: {decoded_fouten}')

        if(decoded_fouten != []):
            
            while(T < T_max and len(decoded_fouten) > 0):
                print(f'---Retransmissie {T}---')
                
                #Retransmit
                retrans_pack = [encoded[i] for i in decoded_fouten]
                retransmitted = self.kanaal_simulatie(retrans_pack,p)

                #Decode
                d = self.kanaaldecodering_2(retransmitted, g_x)
                decoded_retransmitted = d[0]
                fouten_retransmitted = d[1]

                nieuwe_fouten = [decoded_fouten[i] for i in fouten_retransmitted]
                print(f'oude fouten: {decoded_fouten}')
                print(f'nieuwe fouten: {nieuwe_fouten}')
                    
                if T < T_max:
                    to_remove = []
                    for index, row in enumerate(decoded_fouten):
                        if row not in nieuwe_fouten:
                            print(f'{row}, {decoded_retransmitted[index]}, {rand_grouped[row]}')
                            decoded_corrected[row] = decoded_retransmitted[index]
                            to_remove.append(row)
                    
                    for i in to_remove: decoded_fouten.remove(i)
                    
                    T += 1

        fout = False if np.all(rand_grouped == decoded_corrected) else True
        print(fout)

        # fouten = 0
        # for idx,i in enumerate(rand_grouped):
        #     if np.any(rand_grouped[idx] != decoded_corrected[idx]):
        #         fouten += 1
        
        # print(rand_grouped)
        # print(encoded)
        # print(decoded)
        #pe = fouten/(n/5) if fouten!= 0 else 0
        #print(decoded_corrected)
        return

    def plot_s_1(self):
        pe = []
        T = range(16)
        for t_max in range(16):
            pe.append(self.simulation_1(0.05, n=1000000, T_max=t_max ))
        
        plt.plot(pe)
        plt.yscale("log")
        plt.ylabel("p_e")
        plt.xlabel("T")
        plt.savefig("p_e_1.png")
        plt.close()


        print(pe)
    
    def kanaalencodering_2(self, bit_vec, g_x):
        crc = np.array([self.encodeer_inwendig(group, g_x) for group in bit_vec])
        print(crc)
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
# print(obj.simulation_2(0.05,20, 5))

#print(obj.encodeer_inwendig([0,0,0,1,1], [1,1,1,1,0,1]))
