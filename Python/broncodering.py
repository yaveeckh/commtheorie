
import numpy as np
from numpy.core.numeric import True_
import numpy.matlib
import math

class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        # left child
        self.left = left
        # right child
        self.right = right
        # 0 or 1
        self.code = ''
    
def code(node, val = ''):
    newVal = val + str(node.code)

    if(node.left):
        code(node.left,newVal)
    if(node.right):
        code(node.right,newVal)
    if(not node.left and not node.right):
        dictionary[node.symbol] = newVal
    
    return dictionary

class Broncodering():
    def __init__(self):
        pass
    
    # functie die codetabel opsteld voor de Huffmancode
    def  maak_codetabel_Huffman(self, rel_freq, alfabet):
        # rel_freq : vector met relatieve frequenties 
        # alfabet : vector met alle mogelijke symbolen in dezelfde volgorde als rel_freq 
        
        # Implementeer vanaf hier
        M = len(rel_freq)
        # dictionary : dictionary met symbolen als key en codewoord als value
        dictionary = dict()
        for key in alfabet:
            dictionary[key] = ''

        # boom : matrix met boomstructuur (zie opgave)
        boom = [[0, 0] for _ in range(len(rel_freq))]
        while len(rel_freq) > 1:
            freq_1 = min(rel_freq)
            index_1 = rel_freq.index(freq_1)
            key_1 = alfabet[index_1]
            del rel_freq[index_1]
            del alfabet[index_1]
            
            freq_0 = min(rel_freq)
            index_0 = rel_freq.index(freq_0)
            key_0 = alfabet[index_0]
            del rel_freq[index_0]
            del alfabet[index_0]

            boom.append([key_0, key_1])
            rel_freq.append(freq_0 + freq_1)
            alfabet.append([key_0, key_1])
        
        dictionary = self.recursive_code(alfabet[0], dictionary)

        # gem_len : gemiddelde codewoordlengte
        gem_len = 0
        for value in dictionary.values():
            gem_len += len(value)
        gem_len /= M

        return (dictionary,gem_len,boom)
    

    # functie voor het encoderen met vaste-lengte code
    def vaste_lengte_encodeer(data,alfabet):
        # data : de data die geëncodeerd moet worden
        # alfabet : vector met alle mogelijke symbolen
        
        # Implementeer vanaf hier
               
        # data_gecodeerd : de geëncodeerde data
        return data_geencodeerd
       
    # functie voor het decoderen met vaste-lengte code
    def vaste_lengte_decodeer(data,alfabet):
        # data :  te decoderen data
        # alfabet : vector met alle mogelijke symbolen
        
        # Implementeer vanaf hier
               
        # data_gedecodeerd : gedecodeerde data
        return data_gedecodeerd
        
    # functie die sequentie van bronsymbolen omzet naar macrosymbolen en de relatieve frequenties ervan berekent
    def scalair_naar_vector(bronsymbolen,alfabet_scalair):
        # bronsymbolen : vector met bronsymbolen die omgezet moet worden naar macrosymbolen
        # alfabet_scalair : vector met alle mogelijke bronsymbolen
        
        # Implementeer vanaf hier
       
               
        # macrosymbolen : vector met bronsymbolen omgezet in macrosymbolen
        # alfabet_vector : vector met alle mogelijke macrosymbolen
        # rel_freq : vector met relative frequentie van alle macrosymbolen
        return (macrosymbolen,alfabet_vector,rel_freq)
     
    # functie die sequentie van macrosymbolen omzet naar sequentie van bronsymbolen
    def vector_naar_scalair(macrosymbolen,alfabet_scalair):
        # macrosymbolen : vector met macrosymbolen die omgezet moet worden naar bronsymbolen
        # alfabet_scalair : vector met alle mogelijke bronsymbolen
        
        # Implementeer vanaf hier
        
        # bronsymbolen : vector met macrosymbolen omgezet in bronsymbolen
        return bronsymbolen
    
    # functie die de data sequentie encodeert met Huffman code - niet veranderen
    def Huffman_encodeer(data,dictionary):
        # data : de data die geëncodeerd moet worden
        # dictionary : dictionary met symbolen als key en codewoord als value
        
        N = len(data)
               
        alfabet = dictionary.keys()
        N_symbols = len(alfabet)
        
        output_temp = np.zeros(N,dtype=object)
                
        row_index = np.arange(N)
        
        for symbol in alfabet:
            
            mask = data == symbol
            value = dictionary.get(symbol)
            indices = row_index[mask]
            output_temp[indices] = [value]
        
        output = np.zeros(0,dtype=int)
        output = np.hstack(output_temp)
                       
        return output
   
    # functie die de data sequentie decodeert met Huffman code - niet veranderen
    def Huffman_decodeer(data,boom,alfabet):
        # data : de data die gedecodeerd moet worden
        # boom : matrix met boomstructuur (zie opgave)
        # alfabet : vector met alle mogelijke symbolen
        
        N = len(data)
        M = len(alfabet)
        
        output = []
        idx = 0
        indx_reset = boom.shape[0]
        indx_tree = indx_reset
                
        for idx in range(N):
            next_index = data[idx]
            indx_tree = boom[indx_tree-1,next_index]
            if(indx_tree<=M):
                output.append(indx_tree-1)
                indx_tree = indx_reset
        return alfabet[output]
    
        
        
        
