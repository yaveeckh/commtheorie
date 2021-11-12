
import numpy as np
from numpy.core.fromnumeric import sort
from numpy.core.numeric import True_
import numpy.matlib
import math

# Node in the Huffman tree
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

# Helper function to make codes for the nodes
def code(node, dictionary, val = ''):
    newVal = val + str(node.code)

    if(node.left):
        dictionary.update(code(node.left,dictionary, newVal))
    if(node.right):
        dictionary.update(code(node.right,dictionary, newVal))
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
        # boom : matrix met boomstructuur (zie opgave)
        nodes = []
        for i, symbol in enumerate(alfabet):
            nodes.append(Node(rel_freq[i], symbol))

        boom = [[0, 0] for _ in range(M)]
        while len(nodes) > 1:
            nodes = sorted(nodes, key=lambda x: x.prob)

            right = nodes[0]
            left = nodes[1]

            left.code = 1
            right.code = 0

            newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)

            boom.append([left.symbol, right.symbol])

            nodes.remove(left)
            nodes.remove(right)
            nodes.append(newNode)
        
        dictionary = code(nodes[0], dictionary)
        # gem_len : gemiddelde codewoordlengte
        gem_len = 0
        for value in dictionary.values():
            gem_len += len(value)
        gem_len /= M

        return (dictionary,gem_len,boom)

    # functie voor het encoderen met vaste-lengte code
    def vaste_lengte_encodeer(self, data,alfabet):
        # data : de data die geëncodeerd moet worden
        # alfabet : vector met alle mogelijke symbolen
        
        # Implementeer vanaf hier
               
        # data_gecodeerd : de geëncodeerde data
        return data_geencodeerd
       
    # functie voor het decoderen met vaste-lengte code
    #def vaste_lengte_decodeer(self, data,alfabet):
        # data :  te decoderen data
        # alfabet : vector met alle mogelijke symbolen
        
        # Implementeer vanaf hier
               
        # data_gedecodeerd : gedecodeerde data
        #return data_gedecodeerd
        
    # functie die sequentie van bronsymbolen omzet naar macrosymbolen en de relatieve frequenties ervan berekent
    def scalair_naar_vector(self, bronsymbolen, alfabet_scalair):
        # bronsymbolen : vector met bronsymbolen die omgezet moet worden naar macrosymbolen
        # alfabet_scalair : vector met alle mogelijke bronsymbolen ['1', '2', '3']

        # Implementeer vanaf hier
        M = len(alfabet_scalair)
        alfabet_vector = []
        for first in alfabet_scalair:
            for second in alfabet_scalair:
                alfabet_vector.append(first + second)

        macrosymbolen = []
        rel_freq = [0 for _ in range(len(alfabet_vector))]
        aantal_macrosymbolen = 0
        while len(bronsymbolen) > 1:
            aantal_macrosymbolen += 1
            index = alfabet_vector.index(bronsymbolen[0] + bronsymbolen[1])
            macrosymbolen.append(str(index))
            rel_freq[index] += 1
            del bronsymbolen[0]
            del bronsymbolen[0]
        
        for index, element in enumerate(rel_freq):
            rel_freq[index] = element / aantal_macrosymbolen

        # macrosymbolen : vector met bronsymbolen omgezet in macrosymbolen ['11', '12', '13',..., '33']
        # alfabet_vector : vector met alle mogelijke macrosymbolen
        # rel_freq : vector met relative frequentie van alle macrosymbolen
        return (macrosymbolen,alfabet_vector, rel_freq)
     
    # functie die sequentie van macrosymbolen omzet naar sequentie van bronsymbolen
    def vector_naar_scalair(self, macrosymbolen,alfabet_vector):
        # macrosymbolen : vector met macrosymbolen die omgezet moet worden naar bronsymbolen
        # alfabet_vector : vector met alle mogelijke macrosymbolen
        
        # Implementeer vanaf hier
        bronsymbolen = []
        for macro in macrosymbolen:
            bronsymbolen.append(alfabet_vector[int(macro)])
            
        # bronsymbolen : vector met macrosymbolen omgezet in bronsymbolen
        return bronsymbolen
    
    # functie die de data sequentie encodeert met Huffman code - niet veranderen
    def Huffman_encodeer(self, data,dictionary):
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
    def Huffman_decodeer(self, data,boom,alfabet):
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
