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
def code(node,gemiddeldeLengtes, dictionary, val = ''):
    newVal = val + str(node.code)

    if(node.left):
        dictionary.update(code(node.left, gemiddeldeLengtes, dictionary, newVal)[0])
    if(node.right):
        dictionary.update(code(node.right, gemiddeldeLengtes, dictionary, newVal)[0])
    if(not node.left and not node.right):
        dictionary[node.symbol] = newVal
        gemiddeldeLengtes[node.symbol - 1] = len(newVal)*node.prob
    
    return [dictionary, gemiddeldeLengtes]

class Broncodering():
    def __init__(self):
        pass
    
    # functie die codetabel opsteld voor de Huffmancode
    def  maak_codetabel_Huffman(self, rel_freq, alfabet):
        # rel_freq : vector met relatieve frequenties 
        # alfabet : vector met alle mogelijke symbolen in dezelfde volgorde als rel_freq 
        
        # Implementeer vanaf hier
        M = len(alfabet)
        # dictionary : dictionary met symbolen als key en codewoord als value
        dictionary = dict()
        # boom : matrix met boomstructuur (zie opgave)
        nodes = []
        for i, symbol in enumerate(alfabet):
            if(rel_freq[i] != 0.0):
                nodes.append(Node(rel_freq[i], symbol))

        boom = [[0, 0] for _ in range(M)]
        counter = M
        nodes = sorted(nodes, key=lambda x: x.prob)

        while len(nodes) > 1:
            
            right = nodes[0]
            left = nodes[1]

            left.code = 0
            right.code = 1

            counter += 1
            newNode = Node(left.prob + right.prob, counter, left, right)

            boom.append([left.symbol, right.symbol])

            # Gebruikte nodes om samen te voegen, nu verwijderen
            nodes.remove(left)
            nodes.remove(right)

            # Node op de juiste positie toevoegen
            if nodes:
                index = 0
                while(newNode.prob > nodes[index].prob):
                    if index == len(nodes) - 1:
                        break
                    index += 1
                if index == len(nodes) - 1 and newNode.prob > nodes[index].prob:
                    nodes.append(newNode)
                else:
                    nodes.insert(index, newNode)
            else:
                nodes.append(newNode)

        gemiddeldeLengtes = [0 for _ in range(len(rel_freq))]
        result = code(nodes[0], gemiddeldeLengtes, dictionary)
        dictionary = result[0]
        gemiddeldeLengtes = result[1]

        gem_len = 0
        for value in gemiddeldeLengtes:
            gem_len += value

        return (dictionary,gem_len,boom)

    # functie voor het encoderen met vaste-lengte code
    def vaste_lengte_encodeer(self, data,alfabet):
        # data : de data die geencodeerd moet worden (lijst?)
        # alfabet : vector met alle mogelijke symbolen

        # Implementeer vanaf hier
        lengte = math.ceil(np.log2(len(alfabet)))
        data_geencodeerd = []
        for datapoint in data:
            data_geencodeerd.append(bin(alfabet.index(datapoint))[2:].zfill(lengte))
        # data_gecodeerd : de geencodeerde data
        return data_geencodeerd
       
    # functie voor het decoderen met vaste-lengte code
    def vaste_lengte_decodeer(self, data,alfabet):
        # data :  te decoderen data  (np.array)
        # alfabet : vector met alle mogelijke symbolen
        
        data_matrix = np.reshape(data, (len(data)//int(np.log2(len(alfabet))),int(np.log2(len(alfabet)))))
        data_gedecodeerd = []
        for row in data_matrix:
            bitstring = ''
            for bit in row:
                bitstring += str(bit)
            data_gedecodeerd.append(alfabet[int(str(bitstring), 2)])

        # data_gedecodeerd : gedecodeerde data
        return data_gedecodeerd
        
    # functie die sequentie van bronsymbolen omzet naar macrosymbolen en de relatieve frequenties ervan berekent
    def scalair_naar_vector(self, bronsymbolen, alfabet_scalair):
        # bronsymbolen : vector met bronsymbolen die omgezet moet worden naar macrosymbolen
        # alfabet_scalair : vector met alle mogelijke bronsymbolen ['1', '2', '3']

        # Implementeer vanaf hier

        M = len(alfabet_scalair)
        alfabet_vector = []
        for first in alfabet_scalair:
            for second in alfabet_scalair:
                alfabet_vector.append(str(first) + str(second))

        macrosymbolen = []
        rel_freq = [0 for _ in range(len(alfabet_vector))]
        aantal_macrosymbolen = 0
        while len(bronsymbolen) > 1:
            aantal_macrosymbolen += 1
            index = alfabet_vector.index(str(bronsymbolen[0]) + str(bronsymbolen[1]))
            macrosymbolen.append(index+1)
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
    def vector_naar_scalair(self, macrosymbolen,alfabet_scalair):
        # macrosymbolen : vector met macrosymbolen die omgezet moet worden naar bronsymbolen
        # alfabet_scalair : vector met alle mogelijke bronsymbolen ['1', '2', '3']
        
        # Implementeer vanaf hier
        bronsymbolen = []
        for macro in macrosymbolen:
            bronsymbolen.append(alfabet_scalair[(int(macro) - 1)//len(alfabet_scalair)])
            bronsymbolen.append(alfabet_scalair[(int(macro) - 1)%len(alfabet_scalair)])
            
        # bronsymbolen : vector met macrosymbolen omgezet in bronsymbolen
        return bronsymbolen
    
    # functie die de data sequentie encodeert met Huffman code - niet veranderen
    def Huffman_encodeer(self, data,dictionary):
        # data : de data die geencodeerd moet worden
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

        #datatype = np.array
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
            indx_tree = boom[indx_tree-1, next_index]
            if(indx_tree<=M):
                output.append(indx_tree-1)
                indx_tree = indx_reset
        #datatype = np.array
        return alfabet[output]

