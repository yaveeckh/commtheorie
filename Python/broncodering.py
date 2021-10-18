
import numpy as np
import numpy.matlib
import math


class Broncodering():
    def __init__():
        pass
        
    # functie die codetabel opsteld voor de Huffmancode
    def  maak_codetabel_Huffman(rel_freq,alfabet):
        # rel_freq : vector met relatieve frequenties 
        # alfabet : vector met alle mogelijke symbolen in dezelfde volgorde als rel_freq 
        
        # Implementeer vanaf hier
            
        # dictionary : dictionary met symbolen als key en codewoord als value
        # gem_len : gemiddelde codewoordlengte
        # boom : matrix met boomstructuur (zie opgave)
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
    
        
        
        
