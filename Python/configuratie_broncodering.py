# Vaste-lengte encodering
def run_broncodering_12():
    obj = Broncodering()
    
    print('Kwantisatie\n')
    r, q, bronsymbolen = run_kwantisatie()
    r = r.tolist()
    q = q.tolist()

    print('Vaste-lengte\n')
    data_encoded = obj.vaste_lengte_encodeer(bronsymbolen, q)
    data_encoded_str = ''
    for bitstring in data_encoded:
        for bit in bitstring:
            data_encoded_str += bit
            
    data_encoded_lijst = []
    for bit in data_encoded_str:
        data_encoded_lijst.append(bit)

    return data_encoded_lijst

# Scalaire Huffmancodering
def run_broncodering_345():
    obj = Broncodering()
    
    print('Kwantisatie\n')
    r, q, bronsymbolen = run_kwantisatie()
    r = r.tolist()
    q = q.tolist()


    print('rel_freq')
    alfabet_scalair = q
    rel_freq = [0 for _ in range(len(alfabet_scalair))]
    aantal_symbolen = 0
    while len(bronsymbolen) > 1:
        aantal_symbolen += 1
        index = alfabet_scalair.index(bronsymbolen[0])
        rel_freq[index] += 1
        del bronsymbolen[0]

    for index, element in enumerate(rel_freq):
        rel_freq[index] = element / aantal_symbolen

    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie, '\n')
    

    print('Codetabel + dictionary')
    index_lijst = [i + 1 for i in range(len(alfabet_scalair))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('gem_len = ', gem_len, '\n')


    print('Huffman_encodeer\n')
    data_binair = obj.Huffman_encodeer(np.array(bronsymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += str(datapoint)
    
    data_binair_lijst = []
    for bit in data_binair_str:
        data_binair_lijst.append(bit)

    return data_binair_lijst

# Vectoriele Huffmancodering
def run_broncodering_6():
    obj = Broncodering()
    
    print('Kwantisatie')
    r, q, bronsymbolen = run_kwantisatie()
    r = r.tolist()
    q = q.tolist()


    print('Bron -> Macro')
    alfabet_scalair = q
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet_scalair)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    

    print('Codetabel + dictionary')
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('gem_len = ', gem_len)


    print('Macro -> binair')
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += datapoint

    data_binair_lijst = []
    for bit in data_binair_str:
        data_binair_lijst.append(bit)

    return data_binair_lijst

