def run_broncodering():
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

    return data_binair_str
