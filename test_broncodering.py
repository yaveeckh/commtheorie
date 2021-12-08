    # TEST

    # Bron -> Macro
    # ['1', '1', '1', '2', '1', '3',...] -> ['11', '12', '13',...] -> ['0', '1', '2',...]
    alfabet = ['1','2']
    stream = '11122122122121222222'
    bronsymbolen = []
    for symbol in stream:
        bronsymbolen.append(symbol)
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet)
    print('macrosymbolen = ', macrosymbolen)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    #print('entropie = ', entropie)
    
    # Codetabel
    # ['0', '1', '2',...] -> [11001, 1111, 000]
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('dictionary = ', dictionary)
    #print('gem_len = ', gem_len)
    print('codetabel = ',codetabel)

    # Macro -> binair : KLOPT
    #data_binair = [obj.Huffman_encodeer(macrosymbolen[i], dictionary) for i in range(len(macrosymbolen))]
    #data_binair = [x[0] for x in data_binair]
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    print('data_binair = ', data_binair)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += datapoint
    # Binair -> macro : KLOPT NIET
    #data_decoded = [obj.Huffman_decodeer([int(char) for char in data_binair[i]], np.array(codetabel), np.array(index_lijst)) for i in range(len(data_binair))]
    #data_decoded = [x[0] for x in data_decoded if x != []]
    data_decoded = obj.Huffman_decodeer(data_binair_str, np.array(codetabel), np.array(index_lijst))
    print('data_decoded = ', data_decoded, '\n')


    # Macro -> Bron
    # ['0', '1', '2',...] -> ['1', '1', '1', '2', '1', '3',...]
    bronsymbolen_nadien = obj.vector_naar_scalair(data_decoded, alfabet)
    print('bronsymbolen = ', bronsymbolen_nadien)

    # Vaste-lengte
    encoded = obj.vaste_lengte_encodeer(stream, alfabet)
    print('bronsymbolen_encoded = ', encoded)
    decoded = obj.vaste_lengte_decodeer(encoded, alfabet)
    print('bronsymbolen_decoded = ', decoded)
