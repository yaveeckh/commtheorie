def run_broncodering():
    obj = Broncodering()
    
    print('Kwantisatie')
    start = time.time()
    r, q, bronsymbolen = run_kwantisatie()
    print('q = ', q, '\n')
    bronsymbolen_vast = copy.deepcopy(bronsymbolen)
    stop = time.time()
    print('Time: kwantisatie = ', stop - start, '\n')


    start_0 = time.time()
    print('Bron -> Macro')
    alfabet_scalair = q
    macrosymbolen, alfabet_vector, rel_freq = obj.scalair_naar_vector(bronsymbolen, alfabet_scalair)
    entropie = 0.0
    for kans in rel_freq:
        if kans != 0.0:
            entropie -= kans*np.log2(kans)
    print('entropie = ', entropie)
    stop_0 = time.time()
    print('Time: scalair_naar_vector = ', stop_0 - start_0, '\n')
    

    print('Codetabel + dictionary')
    start_1 = time.time()
    index_lijst = [i + 1 for i in range(len(alfabet_vector))]
    dictionary, gem_len, codetabel = obj.maak_codetabel_Huffman(rel_freq, index_lijst)
    print('rel_freq = ', rel_freq, '\n')
    #print('dictionary = ', dictionary, '\n')
    print('gem_len = ', gem_len)
    stop_1 = time.time()
    print('Time: maak_codetabel_Huffman = ', stop_1 - start_1, '\n')


    print('Macro -> binair')
    start_2 = time.time()
    data_binair = obj.Huffman_encodeer(np.array(macrosymbolen), dictionary)
    data_binair_str = ''
    for datapoint in data_binair:
        data_binair_str += datapoint
    stop_2 = time.time()
    print('Time: Huffman_encodeer = ', stop_2 - start_2, '\n')
    with open('data_compansie.txt', 'w') as file_out:
        file_out.write('\n'.join(data_binair))


    print('Binair -> macro')
    start_3 = time.time()
    data_macro = obj.Huffman_decodeer(data_binair_str, np.array(codetabel), np.array(index_lijst))
    stop_3 = time.time()
    print('Time: Huffman_decodeer = ', stop_3 - start_3, '\n')

    
    print('Macro -> Bron')
    start_4 = time.time()
    data_bron = obj.vector_naar_scalair(data_macro, alfabet_scalair)
    stop_4 = time.time()
    print('Time: vector_naar_scalair = ', stop_4 - start_4, '\n')
    data_bron_str = []
    for data_point in data_bron:
        data_bron_str.append(str(data_point))
    with open('Macro>Bron.txt', 'w') as file_out:
        file_out.write('\n'.join(data_bron_str))


    print('Vaste-lengte')
    start_5 = time.time()
    encoded_vast = obj.vaste_lengte_encodeer(bronsymbolen_vast, alfabet_scalair)
    decoded_vast = obj.vaste_lengte_decodeer(encoded_vast, alfabet_scalair)
    stop_5 = time.time()
    print('Time: vaste_lengte_encodeer + decodeer = ', stop_5 - start_5, '\n')

    return 1
