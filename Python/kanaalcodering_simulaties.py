
from kanaalcodering import Kanaalcodering
import numpy as np
from matplotlib import pyplot as plt
import copy
obj = Kanaalcodering()

#--Simulatie van de blokccode zonder ARQ--
def simuleer_blokcode(n, p):
    #Genereer random input
    rand = np.reshape(np.random.randint(0,2, n), (n//10, 10))
    #Encodeer rand
    encoded = np.array([obj.encodeer_uitwendig(woord) for woord in rand])
    #Simuleer een kanaal met p=0.05
    encoded_ch = obj.kanaal_simulatie(encoded, p)
    #Decodeer
    decoded = np.array([obj.decodeer_uitwendig(woord, True)[0] for woord in encoded_ch])
    #Tel fouten
    fouten = 0
    for i in range(len(rand)):
        if np.any(rand[i] != decoded[i]):
            fouten += 1
    pe = fouten/(n//10)
    
    return pe

def run_simulation_1():
    p_array = np.arange(0.001, 0.5, 0.01)
    pe = []
    for p in p_array:
        pe += [simuleer_blokcode(100000, p)]
    return pe

    
def plot1():
    # Plot simulatie 1
    x = np.arange(0.001, 0.5, 0.01)
    y = [0.0001, 0.0109, 0.0327, 0.0656, 0.1065, 0.1586, 0.2048, 0.2655, 0.3148, 0.3687, 0.4161, 0.4712, 0.5194,
     0.5633, 0.6121, 0.6443, 0.69, 0.7206, 0.7439, 0.7755, 0.799, 0.8253, 0.8526, 0.8653, 0.879, 0.8967,
     0.9134, 0.9201, 0.9315, 0.9444, 0.9546, 0.9591, 0.9633, 0.972, 0.9741, 0.9778, 0.9833, 0.9867, 0.99,
     0.9905, 0.9921, 0.9924, 0.9944, 0.995, 0.9961, 0.9966, 0.9988, 0.9991, 0.9984, 0.9978]
    y2 = lambda p: 1 - np.power((1-p),14) - 14 *p*np.power(1-p,13) - np.power(p,2) * np.power((1-p),12)
    plt.plot(x, y, linestyle='dashed')
    plt.plot(x, y2(x))
    plt.axvline(x=0.05, color='r')
    plt.axvline(x=0.0031, color='g')
    plt.xlabel("p")
    plt.ylabel("p_e")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(['Experimenteel', 'Analytisch'])
    plt.grid(True)
    plt.savefig("kc_plots/p_e1.png")
    plt.close()
    return

# --Simulatie van het eerste retransmissie protocol--

def simuleer_protocol1(n, p, T_max = 6):

    #N = aantal verstuurde bits
    N = 0

    #Maak random input 
    rand_grouped = np.resize(np.random.randint(0,2, n), (n//10,10))

    #Encodeer de random input
    encoded = obj.kanaalencodering_1(rand_grouped)
    N += len(encoded.flatten())

    #Simuleer het kanaal
    encoded_ch = obj.kanaal_simulatie(encoded,p)

    #Decodeer
    decoded = obj.kanaaldecodering_1(encoded_ch, True)
    decoded_bits = decoded[0]
    decoded_corrected = copy.deepcopy(decoded[0])
    decoded_fouten = decoded[1]

    T = 0

    if(decoded_fouten != []):
        #Zolang het aantal fouten > 0 en het aantal transities zijn maximum nog niet berijkt heeft, doe retransmissie
        while(T < T_max and len(decoded_fouten) > 0):
            #Volledige decodering op laatste transitie
            ARQ = False if T == T_max-1 else True 

            #print(f'---Retransmissie {T}---')

            #Stel het ratransmissie pak samen
            retrans_pack = np.array([encoded[i] for i in decoded_fouten])
            N += len(retrans_pack.flatten())

            #Simuleer het kanaal voor retransmissie
            retransmitted = obj.kanaal_simulatie(retrans_pack,p)

            r_decoded= obj.kanaaldecodering_1(retransmitted, ARQ=True)
            r_decoded_bits = r_decoded[0]
            r_fouten = r_decoded[1]

            #Zet fouten om naar index in gedecodeerde matrix
            nieuwe_fouten = [decoded_fouten[i] for i in r_fouten]
 
            if T < T_max-1:
                #Verwijder en pas de juist gedecodeerde retransmissies aan in de fouten lijst

                to_remove = []
                
                for index, row in enumerate(decoded_fouten):
                    if row not in nieuwe_fouten:
                        decoded_corrected[row] = r_decoded_bits[index]
                        to_remove.append(row)
                
                for i in to_remove: decoded_fouten.remove(i)
                
            else:
                #Volledige decodering van laatste transmissie

                for i, row in enumerate(decoded_fouten):
                    decoded_corrected[row] = r_decoded_bits[i]
            
            T += 1

    #Tel fouten
    fouten = 0

    for idx,i in enumerate(rand_grouped):
        if np.any(rand_grouped[idx] != decoded_corrected[idx]):
            fouten += 1

    pe = fouten/(n/10)

    return pe, n/N

def run_simulation_protocol1():
    T_max = range(16)
    n = 100000
    p = 0.05

    pe = []
    deb = []
    for i in T_max:
        s = simuleer_protocol1(n, p, i)
        pe += [s[0]]
        deb += [s[1]]

    return pe, deb

def plot_protocol1():
    x = range(16)
    pe = [0.402, 0.2107, 0.105, 0.0578, 0.0293, 0.0191, 0.0111, 0.0091, 0.0065, 0.0067, 
     0.0065, 0.0053, 0.0045, 0.0064, 0.0051, 0.0048]
    deb = [0.7142857142857143, 0.4721613658683992, 0.4042135217507296, 0.3735217875258664,
     0.3660750893223218, 0.3518822179839964, 0.3549069433994407, 0.3534842946927868,
     0.350071414568572, 0.3517782389981356, 0.35278595065230123, 0.35082795397137245,
     0.3507762678808203, 0.352803375622698, 0.34719570032844715, 0.34778737670937493]
    
    plt.plot(x, pe)
    plt.xlabel("T_max")
    plt.ylabel("p_e")
    plt.yscale("log")
    plt.hlines(0.001, 0, 15, color="r")
    plt.grid(True)
    plt.savefig("kc_plots/p_e_protocol1.png")
    plt.close()

    plt.plot(x, deb)
    #plt.axvline(x=0.05, color='r')
    plt.xlabel("T_max")
    plt.ylabel("Gemiddeld informatiedebiet")
    plt.grid(True)
    plt.savefig("kc_plots/deb_protocol1.png")
    plt.close()
    return

# --Simulatie van het tweede retransmissie protocol a--
def simuleer_protocol2a(n, p  ,T_max, g_x=[1,1,0,1,0,1]):

    #Verstuurd aantal bits N
    N = 0

    #Genereer random input
    rand_grouped = np.resize(np.random.randint(0,2, n), (n//5,5))

    #Encodeer
    encoded = obj.kanaalencodering_2(rand_grouped, g_x)
    N += len(encoded.flatten())
    
    #Simuleer kanaal
    encoded_ch = obj.kanaal_simulatie(encoded,p)

    #Decodeer wat door het kanaal komt
    decoded = obj.kanaaldecodering_2(encoded_ch, g_x)
    decoded_bits = decoded[0]
    decoded_fouten = decoded[1]
    decoded_corrected = copy.deepcopy(decoded[0])

    T = 0
    if(decoded_fouten != []):
        while(T < T_max and len(decoded_fouten) > 0):
            #Stel retrans pak samen
            retrans_pack = np.array([encoded[i] for i in decoded_fouten])
            N += len(retrans_pack.flatten())

            #Simuleer kanaal
            retransmitted = obj.kanaal_simulatie(retrans_pack,p)
            

            #Decode
            r_decoded = obj.kanaaldecodering_2(retransmitted, g_x)
            r_bits = r_decoded[0]
            r_fouten = r_decoded[1]

            nieuwe_fouten = [decoded_fouten[i] for i in r_fouten]

            to_remove = []
            for index, row in enumerate(decoded_fouten):
                if row not in nieuwe_fouten:
                    decoded_corrected[row] = r_bits[index]
                    to_remove.append(row)
            
            for i in to_remove: decoded_fouten.remove(i)
                
            T += 1

    fouten = 0
    for idx,i in enumerate(rand_grouped):
        if np.any(rand_grouped[idx] != decoded_corrected[idx]):
            fouten += 1
    pe = fouten/(n/5)

    return pe, n/N

def run_simulation2a():
    T_max = range(7)
    n = 100000
    p = 0.05

    pe = []
    deb = []

    for i in T_max:
        s = simuleer_protocol2a(n, p, i)
        pe += [s[0]]
        deb += [s[1]]

    return pe, deb

def plot_protocol2a():
    x = range(7)
    pe = [0.11385, 0.01805, 0.00345, 0.0009, 0.0004, 0.0004, 0.00045]
    deb = [0.35714285714285715, 0.31041054899209697, 0.30452153576300917, 
    0.3044047365377005, 0.30317729808391947, 0.3033446784243064, 0.3030615275513235]

    ben = lambda i : np.power(0.1516,(np.add(i,1)))
    plt.plot(x, pe)
    plt.plot(x, ben(x))
    plt.legend(['Experimenteel', 'p_out^(T_max + 1)'])
    plt.xlabel("T_max")
    plt.ylabel("p_e")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("kc_plots/p_e_protocol2a.png")
    plt.close()

    plt.plot(x, deb)
    #plt.axvline(x=0.05, color='r')
    plt.xlabel("T_max")
    plt.ylabel("Gemiddeld informatiedebiet")
    plt.grid(True)
    plt.savefig("kc_plots/deb_protocol2a.png")
    plt.close()
    return

# --Simulatie van het tweede retransmissie protocol b--
def simuleer_protocol2b(n, p  ,T_max, g_x=[1,1,0,0,1,1,0,1,1]):

    #Verstuurd aantal bits N
    N = 0

    #Genereer random input
    rand_grouped = np.resize(np.random.randint(0,2, n), (n//2,2))

    #Encodeer
    encoded = obj.kanaalencodering_2(rand_grouped, g_x)
    N += len(encoded.flatten())
    
    #Simuleer kanaal
    encoded_ch = obj.kanaal_simulatie(encoded,p)

    #Decodeer wat door het kanaal komt
    decoded = obj.kanaaldecodering_2(encoded_ch, g_x)
    decoded_bits = decoded[0]
    decoded_fouten = decoded[1]
    decoded_corrected = copy.deepcopy(decoded[0])

    T = 0
    if(decoded_fouten != []):
        while(T < T_max and len(decoded_fouten) > 0):
            #Stel retrans pak samen
            retrans_pack = np.array([encoded[i] for i in decoded_fouten])
            N += len(retrans_pack.flatten())

            #Simuleer kanaal
            retransmitted = obj.kanaal_simulatie(retrans_pack,p)
            
            #Decode
            r_decoded = obj.kanaaldecodering_2(retransmitted, g_x)
            r_bits = r_decoded[0]
            r_fouten = r_decoded[1]

            nieuwe_fouten = [decoded_fouten[i] for i in r_fouten]

            to_remove = []
            for index, row in enumerate(decoded_fouten):
                if row not in nieuwe_fouten:
                    decoded_corrected[row] = r_bits[index]
                    to_remove.append(row)
            
            for i in to_remove: decoded_fouten.remove(i)
                
            T += 1

    fouten = 0
    for idx,i in enumerate(rand_grouped):
        if np.any(rand_grouped[idx] != decoded_corrected[idx]):
            fouten += 1
    pe = fouten/(n/2)

    return pe, n/N

def run_simulation2b():
    T_max = range(7)
    n = 100000
    p = 0.05

    pe = []
    deb = []

    for i in T_max:
        s = simuleer_protocol2b(n, p, i)
        pe += [s[0]]
        deb += [s[1]]

    return pe, deb

def plot_protocol2b():
    x = range(7)

    pe = [0.06832, 0.01006, 0.00132, 0.00024, 4e-05, 2e-05, 2e-05]
    deb = [0.14285714285714285, 0.12394768416146913, 0.12191256430887767, 0.12128121475264696,
           0.12131005150824788, 0.12116806009935781, 0.12067063914410729]
    
    plt.plot(x, pe)
    plt.xlabel("T_max")
    plt.ylabel("p_e")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("kc_plots/p_e_protocol2b.png")
    plt.close()

    plt.plot(x, deb)
    #plt.axvline(x=0.05, color='r')
    plt.xlabel("T_max")
    plt.ylabel("Gemiddeld informatiedebiet")
    plt.grid(True)
    plt.savefig("kc_plots/deb_protocol2b.png")
    plt.close()
    return
