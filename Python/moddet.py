
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math

class ModDet():
    def __init__():
        pass
    
    # funcite die bitstring omzet naar complexe symbolen
    def mapper(bitstring, constellatie):
        # bitstring : sequentie van bits
        # constellatie: ofwel 'BPSK',ofwel '4QAM',ofwel '4PSK',ofwel'4PAM'
                
        # Implementeer vanaf hier
        a = []

        if constellatie == 'BPSK':
            for bit in bitstring:
                a.append(-1 if bit == 0 else 1)

        else:
            bitvector = []
            for i in range(0, len(bitstring), 2):
                bitvector.append("".join(str(bit) for bit in bitstring[i:i+2]))
            
            if constellatie == '4PSK':
                for bits in bitvector:
                    if bits == '00': a.append(1)
                    if bits == '01': a.append(complex(0,1))
                    if bits == '11': a.append(-1)
                    if bits == '10': a.append(complex(0,-1))

            elif constellatie == '4QAM':
                for bits in bitvector:
                    if bits == '00': a.append(complex(math.sqrt(2)/2,math.sqrt(2)/2))
                    if bits == '01': a.append(complex(-math.sqrt(2)/2,math.sqrt(2)/2))
                    if bits == '11': a.append(complex(-math.sqrt(2)/2,-math.sqrt(2)/2))
                    if bits == '10': a.append(complex(math.sqrt(2)/2,-math.sqrt(2)/2))
            
            elif constellatie == '4PAM':
                for bits in bitvector:
                    if bits == '00': a.append(-3*math.sqrt(5)/5)
                    if bits == '01': a.append(-math.sqrt(5)/5)
                    if bits == '11': a.append(math.sqrt(5)/5)
                    if bits == '10': a.append(3*math.sqrt(5)/5)

        # a: sequentie van data symbolen
        return a
    
    # functie die complexe symbolen omzet naar bits
    def demapper(a, constellatie):
        # a: sequentie van data symbolen
        # constellatie: ofwel 'BPSK',ofwel '4QAM',ofwel '4PSK',ofwel'4PAM'
        
        # Implementeer vanaf hier
        
        # bitstring : sequentie van bits
        return bitstring
    
    # functie die decisie toepast op u
    def decisie(u,constellatie):
        # u: vector met ruizige (complexe) symbolen
        # constellatie: ofwel 'BPSK',ofwel '4QAM',ofwel '4PSK',ofwel'4PAM'
        
        # Implementeer vanaf hier
        
        
        # a_estim : vector met geschatte (complexe) symbolen
        return a_estim
    
    # funcie die de decisie variabele aanmaakt
    def maak_decisie_variabele(rdown,hch_hat,theta_hat):
        # rdown : vector met het gedecimeerde ontvangen signaal
        # hch_hat : schatting van amplitude van het kanaal
        # theta_hat : schatting van fase van de demodulator
        
        # Implementeer vanaf hier
                
        # u : vector met decisie-variabele
        return u
    
    # functie die de modulatie implementeert
    def moduleer(a,T,Ns,frequentie,alpha,Lf):
        # a : sequentie van data symbolen 
        # T : symboolperiode in seconden
        # Ns : aantal samples per symbool
        # frequentie : carrier frequentie in Hz
        # alpha : roll-off factor
        # Lf : pulse duur uitgedrukt in aantal symboolintervallen
        
        # Implementeer vanaf hier
        
        # s : vector met gemoduleerde samples
        return s
    
    # functie die de demodulatie implementeert
    def demoduleer(r,T,Ns,frequentie,alpha,Lf,theta):
        # r : sequentie van ontvangen samples
        # T : symboolperiode in seconden
        # Ns : aantal samples per symbool
        # frequentie : carrier frequentie in Hz
        # alpha : roll-off factor
        # Lf : pulse duur uitgedrukt in aantal symboolintervallen
        # theta : fase van de demodulator
        
        # Implementeer vanaf hier
               
               
        
        return rdemod
    
    # functie die de pulse aanmaakt - niet veranderen
    def pulse(t,T,alpha):
        een = (1-alpha)*np.sinc(t*(1-alpha)/T)
        twee = (alpha)*np.cos(math.pi*(t/T-0.25))*np.sinc(alpha*t/T-0.25)
        drie = (alpha)*np.cos(math.pi*(t/T+0.25))*np.sinc(alpha*t/T+0.25)
        y = 1/np.sqrt(T)*(een+twee+drie);
        return y
    
    # functie die het decimeren implementeert
    def decimatie(rdemod,Ns,Lf):
        # rdemod : vector met Ns samples per symbool
        # Ns : aantal samples per symbool
        # Lf : pulse duur uitgedruikt in aantal symboolintervallen
        s
        # Implementeer vanaf hier
        
        # rdown: vector met 1 sample per symbool 
        return rdown
    
    # funcitie die het AWGN kanaal simuleert
    def kanaal(s,sigma,hch):
        # s : ingang van het kanaal
        # sigma : standaard deviatie van de ruis
        # hch : amplitude van het kanaal
    
        # Implementeer vanaf hier
        
        # r : uitgang van het kanaal
        return r
    
