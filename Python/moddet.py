
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import cmath

import pulse


class ModDet():
    def __init__(self):
        pass

    def modulation_detection(self, bitarray, constellatie, T, Ns, f0, alpha, Lf, N0, hch, theta, epsilon=0, phi=0):
        sigma = math.sqrt(N0*Ns/2/T)
        hch_hat = hch* (1+epsilon)
        theta_hat = theta + phi

        a = self.mapper(bitarray, constellatie)
        s = self.moduleer(a, T, Ns, f0, alpha, Lf)
        r = self.kanaal(s, sigma, hch)
        rdemod = self.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
        rdown = self.decimatie(rdemod, Ns, Lf)
        u = self.maak_decisie_variabele(rdown, hch_hat, theta_hat)
        a_estim = self.decisie(u, constellatie)
        bitarray_out = self.demapper(a_estim, constellatie)

        return bitarray_out
    
    def mappingstabel(self, constellatie):
        tabel = []
        if   constellatie == 'BPSK': tabel = np.array([-1,1])
        elif constellatie == '4PSK': tabel = np.array([1,1j,-1,-1j])
        elif constellatie == '4QAM': tabel = np.array([1+1j,-1+1j,-1-1j,1-1j]) * math.sqrt(2)/2
        elif constellatie == '4PAM': tabel = np.array([-3, -1, 1, 3]) * math.sqrt(5)/5
        return tabel

    # functie die bitstring omzet naar complexe symbolen
    def mapper(self, bitstring, constellatie):
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
                    elif bits == '01': a.append(complex(0,1))
                    elif bits == '11': a.append(-1)
                    elif bits == '10': a.append(complex(0,-1))

            elif constellatie == '4QAM':
                for bits in bitvector:
                    if bits == '00': a.append(complex(math.sqrt(2)/2,math.sqrt(2)/2))
                    elif bits == '01': a.append(complex(-math.sqrt(2)/2,math.sqrt(2)/2))
                    elif bits == '11': a.append(complex(-math.sqrt(2)/2,-math.sqrt(2)/2))
                    elif bits == '10': a.append(complex(math.sqrt(2)/2,-math.sqrt(2)/2))
            
            elif constellatie == '4PAM':
                for bits in bitvector:
                    if bits == '00': a.append(-3*math.sqrt(5)/5)
                    elif bits == '01': a.append(-math.sqrt(5)/5)
                    elif bits == '11': a.append(math.sqrt(5)/5)
                    elif bits == '10': a.append(3*math.sqrt(5)/5)
            
        # a: sequentie van data symbolen
        return np.array(a)
    
    # functie die complexe symbolen omzet naar bits
    def demapper(self, a, constellatie):
        # a: sequentie van data symbolen
        # constellatie: ofwel 'BPSK',ofwel '4QAM',ofwel '4PSK',ofwel'4PAM'
        
        # Implementeer vanaf hier
        bitstring = []
        if constellatie == 'BPSK':
            for datasymbol in a:
                if datasymbol == -1: bitstring.append(0)
                elif datasymbol == 1: bitstring.append(1)
        elif constellatie == '4PSK':
            for datasymbol in a:
                if datasymbol == 1: bitstring.extend([0,0])
                elif datasymbol == complex(0,1): bitstring.extend([0,1])
                elif datasymbol == -1: bitstring.extend([1,1])
                elif datasymbol == complex(0,-1): bitstring.extend([1,0])
        elif constellatie == '4QAM':
            for datasymbol in a:
                if datasymbol.real == math.sqrt(2)/2:
                    if datasymbol.imag == math.sqrt(2)/2: bitstring.extend([0,0])
                    elif datasymbol.imag == -math. sqrt(2)/2: bitstring.extend([1,0])
                elif datasymbol.real == -math.sqrt(2)/2:
                    if datasymbol.imag == math.sqrt(2)/2: bitstring.extend([0,1])
                    elif datasymbol.imag == -math. sqrt(2)/2: bitstring.extend([1,1])
        elif constellatie == '4PAM':
            for datasymbol in a:
                if datasymbol == -3*math.sqrt(5)/5: bitstring.extend([0,0])
                elif datasymbol == -math.sqrt(5)/5: bitstring.extend([0,1])
                elif datasymbol == math.sqrt(5)/5: bitstring.extend([1,1])
                elif datasymbol == 3*math.sqrt(5)/5: bitstring.extend([1,0])

        # bitstring : sequentie van bits
        return bitstring
    
    # functie die decisie toepast op u
    def decisie(self,u,constellatie):
        # u: vector met ruizige (complexe) symbolen
        # constellatie: ofwel 'BPSK',ofwel '4QAM',ofwel '4PSK',ofwel'4PAM'
        
        # Implementeer vanaf hier
        mappingstabel = self.mappingstabel(constellatie)
        a_estim = []
        for uk in u:
            distances = abs(mappingstabel - uk)
            index = np.where(distances == np.amin(distances))
            a_estim_k = mappingstabel[index[0][0]]
            a_estim.append(a_estim_k)

        # a_estim : vector met geschatte (complexe) symbolen
        return np.array(a_estim)
    
    # funcie die de decisie variabele aanmaakt
    def maak_decisie_variabele(self,rdown,hch_hat,theta_hat):
        # rdown : vector met het gedecimeerde ontvangen signaal
        # hch_hat : schatting van amplitude van het kanaal
        # theta_hat : schatting van fase van de demodulator
        
        # Implementeer vanaf hier
        u = rdown/hch_hat*np.exp(1j*theta_hat)

        # u : vector met decisie-variabele
        return u
    
    # functie die de modulatie implementeert
    def moduleer(self, a,T,Ns,frequentie,alpha,Lf):
        # a : sequentie van data symbolen 
        # T : symboolperiode in seconden
        # Ns : aantal samples per symbool
        # frequentie : carrier frequentie in Hz
        # alpha : roll-off factor
        # Lf : pulse duur uitgedrukt in aantal symboolintervallen
        
        # Implementeer vanaf hier
        t_vector = np.linspace(-Lf*T, Lf*T, 2*Lf*Ns + 1)
        pulsevector = self.pulse(t_vector, T, alpha)
        
        # plt.plot(t_vector, pulsevector)
        # plt.show()
        # plt.close()
        x = np.zeros((2*Lf + len(a) - 1)*Ns + 1, dtype=np.complex128)

        for i in range(len(a)):
            begin_index = i*Ns
            eind_index = begin_index + 2*Lf*Ns + 1
            x[begin_index:eind_index] += a[i]*pulsevector

        t = np.linspace(0, (2*Lf + len(a) - 1)*T, (2*Lf + len(a) - 1)*Ns + 1)

        c = x*np.exp(1j*2*math.pi*frequentie*t)

        # s : vector met gemoduleerde samples
        s = math.sqrt(2)*c.real
        # plt.plot(t, s)
        # plt.show()
        # plt.close()
        return s
    
    # functie die de demodulatie implementeert
    def demoduleer(self,r,T,Ns,frequentie,alpha,Lf,theta):
        # r : sequentie van ontvangen samples
        # T : symboolperiode in seconden
        # Ns : aantal samples per symbool
        # frequentie : carrier frequentie in Hz
        # alpha : roll-off factor
        # Lf : pulse duur uitgedrukt in aantal symboolintervallen
        # theta : fase van de demodulator
        
        # Implementeer vanaf hier
        Ts = T/Ns
        t = np.array([i*Ts for i in range(0,len(r))])

        rr = math.sqrt(2)*r*np.exp(-1j*(2*math.pi*frequentie*t+ theta))
        t_vector = np.linspace(-Lf*T, Lf*T, 2*Lf*Ns + 1)
        pulsevector = self.pulse(t_vector, T, alpha)
       
        rdemod = Ts * np.convolve(pulsevector,rr) 
        return rdemod
    
    # functie die de pulse aanmaakt - niet veranderen
    def pulse(self,t,T,alpha):
        een = (1-alpha)*np.sinc(t*(1-alpha)/T)
        twee = (alpha)*np.cos(math.pi*(t/T-0.25))*np.sinc(alpha*t/T-0.25)
        drie = (alpha)*np.cos(math.pi*(t/T+0.25))*np.sinc(alpha*t/T+0.25)
        y = 1/np.sqrt(T)*(een+twee+drie)
        return y
    
    # functie die het decimeren implementeert
    def decimatie(self,rdemod,Ns,Lf):
        # rdemod : vector met Ns samples per symbool
        # Ns : aantal samples per symbool
        # Lf : pulse duur uitgedruikt in aantal symboolintervallen
        
        # Implementeer vanaf hier
        rdown = rdemod[2*Lf*Ns:-2*Lf*Ns:Ns] 
        # rdown: vector met 1 sample per symbool 
        return rdown
    
    # funcitie die het AWGN kanaal simuleert
    def kanaal(self,s,sigma,hch):
        # s : ingang van het kanaal
        # sigma : standaard deviatie van de ruis
        # hch : amplitude van het kanaal
    
        # Implementeer vanaf hier
        noise = np.random.normal(0, sigma, len(s))
        
        # r : uitgang van het kanaal
        r = hch*s + noise
        return r
    
