import wave
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate._ivp.radau import P

from pulse import pulse
from playsound import playsound



class Kwantisatie():
    
    # constructor - niet veranderen
    def __init__(self,bool_play=0): 
        self.bestand_name ="input.wav"
        if bool_play:
            playsound(self.bestand_name)
                
        wave_object = wave.open(self.bestand_name,'rb')
        number_frames = wave_object.getnframes()
        self.fs = wave_object.getframerate()
        
        wave_all = wave_object.readframes(number_frames)
        
        data = np.zeros((number_frames))
        
        for i in range(0,number_frames):
            temp = struct.unpack("<h",wave_all[2*i:2*i+2])
            data[i] = temp[0]
            
        data = data/(2**15)
        self.data = data
        wave_object.close()
        self.f_u = self.make_f_u(1000,5000,25)
        
    # functie om data weg te schrijven en music af te spelen - niet veranderen
    def save_and_play_music(self,datastream,wav_file,bool_play = 1):
        # datastream: np array
        datastream = datastream.flatten()
        wave_object = wave.open(wav_file,'wb')
        wave_object.setnchannels(1)
        wave_object.setsampwidth(2)
        wave_object.setframerate(self.fs)
        n_length = len(datastream)
        for i in range(n_length):
            value = datastream[i]
            data = np.round(value*2**15)
            data_int = data.astype(int)
            data_packed = struct.pack('<h',data_int)
            wave_object.writeframesraw(data_packed)
        wave_object.close()
        if bool_play:
            playsound(wav_file)
    
    # functie om anonieme functie te maken - niet veranderen        
    def make_f_u(self,Fs1,Fs2,factor_T):
        
        T = factor_T/Fs1
        edges = np.arange(-1,1,1/Fs1)
        edges = np.append(edges,1)
        
        val,bin_edges = np.histogram(self.data,edges,density=True)
        
        
        # filter histograam 
        K = 30;
        Lt = K*T;
        
        t = np.arange(-Lt,Lt,1/Fs1)
        t = np.append(t,Lt)
        p = pulse(t,T,0.2)/np.sqrt(T);
        
                
        val_filtered = np.convolve(val,p)/Fs1
        val_filtered = val_filtered[int(Lt*Fs1):-int(Lt*Fs1)]
        scaling = Fs1/np.sum(val_filtered)
        val_filtered = val_filtered*scaling
     
        # --- upscale ---
        Ns = int(Fs2/Fs1);
        val_up = np.zeros(len(val_filtered)*Ns)
        val_up[::Ns] = val_filtered # voeg nullen toe
        
        # maak pulse
        T = 1/Fs1
        K = 30;
        Lt = K*T;
        t = np.arange(-Lt,Lt,1/Fs2)
        t = np.append(t,Lt)
        p = pulse(t,T,0.2)/np.sqrt(T);
        
        val_up = np.convolve(val_up,p)/Fs2;
        val_up = val_up[int(Lt*Fs2):-int(Lt*Fs2)]
        scaling = Fs2/np.sum(val_up)
        val_up = val_up*scaling
        
        # linear interpolation
        edges = np.arange(-1,1,1/Fs2)
        edges = np.append(edges,1)
        
        pos = (edges[1:]+edges[:-1])/2
        pos = np.insert(pos,0,pos[0]-1/Fs2)
        pos = np.append(pos,pos[-1]+1/Fs2)
        
        val_up[-1] = 0;
        val_up[0] = 0;
        val_up = np.insert(val_up,0,0)
        val_up = np.append(val_up,0)
        
        Y1 = val_up[:-1]
        Y2 = val_up[1:]
        X1 = pos[:-1]
        X2 = pos[1:]
        
        rico = (Y2-Y1)/(X2-X1)
        
        Lval_up = len(val_up)-1;
        index = lambda x : min(max(math.floor((x+1)*Fs2+1/2),0),Lval_up-1)
        ftemp = lambda x : (min(max(-1,x),1) - X1[index(x)])*rico[index(x)] + Y1[index(x)]
        
        scaling,error = integrate.quad(ftemp,-1,1)
        
        f = lambda x : ftemp(x)/scaling
        return f
    
    # functie om inverse van een functie numeriek te bepalen - niet veranderen
    def inverse(g,Yvec):
        Uvec = np.zeros(len(Yvec))
        eps = 1e-6
        for i in range(len(Yvec)):
            y = Yvec[i]
            a = -1
            g_a = g(a)
            b = 1
            g_b = g(b)
            nit = 1
            while abs(g_a-y)>eps and abs(g_b-y)>eps and nit<1000:
                u_test = (a+b)/2
                g_test = g(u_test)
                if(g_test <= y):
                    a = u_test
                    g_a = g_test
                else:
                    b = u_test
                    b_b = g_test
                nit = nit+1
            if(abs(g_a-y)<=eps):
                Uvec[i] = a
            else:
                Uvec[i] = b
        return Uvec
    
    # functie om distributie en genormaliseerd histogram te plotten
    def plot_distributie(self):
        data = self.data # originele monsterwaarden
        f_u = self.f_u # w.d.f. - anonieme functie
        
        # Implementeer vanaf hier
        u = np.linspace(-1, 1, 200)
        fu = np.vectorize(f_u)
        y = fu(u)
        plt.plot(u,y)
        plt.hist(data, density=True)
        plt.savefig('fu.png')
        plt.show()
        
    
    def sigma_gr(self, delta, M, f_u):
        sigma_gr = 0.0
        for i in range(1, M+1):
            sigma_gr += float(integrate.quad(lambda u: ((i - (M+1)/2)*delta - u)**2 *f_u(u), (i - (M+1)/2)*delta - delta/2, (i - (M+1)/2)*delta + delta/2)[0])
        return sigma_gr

    def sigma_ol(self, delta, M, f_u):
            sigma_ol = integrate.quad(lambda u: ((1 - (M+1)/2)*delta - u)**2 *f_u(u), -np.Inf, (1 - (M+1)/2)*delta - delta/2)[0]
            sigma_ol += integrate.quad(lambda u: ((M - (M+1)/2)*delta - u)**2 *f_u(u), (M - (M+1)/2)*delta + delta/2, np.Inf)[0]
            return sigma_ol

    def sigma(self, delta, M, f_u):
        return self.sigma_gr(delta, M, f_u) + self.sigma_ol(delta, M, f_u)

    # functie om de optimale uniforme kwantisator te bepalen
    # M : aantal reconstructieniveaus
    def bepaal_optimale_lineaire_kwantisator(self,M):
        
        f_u = self.f_u # w.d.f. - anonieme functie
        
        # Implementeer vanaf hier

        # Blauwe plot
        sigmagr = np.vectorize(self.sigma_gr)
        delta = np.linspace(0, 0.4, 100)
        y = sigmagr(delta, M, f_u)
        plt.plot(delta,y)

        # Oranje plot
        sigmaol = np.vectorize(self.sigma_ol)
        delta_2 = np.linspace(0, 0.4, 100)
        y_2 = sigmaol(delta, M, f_u)
        plt.plot(delta_2,y_2)

        # Groene plot
        sigma_vect = np.vectorize(self.sigma)
        delta_3 = np.linspace(0, 0.4, 200)
        y_3 = sigma_vect(delta_3, M, f_u)
        plt.plot(delta_3,y_3)
        plt.savefig('sigma.png')

        # delta_opt : optimale stapgrootte
        delta_opt = delta_3[np.where(y_3 == min(y_3))][0]

        # GKD_min : minimale GKD van de optimale uniforme kwantisator
        GKD_min = self.sigma(delta_opt, M, f_u)

        # SQR : SQR van de optimale kwantisator
        mean = integrate.quad(lambda u: u*f_u(u), -np.Inf, np.Inf)[0]
        sigma_U = integrate.quad(lambda u: (u**2) * f_u(u), -np.Inf, np.Inf)[0] - mean**2
        SQR = 10 * np.log10(sigma_U/GKD_min)
        #SQR = (sigma_U/GKD_min)

        # r : kwantisatiedrempels ri = x0 + (2i−M)∆/2
        r_functie = lambda i: (2*i-M)*delta_opt/2
        r_opt = [-np.Inf]
        for i in range(1, M+1):
            r_opt.append(r_functie(i))

        # q : kwantisatieniveaus  qi = x0+(i−(M+1)/2)∆
        q_functie = lambda i: (i - ((M+1)/2))*delta_opt
        q_opt = [q_functie(i) for i in range(1, M+1)]

        # p : relatieve frequentie kwantisatieniveus
        p_functie = lambda i: integrate.quad(lambda u: f_u(u), r_opt[i-1], r_opt[i])[0]
        p_opt = [p_functie(i) for i in range(1, M+1)] 
        p_opt.append(integrate.quad(lambda u: f_u(u), r_opt[len(r_opt)-1], np.Inf)[0])

        # entropie : entropie van het gekwantiseerde signaal
        entropie = 0.0
        for i in range(M):
            entropie += -p_opt[i]*np.log2(p_opt[i])

        return (delta_opt,GKD_min,SQR,entropie,r_opt,q_opt,p_opt)
        
        
    # functie om Lloyd-Max kwantisator te bepalen
    def bepaal_Lloyd_Max_kwantisator(self,M):
        # M : aantal reconstructieniveaus
        
        f_u = self.f_u # w.d.f. - anonieme functie
        
        # Implementeer vanaf hier
        
        #Eerst bepalen we de initiële q_i's
        q = np.array([-1 + 1/M + i*2/M for i in range(0,M)])

        #r waarde initialiseren
        r = np.zeros(M+1)
        r[0] = -1
        r[-1] = 1

        #Initialiseren van gkd's
        sigma_0 = 10000
        sigma_1 = 1

        #Functies
        teller = lambda i : integrate.quad(lambda u: u*f_u(u), r[i-1], r[i])[0]
        noemer = lambda i : integrate.quad(lambda u: f_u(u), r[i-1], r[i])[0]
        update_q_f = lambda i: teller(i)/noemer(i)
        sigma_f = lambda i : integrate.quad(lambda u: pow(q[i-1]-u, 2) * f_u(u), r[i-1], r[i])[0]

        #Iteratieve stappen
        while(True):
            for i in range(1, M):
                r[i] = (q[i-1] + q[i])/2
            q_new = np.array([update_q_f(i) for i in range(1, M+1)])
            q = q_new
            sigma_0 = sigma_1
            sigma_1 = sum(sigma_f(i) for i in range(1, M+1))
            if((sigma_0-sigma_1)/sigma_0 < 0.001): break

        GKD_min = sigma_1
        
        #SQR
        mean = integrate.quad(lambda u: u*f_u(u), -np.Inf, np.Inf)[0]
        SQR = (integrate.quad(lambda u: (u**2) * f_u(u), -np.Inf, np.Inf)[0] - mean**2)/GKD_min
            
        #p
        p_functie = lambda i: integrate.quad(lambda u: f_u(u), r[i-1], r[i])[0]
        p = [p_functie(i) for i in range(1, M+1)] 
        p.append(integrate.quad(lambda u: f_u(u), r[len(r)-1], np.Inf)[0])

        # entropie : entropie van het gekwantiseerde signaal
        entropie = 0.0
        for i in range(M):
            entropie += -p[i]*np.log2(p[i])

        # GKD_min : minimale GKD van de Lloyd-Max kwantisator
        # SQR : SQR van de Lloyd-Max kwantisator
        # entropie : entropie van het gekwantiseerde signaal
        # r : kwantisatiedrempels
        # q : kwantisatieniveaus
        # p : relatieve frequentie kwantisatieniveus
        return (GKD_min,SQR,entropie,r,q,p)
    
    # functie om de compansie kwantisator te bepalen
    def bepaal_compansie_kwantisator(self,M):
        # M : aantal reconstructieniveaus
        
        f_u = self.f_u # w.d.f. - anonieme functie
        
        # Implementeer vanaf hier
        
                
        # GKD : GKD van de compansie kwantisator
        # SQR : SQR van de compansie kwantisator
        # entropie : entropie van het gekwantiseerde signaal
        # r : kwantisatiedrempels
        # q : kwantisatieniveaus
        # p : relatieve frequentie kwantisatieniveus
        return (GKD,SQR,entropie,r,q,p)
                   
    # functie die de kwantisatie uitvoert
    def kwantiseer(self,r,q):
        # r : kwantisatiedremples
        # q : kwantisatieniveaus
                
        data = self.data # originele monsterwaarden
        # Implementeer vanaf hier
        
        # sequentie gekwantiseerd signaal
        samples_kwantiseerd_0 = [0 for _ in range(len(data))]

        for index, data_point in enumerate(data):
            for i in range(len(r) - 1):
                if data_point > r[i] and data_point < r[i+1]:
                    samples_kwantiseerd_0[index] = q[i]
                    break
        samples_kwantiseerd = np.array(samples_kwantiseerd_0)
        return samples_kwantiseerd