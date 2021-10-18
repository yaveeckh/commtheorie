import wave
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

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
        plt.hist(data, normed=True)
        plt.savefig('fu_hist.png')
        plt.show()
        
    # functie om de optimale uniforme kwantisator te bepalen
    def bepaal_optimale_lineaire_kwantisator(self,M):
        # M : aantal reconstructieniveaus
        
        f_u = self.f_u # w.d.f. - anonieme functie
        
        # Implementeer vanaf hier
                
        
        # delta_opt : optimale stapgrootte
        # GKD_min : minimale GKD van de optimale uniforme kwantisator
        # SQR : SQR van de optimale kwantisator
        # entropie : entropie van het gekwantiseerde signaal
        # r : kwantisatiedrempels ri = x0 + (2i−M)∆/2
        # q : kwantisatieniveaus  qi = x0+(i−(M+1)/2)∆ 
        # p : relatieve frequentie kwantisatieniveus
        return (delta_opt,GKD_min,SQR,entropie,r_opt,q_opt,p_opt)
        
    
    # functie om Lloyd-Max kwantisator te bepalen
    def bepaal_Lloyd_Max_kwantisator(self,M):
        # M : aantal reconstructieniveaus
        
        f_u = self.f_u # w.d.f. - anonieme functie
        
        # Implementeer vanaf hier
        
               
        # GKD_min : minimale GKD van de Lloyd-Max kwantisator
        # SQR : SQR van de Lloyd-Max kwantisator
        # entropie : entropie van het gekwantiseerde signaal
        # r : kwantisatiedrempels
        # q : kwantisatieniveaus
        # p : relatieve frequentie kwantisatieniveus
        return (GKD_min,SQR,entropie,ri,qi,p)
    
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
        return samples_kwantiseerd 
