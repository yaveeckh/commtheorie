from numpy.lib.index_tricks import IndexExpression, MGridClass
from moddet import ModDet
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy
import time
import random
import math
from scipy.fft import fft, fftfreq
from scipy import special

from playsound import playsound

#variables:
T = 10**(-6)
Ns = 6
f0 = 2*10**6
alpha = 0.5
Lf = 10
N0 = 0.0005
sigma = math.sqrt(N0*Ns/2/T)
hch = 1
theta = math.pi / 16

obj = ModDet()
    
######### PLOT |P(f)| #########
def plot_pulsen():
    def plot_puls(alhpa, filename):
        t_theorie = np.linspace(-1000*Lf*T, 1000*Lf*T, 2*Lf*Ns*1000 + 1)
        t_vector = np.linspace(-Lf*T, Lf*T, 2*Lf*Ns + 1)
        afgeknot = obj.pulse(t_vector, T, alhpa)
        afgeknot_fourier = fft(afgeknot)*T/Ns
        afgeknot_x = fftfreq(2*Lf*Ns + 1, d=T/Ns)

        theorie = obj.pulse(t_theorie, T, alpha)
        theorie_fourier = fft(theorie)*T/Ns
        theorie_x = fftfreq(2*1000*Lf*Ns +1, d=T/Ns)

        #plt.plot(afgeknot_x, np.abs(afgeknot_fourier), label = "afgeknot")
        plt.semilogy(afgeknot_x, np.abs(afgeknot_fourier), label = "afgeknot")
        #plt.plot(theorie_x, np.abs(theorie_fourier), label = "theoretisch")
        plt.semilogy(theorie_x, np.abs(theorie_fourier), label = "theoretisch")
        plt.legend()
        plt.xlabel('frequentie [Hz]')
        plt.ylabel('|P(f)|')
        plt.ylim((10**(-7), 5*10**(-3)))
        plt.savefig(filename)
        plt.close()
        return

    # plot_puls(0.05, "moddet/puls/alpha_05.png")
    # plot_puls(0.5, "moddet/puls/alpha_50.png")
    # plot_puls(0.95, "moddet/puls/alpha_95.png")
    return

######### PLOT BER #########
def plot_BER_alle3():
    def plot_ber(constellatie, L):
        tabel = obj.mappingstabel(constellatie)
        mc = math.log2(len(tabel))
        Eb = 1/mc
        N0_list = [i*0.005 + 0.0001 for i in range(200)]
        N = 100 #aantal steekproeven

        #bitvector met lengte L
        bitstring = bin(random.randint(0,2**L))[2:].zfill(L)
        bitvector_in = []
        for bit in bitstring:
            bitvector_in.append(int(bit))

        BER_mx = np.reshape(np.zeros(len(N0_list)*N), (len(N0_list), N))
        for i in range(0,len(N0_list)):
            for j in range(0,100):
                N0 = N0_list[i]
                bitvector_out = obj.modulation_detection(bitvector_in, constellatie, T, Ns, f0, alpha, Lf, N0, hch, theta)
                aantal_fout = sum(abs(np.array(bitvector_out)-np.array(bitvector_in)))
                BER_mx[i,j] = aantal_fout/L
        BER = np.mean(BER_mx, 1)
        x = 10*np.log10(np.ones(len(N0_list))*Eb/N0_list)
        plt.semilogy(x, BER, label = constellatie)
        return

    plot_ber('BPSK', 1000)
    plot_ber('4QAM', 1000)
    plot_ber('4PAM', 1000)
    x = np.arange(0, 14, 0.01)
    # Q(x) = 0.5 - 0.5*erf(x/sqrt(2))
    Q = 0.5 - 0.5*special.erf(np.sqrt(2*10**(x/10)) / math.sqrt(2))
    plt.semilogy(x, Q, label = 'Q(sqrt(2Eb/N0)')

    plt.ylim((10**(-4), 10**(-1)))
    plt.xlim((0,14))
    plt.xlabel('Eb/N0 [dB]')
    plt.ylabel('BER')
    plt.legend()
    plt.savefig('moddet/BER/alle3_v2.png')
    plt.close()

    return

def plot_BER_037():
    def plot_ber(constellatie, L):
        tabel = obj.mappingstabel(constellatie)
        mc = math.log2(len(tabel))
        Eb = 1/mc
        N0_list = [i*0.005 + 0.0001 for i in range(200)]
        N = 100 #aantal steekproeven

        #bitvector met lengte L
        bitstring = bin(random.randint(0,2**L))[2:].zfill(L)
        bitvector_in = []
        for bit in bitstring:
            bitvector_in.append(int(bit))

        BER_mx = np.reshape(np.zeros(len(N0_list)*N), (len(N0_list), N))
        for i in range(0,len(N0_list)):
            for j in range(0,100):
                N0 = N0_list[i]
                bitvector_out = obj.modulation_detection(bitvector_in, constellatie, T, Ns, f0, alpha, Lf, N0, hch, theta)
                aantal_fout = sum(abs(np.array(bitvector_out)-np.array(bitvector_in)))
                BER_mx[i,j] = aantal_fout/L
        BER = np.mean(BER_mx, 1)
        x = 10*np.log10(np.ones(len(N0_list))*Eb/N0_list)
        plt.semilogy(x, BER)
        plt.vlines(1.33, 0, 0.05, color = 'k', lw = 0.5)
        plt.hlines(0.05, 0, 1.33, color = 'k', lw =0.5)
        plt.plot([1.33], [0.05], marker='o', markersize=3, color="red")
        plt.ylim((10**(-4), 10**(-1)))
        plt.xlim((0,10))
        plt.xlabel('Eb/N0 [dB]')
        plt.ylabel('BER')
        plt.savefig('moddet/BER/BPSK.png')
        plt.close()
        return

    plot_ber('BPSK', 1000)
    return

######### PLOT SCATTER #########

def plot_scatter_4QAM(N0, filename):
    sigma = math.sqrt(N0*Ns/2/T)

    bitstring = bin(random.randint(0,2**500))[2:].zfill(500)
    bitvector_in = []
    for bit in bitstring:
        bitvector_in.append(int(bit))

    a = obj.mapper(bitvector_in, '4QAM')
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, sigma, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
    rdown = obj.decimatie(rdemod, Ns, Lf)
    u = obj.maak_decisie_variabele(rdown, hch, theta)

    u_real =  [e.real for e in u]
    u_complex = [e.imag for e in u]
    plt.scatter(u_real, u_complex, marker=".")
    plt.xlabel('real(u)')
    plt.ylabel('imag(u)')
    plt.savefig(filename)
    plt.close()
                

    return

######### PLOT AMPLITUDESCHATTINGSFOUT BPSK #########
def plot_epsilon_scatter(constellatie, epsilon, filename):
    hch_hat = hch * (1+epsilon)
    bitstring = bin(random.randint(0,2**500))[2:].zfill(500)
    bitvector_in = []
    for bit in bitstring:
        bitvector_in.append(int(bit))

    a = obj.mapper(bitvector_in, constellatie)
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, 0, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
    rdown = obj.decimatie(rdemod, Ns, Lf)
    u = obj.maak_decisie_variabele(rdown, hch_hat, theta)

    u_real =  [e.real for e in u]
    u_complex = [e.imag for e in u]
    plt.scatter(u_real, u_complex, marker=".")
    plt.xlabel('real(u)')
    plt.ylabel('imag(u)')
    plt.savefig(filename)
    plt.close()
    return

def plot_epsilon_BER(constellatie):
    def plot_ber(constellatie, L, epsilon):

        tabel = obj.mappingstabel(constellatie)
        mc = math.log2(len(tabel))
        Eb = 1/mc
        N0_list = [i*0.005 + 0.0001 for i in range(200)]
        N = 100 #aantal steekproeven

        #bitvector met lengte L
        bitstring = bin(random.randint(0,2**L))[2:].zfill(L)
        bitvector_in = []
        for bit in bitstring:
            bitvector_in.append(int(bit))

        BER_mx = np.reshape(np.zeros(len(N0_list)*N), (len(N0_list), N))
        for i in range(0,len(N0_list)):
            for j in range(0,100):
                N0 = N0_list[i]
                bitvector_out = obj.modulation_detection(bitvector_in, constellatie, T, Ns, f0, alpha, Lf, N0, hch, theta, epsilon, 0)
                aantal_fout = sum(abs(np.array(bitvector_out)-np.array(bitvector_in)))
                BER_mx[i,j] = aantal_fout/L
        BER = np.mean(BER_mx, 1)
        x = 10*np.log10(np.ones(len(N0_list))*Eb/N0_list)
        lbl = 'epsilon = ' + str(epsilon)
        plt.semilogy(x, BER, label = lbl)
        return

    plot_ber(constellatie, 1000, 0.0)
    plot_ber(constellatie, 1000, 0.1)
    plot_ber(constellatie, 1000, 0.2)

    plt.ylim((10**(-4), 10**(-1)))
    plt.xlim((0,14))
    plt.xlabel('Eb/N0 [dB]')
    plt.ylabel('BER')
    plt.legend()
    flnm = 'moddet/BER/BER_epsilon_' + constellatie
    plt.savefig(flnm)
    plt.close()

    return

######### PLOT FASESCHATTINGSFOUT BPSK #########

def plot_phi_scatter(constellatie, phi, filename):

    theta_hat = theta + phi
    bitstring = bin(random.randint(0,2**500))[2:].zfill(500)
    bitvector_in = []
    for bit in bitstring:
        bitvector_in.append(int(bit))

    a = obj.mapper(bitvector_in, constellatie)
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, 0, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
    rdown = obj.decimatie(rdemod, Ns, Lf)
    u = obj.maak_decisie_variabele(rdown, hch, theta_hat)

    u_real =  [e.real for e in u]
    u_complex = [e.imag for e in u]
    plt.scatter(u_real, u_complex, marker=".")
    plt.xlabel('real(u)')
    plt.ylabel('imag(u)')
    plt.savefig(filename)
    plt.close()
    
    return

def plot_phi_BER(constellatie):
    def plot_ber(constellatie, L, phi):
        tabel = obj.mappingstabel(constellatie)
        mc = math.log2(len(tabel))
        Eb = 1/mc
        N0_list = [i*0.005 + 0.0001 for i in range(200)]
        N = 100 #aantal steekproeven

        #bitvector met lengte L
        bitstring = bin(random.randint(0,2**L))[2:].zfill(L)
        bitvector_in = []
        for bit in bitstring:
            bitvector_in.append(int(bit))

        BER_mx = np.reshape(np.zeros(len(N0_list)*N), (len(N0_list), N))
        for i in range(0,len(N0_list)):
            for j in range(0,100):
                N0 = N0_list[i]
                bitvector_out = obj.modulation_detection(bitvector_in, constellatie, T, Ns, f0, alpha, Lf, N0, hch, theta, 0, phi)
                aantal_fout = sum(abs(np.array(bitvector_out)-np.array(bitvector_in)))
                BER_mx[i,j] = aantal_fout/L
        BER = np.mean(BER_mx, 1)
        x = 10*np.log10(np.ones(len(N0_list))*Eb/N0_list)
        if phi == 0: phi_txt = '0'
        elif abs(phi - math.pi/16) < 0.0001:
            phi_txt = 'pi/16'
        elif abs(phi - math.pi/8) <0.0001:
            phi_txt = 'pi/8'
        elif abs(phi - math.pi/4) <0.0001:
            phi_txt = 'pi/4'
        lbl = 'phi = ' + phi_txt
        plt.semilogy(x, BER, label = lbl)
        return

    plot_ber(constellatie, 1000, 0)
    plot_ber(constellatie, 1000, math.pi/16)
    plot_ber(constellatie, 1000, math.pi/8)
    plot_ber(constellatie, 1000, math.pi/4)

    plt.ylim((10**(-4), 5*10**(-1)))
    plt.xlim((0,14))
    plt.xlabel('Eb/N0 [dB]')
    plt.ylabel('BER')
    plt.legend()
    flnm = 'moddet/BER/BER_phi_' + constellatie
    plt.savefig(flnm)
    plt.close()
    return

######### DEMOD  #########
def plot_demod():
    alpha = 0.5
    constellatie = 'BPSK'
    bitstring = bin(random.randint(0,2**100))[2:].zfill(100)
    bitvector_in = []
    for bit in bitstring:
        bitvector_in.append(int(bit))
    
    a = obj.mapper(bitvector_in, constellatie)
    s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
    r = obj.kanaal(s, 0, hch)
    rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)

    rdemod_overgang1 = rdemod[:2*Lf*Ns]
    rdemod_overgang2 = rdemod[-2*Lf*Ns:]
    time1 = np.arange(0, len(rdemod[:2*Lf*Ns]), 1)*T/Ns
    time2 = np.arange(0, len(rdemod[-2*Lf*Ns:]), 1)*T/Ns + 12*Lf*T
    plt.plot(time1, abs(rdemod_overgang1), color = 'r')
    plt.plot(time2, abs(rdemod_overgang2), color = 'r', label = 'overgangsverschijnsel')
    rdemod_af = rdemod[2*Lf*Ns:-2*Lf*Ns + 1] 
    time_af = np.arange(0,len(rdemod_af),1)*T/Ns + 2*Lf*T
    plt.plot(time_af,abs(rdemod_af), color = 'b')
    plt.xlabel('tijd [s]')
    plt.ylabel('|rdemod(t)|')
    plt.legend()
    plt.savefig('moddet/decimatie/demod.png')
    plt.close()


    return


######### OOGDIAGRAM  #########
def plot_all_eyediagrams():
    def plot_eye(constellatie, alpha):
        bitstring = bin(random.randint(0,2**100))[2:].zfill(100)
        bitvector_in = []
        for bit in bitstring:
            bitvector_in.append(int(bit))
        
        a = obj.mapper(bitvector_in, constellatie)
        s = obj.moduleer(a, T, Ns, f0, alpha, Lf)
        r = obj.kanaal(s, 0, hch)
        rdemod = obj.demoduleer(r, T, Ns, f0, alpha, Lf, theta)
        
        r_oog = rdemod[2*Lf*Ns:-2*Lf*Ns + 1].real
        t = np.arange(0,len(r_oog),1)*T/Ns + 2*Lf*T
        plt.plot(t, r_oog)
        plt.show()
        plt.close()
    
        return

    plot_eye('BPSK', 0.5)
    return



# plot_BER_037()
# plot_BER_alle3()
# plot_scatter_4QAM(0.000, 'moddet/scatter/4QAM_000.png')
# plot_scatter_4QAM(0.001, 'moddet/scatter/4QAM_001.png')
# plot_scatter_4QAM(0.01, 'moddet/scatter/4QAM_010.png')
# plot_scatter_4QAM(0.1, 'moddet/scatter/4QAM_100.png')
# plot_epsilon_scatter('BPSK', 0.1, 'moddet/scatter/BPSK_epsilon.png')
# plot_epsilon_scatter('4PAM', 0.1, 'moddet/scatter/4PAM_epsilon.png')
# plot_epsilon_scatter('4QAM', 0.1, 'moddet/scatter/4QAM_epsilon.png')
# plot_phi_scatter('4QAM', math.pi/16, 'moddet/scatter/4QAM_phi.png')
# plot_phi_BER('4QAM')
# plot_epsilon_BER('BPSK')
# plot_epsilon_BER('4PAM')

# plot_demod()
plot_all_eyediagrams()

