classdef ModDet
          
   methods(Static=true)
        
        %% Functies voor Mapping/Demapping
        
        % Functie die bitstring omzet naar complexe symbolen
        function a = mapper(bitstring, constellatie)
            % OUTPUT
             % a : sequentie van data symbolen.
            % INPUT
             % bitstring : sequentie van bits
             % constellatie : ofwel 'BPSK', '4QAM', '4PSK', '4PAM', etc.
            
            switch(constellatie)
                case 'BPSK'                    
                                
                otherwise
                    error('Constellatie niet herkend');
            end
                        
        end
        
        % Functie die complex symbolen omzet naar bits
        function bitstring = demapper(a, constellatie)
            % OUTPUT
             % bitstring : sequentie van bits.
            % INPUT
             % a : sequentie van data symbolen
             % constellatie : ofwel 'BPSK', '4QAM', '4PSK', '4PAM', etc.    
            
            switch(constellatie)
                case 'BPSK'    
                    
                otherwise
                    error('Constellatie niet herkend');
                    
            end
        end        
         
        % Functie die desicie toepast op u
        function a_estim = decisie(u, constellatie)
            % OUTPUT
             % a_estim : vector met geschatte (complexe) symbolen.
            % INPUT
             % x : vector met ruizige (complexe) symbolen.
             % constellatie : ofwel 'BPSK', '4QAM', '4PSK', '4PAM', etc.   
             
             switch(constellatie)
                case 'BPSK'                   
                                    
                otherwise
                    error('Constellation not recognized');
             end
        end
        
        % Functie die decisie variabele aanmaakt
        function u = maak_decisie_variabele(rdec,hch_hat,theta_hat)
            % INPUT
             % rdec : vector met het gedecimeerde ontvangen signaal.
             % hch_hat : schatting van amplitude van het kanaal.
             % theta_hat : schatting van fase van de demodulator.
            % OUTPUT
             % u : vector met decisie-variabele.
        end
        
        %% moduleren/demoduleren
        % Functie die de modulatie implementeert
        function s = moduleer(a,T,Ns,frequency,alpha,Lf)
            % OUTPUT
             % s : vector met gemoduleerde samples.
            % INPUT
             % a: sequentie van data symbolen
             % T : symboolperiode in seconden
             % Ns : aantal samples per symbool. 
             % frequentie : carrier frequentie in Hz.
             % alpha : roll-off factor.
             % Lf : pulse duur uitgedrukt in aantal symboolintervallen
           
        end
        
        % Functie die de demodulatie implementeert
        function [rdemod] = demoduleer(r,T,Ns,frequentie,alpha,Lf,theta)
            % OUTPUT
             % rdemod: vector met gedemoduleerde samples.
            % INPUT
             % r: vector met ontvangen samples.
             % T : symboolperiode in seconden
             % Ns : aantal samples per symbool. 
             % frequentie : carrier frequentie in Hz.
             % alpha : roll-off factor.
             % Lf : pulse duur uitgedrukt in aantal symboolintervallen
             % theta : fase van de demodulator
            
        end
        
        % Functie die de pulse aanmaakt - niet veranderen
        function y = pulse(t,T,alpha)
            % OUTPUT
             % y : samples van de pulse
            % INPUT
             % t : samples van tijdstippen waarop je de pulse-waarde wilt kennen.
             % T : tijdsinterval van 1 symbool in seconden. 
             % alpha : rolloff factor.
                        
            % vb van gebruik
            % alpha = 0.5;
            % t = [-5:0.1:5];
            % s = ModDet.pulse(t, 1, alpha);
            % plot(t, s)
            % xlabel('tijd t/T');
            % ylabel('y(t)');
            % title(['Square root raised cosine pulse met rollofffactor ' num2str(alpha)]);
                        
            een=(1-alpha)*sinc(t*(1-alpha)/T);
            twee=(alpha)*cos(pi*(t/T-0.25)).*sinc(alpha*t/T-0.25);
            drie=(alpha)*cos(pi*(t/T+0.25)).*sinc(alpha*t/T+0.25);
            y=1/(sqrt(T))*(een+twee+drie);

        end
        
        % Functie die het decimeren implementeert
        function rdec = decimatie(rdemod,Ns,Lf)
            % OUTPUT
             % rdown: vector met 1 sample per symbool (gedecimeerd)
            % INPUT
             % rdec : vector met Ns samples per symbool.
             % Ns : aantal samples per symbool.
             % Lf : pulse duur uitgedrukt in aantal symboolintervallen
             
        end   
        
        %% het kanaal
        % Functie die het AWGN kanaal simuleert
        function r = kanaal(s,sigma,hch)
            % OUTPUT
             % r : uitgang van het kanaal
            % INPUT:
             % s : input van het kanaal
             % Ns: aantal samples per symbool
             % sigma: standaard deviatie van de ruis.
            
        end  
        
    end
    
end