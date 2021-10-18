classdef Kwantisatie
    
    properties(Constant)
        % Niet veranderen
        bestandsnaam = 'input.wav'; 
        F1 = 1000;
        F2 = 5000;
        factor_T = 25;
        begin = 1;
        eind = 1104000;
    end
    
    methods(Static=true)
      
        % Functie om de distributie en het genormaliseerd histogram te plotten
        function [data,Fs] = plot_distributie()
            % OUTPUT
             % data : originele monsterwaarden
             % Fs : bemonsteringsfrequentie  audio
            bestandsnaam = Kwantisatie.bestandsnaam;
            F1 = Kwantisatie.F1;
            F2 = Kwantisatie.F2;
            factor_T = Kwantisatie.factor_T;
            begin = Kwantisatie.begin;
            eind = Kwantisatie.eind;
            % genereer monsterwaarde en distributiefunctie
            [fu, Fs, data] = maak_distributie_functie(bestandsnaam,F1,F2,factor_T,begin,eind);   
            
            % Implementeer vanaf hier
                        
        end
        
        % Functie om de optimale uniforme kwantisator te bepalen
        function [Delta_opt,GKD_min,SQR,entropie,r,q,p] = bepaal_optimale_lineaire_kwantisator(M)
             % OUTPUT
              % Delta_opt : optimale stapgrootte
              % GKD_min : minimale GKD van de optimale uniforme kwantisator
              % SQR : SQR van de optimale kwantisator
              % entropie : entropie van het gekwantiseerde signaal
              % r : kwantisatiedrempels
              % q : kwantisatieniveaus
              % p : relatieve frequentie kwantisatieniveaus
              % INPUT
              % M : aantal reconstructieniveaus 
            bestandsnaam = Kwantisatie.bestandsnaam;
            F1 = Kwantisatie.F1;
            F2 = Kwantisatie.F2;
            factor_T = Kwantisatie.factor_T;
            begin = Kwantisatie.begin;
            eind = Kwantisatie.eind;
            
            % genereer monsterwaarde en distributiefunctie
            [fu, ~, ~] = maak_distributie_functie(bestandsnaam,F1,F2,factor_T,begin,eind);
            
            % Implementeer vanaf hier
            
            
        end
        
        % Functie om Lloyd-Max kwantisator te bepalen
        function [GKD,SQR,entropie,r,q,p] = bepaal_Lloyd_Max_kwantisator(M)
            % OUTPUT
             % GKD : minimale GKD van de Lloyd-Max kwantisator
             % SQR : SQR van de Lloyd-Max kwantisator
             % entropie : entropie van het gekwantiseerde signaal
             % r : kwantisatiedrempels
             % q : kwantisatieniveaus
             % p : relatieve frequentie kwantisatieniveaus
            % INPUT
             % M : aantal reconstructieniveaus
            
            bestandsnaam = Kwantisatie.bestandsnaam;
            F1 = Kwantisatie.F1;
            F2 = Kwantisatie.F2;
            factor_T = Kwantisatie.factor_T;
            begin = Kwantisatie.begin;
            eind = Kwantisatie.eind;
            
            % genereer monsterwaarde en distributiefunctie
            [fu, ~, ~] = maak_distributie_functie(bestandsnaam,F1,F2,factor_T,begin,eind);
            
            % Implementeer vanaf hier
           
        end
        
        % Functie om de compansie kwantisator te bepalen
        function [GKD,SQR,entropie,r,q,p] =  bepaal_compansie_kwantisator(M)
            % OUTPUT
             % GKD : GKD Lloyd-Max kwantisator
             % SQR : SQR Lloyd-Max kwantisator
             % entropie : entropie van het gekwantiseerde signaal
             % r : kwantisatiedrempels
             % q : kwantisatieniveaus
             % p : relatieve frequentie kwantisatieniveaus
            % INPUT
             % M : aantal reconstructieniveaus
            
            bestandsnaam = Kwantisatie.bestandsnaam;
            F1 = Kwantisatie.F1;
            F2 = Kwantisatie.F2;
            factor_T = Kwantisatie.factor_T;
            begin = Kwantisatie.begin;
            eind = Kwantisatie.eind;
            
            % genereer monsterwaarde en distributiefunctie
            [fu, ~, ~] = maak_distributie_functie(bestandsnaam,F1,F2,factor_T,begin,eind);
            
            % Implementeer vanaf hier
                        
        end
            
        % Functie die de kwantisatie uitvoert
        function [samples_kwantiseerd,Fs] = kwantiseer(r,q)
            % OUTPUT
             % samples_kwantiseerd : sequentie gekwantiseerd signaal
             % Fs : bemonsteringsfrequentie audio
            % INPUT
             % r : kwantisatiedremples
             % q : kwantisatieniveaus

            bestandsnaam = Kwantisatie.bestandsnaam;
            F1 = Kwantisatie.F1;
            F2 = Kwantisatie.F2;
            factor_T = Kwantisatie.factor_T;
            begin = Kwantisatie.begin;
            eind = Kwantisatie.eind;
            
            % genereer monsterwaarde en distributiefunctie
            [~,Fs, data] = maak_distributie_functie(bestandsnaam,F1,F2,factor_T,begin,eind);
            
            % Implementeer vanaf hier
            
        end
        
        %--------- Hierna niets veranderen -------------
         
        % Functie om numerieke inverse te bepalen van stijgende anonieme functie g voor de y-waarden in Yvec
        function Uvec = inverse(g,Yvec)
            Uvec = zeros(size(Yvec));
            eps = 1e-6;
            for i = 1 : length(Yvec)
                y = Yvec(i);
                a = -1; g_a = g(a);
                b = 1; g_b = g(b);
                nit = 1;
                while(abs(g_a-y)>eps && abs(g_b-y)>eps && nit<1000)
                    u_test = (a+b)/2;
                    g_test = g(u_test);
                    if(g_test<=y)
                        a = u_test;
                        g_a = g_test;
                    else
                        b = u_test;
                        g_b = g_test;
                    end
                    nit = nit+1;
                end
                if(abs(g_a-y)<=eps)
                    Uvec(i) = a;
                else
                    Uvec(i) = b;
                end
            end
        end
         
    end
end
