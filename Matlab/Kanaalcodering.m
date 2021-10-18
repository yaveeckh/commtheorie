classdef Kanaalcodering
    
    methods(Static=true)
        
        % Functie die de encoder van de uitwendige code implementeert
        function [bitenc] = encodeer_uitwendig(bitstring)
            % OUTPUT
             % bitenc : vector met gecodeerde bits
            % INPUT
             % bitstring : vector met ongecodereerde bits
             
            G = []; % vul hier de generatormatrix in

            bitstring = bitstring(:)'; % bitstring zeker een rij vector 
            N = length(bitstring);
            N_codewoorden = ceil(N/10);
            bitstring = [bitstring zeros(1, N_codewoorden*10-N)]; % vul aan met nullen als bitstring geen geheel aantal informatiewoorden is. 
           
        end
        
        % Functie die de decoder van de uitwendige code implementeert 
        function [bitdec,bool_fout] = decodeer_uitwendig(bitstring)
            % OUTPUT
             % bitdec : vector met gedecodeerde bits bij volledige foutcorrectie
             % bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders
            % INPUT
             % bitstring : vector met gecodeerde bits
           
            H = [[1 1 0 1 1 0 0 1 0 0 1 0 0 1];[0 0 1 1 1 1 0 1 0 0 1 1 1 0 ];[0 0 1 1 0 0 1 1 0 1 0 1 0 1];[1 0 0 0 0 0 0 1 1 1 1 1 1 0]];

            bitstring = bitstring(:)';
            N = length(bitstring);
            N_codewords = ceil(N/14);
        end
        
        % Functie die de encoder van de inwendige code implementeert
        function bitenc = encodeer_inwendig(bitstring,g_x)
            % OUTPUT
             % bitenc : vector met gecodeerde bits
            % INPUT
             % bitstring : vector met ongecodereerde bits
             % g_x : CRC-veelterm
            
        end
        
        % Functie die de decoder van de inwendige code implementeert
        function [bitdec,bool_fout] = decodeer_inwendig(bitstring,g_x)
            % INPUT
             % bitstring : vector met gecodeerde bits
             % g_x : CRC-veelterm
            % OUTPUT
             % bitdec : vector met gedecodeerde bits
             % bool_fout : 1 als een fout gedetecteerd is bij zuivere foutdetectie, 0 anders
           
           
            
        end
        

    end
end
