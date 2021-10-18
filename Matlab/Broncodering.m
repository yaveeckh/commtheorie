classdef Broncodering
              
    methods(Static=true)
        
        % Functie die de codetabel opstelt voor de Huffmancode
        function [codewoorden,gem_len,boom_info] = maak_codetabel_Huffman(rel_freq)
            % OUTPUT
             % codewoorden : 1xM cell array met codewoorden
             % gem_len : gemiddelde codewoordlengte
             % boom_info : (M+M-1)x2 matrix met informatie over boomstructuur
            % INPUT
             % rel_freq : 1xM vector met relatieve frequenties
            
            M = numel(rel_freq);
                      
            codewoorden = cell(1,M); % hierin komt output
                        
            
        end
      
        % Functie voor het encoderen met vaste-lengte code
        function data_geencodeerd = vaste_lengte_encodeer(data,alfabet)
            % OUTPUT
             % data_geencodeerd : geëncodeerde data 
            % INPUT
             % data : de data die geëncodeerd moet worden
             % alfabet : rijvector met alle mogelijke symbolen
        
        end
                
        % Functie voor het decoderen met vaste-lengte code
        function data_gedecodeerd = vaste_lengte_decodeer(data,alfabet)
            % OUTPUT
             % data_gedecodeerd : gedecodeerde data
            % INPUT
             % data : data die gedecodeerd moet worden
             % alfabet : rijvector met alle mogelijke symbolen
           
        end
        
        % Functie die sequentie van bronsymbolen omzet naar macrosymbolen en de relative frequenties ervan berekent
        function [macrosymbolen,alfabet_vector,rel_freq] = scalair_naar_vector(bronsymbolen,alfabet_scalair)
            % OUTPUT
             % macrosymbolen : rij met bronsymbolen omgezet in macrosymbolen
             % alfabet_vector : alfabet van de vectorcodering
             % rel_freq : de relative frequentie van elk macrosymbool
            % INPUT
             % bronsymbolen : rij met bronsymbolen die omgezet moet worden
             % alfabet_scalair : rij met alfabet van scalaire codering
            
        end
        
        % Functie die sequentie van macrosymbolen omzet naar sequentie van bronsymbolen
        function [bronsymbolen] = vector_naar_scalair(macrosymbolen,alfabet_scalair)
            % OUTPUT
             % bronsymbolen : rij met macrosymbolen omgezet in bronsymbolen 
            % INPUT
             % macrosymbolen : rij met macrosymbolen  die omgezet moet worden
             % alfabet_scalair : alfabet van scalaire codering
           
        end
        
        %--------- Hierna niets veranderen -------------
       
        % Functie die de data sequentie encodeert met Huffman code
        function output = Huffman_encodeer(data, alfabet, codewoorden)
            % OUTPUT
             % output : geëncodeerde data
            % INPUT
             % data : de data die geëncodeerd moet worden
             % alfabet : cell array of rijvector met alle mogelijke symbolen
             % codewoorden : cell array met de codewoorden 
                        
            N = length(data);
            N_symbols = length(alfabet);            
            
            if iscell(alfabet) && max(cellfun(@length, alfabet))>1
                error('ERROR: een symbool uit het alfabet mag maar 1 cijfer of alfanumerieke waarde bevatten');
            end
            
            if ischar(data)                
                % converteer de alfanumerieke waardes naar numeriek waardes
                numeric_data = double(data);
                numeric_symbols =  cellfun(@double, alfabet);                             
            elseif isrow(data)||iscolumn(data)
                numeric_data = data;
                if iscell(alfabet),    numeric_symbols = cell2mat(alfabet);     
                else numeric_symbols = alfabet;     
                end
            else
                error('ERROR: input moet ofwel een rijvector of een karakater-string zijn');
            end    
             
            % De kern van deze functie: vervang elke letter uit het alfabet 
            % met het corresponderend codewoord            
            output = cell(1,N);
            for n=1:N_symbols
                idx = numeric_data==numeric_symbols(n);
                output(idx) = {codewoorden{n}};
            end
            
            % Als er nog lege cellen overblijven wil dit zeggen dat het
            % woordenboek niet alle mogelijke symbolen uit de data heeft.
            if nnz(cellfun(@isempty,output))
                error('ERROR: woordenboek incompleet');
            end
                   
            % zet om naar een rijvector
            output = cell2mat(output);            
            
        end
        
        % Functie die de data sequentie decodeert met Huffman code
        function data_gedecodeerd = Huffman_decodeer(data, alfabet, codewoorden, boom_info)
            % OUTPUT
             % data_gedecodeerd : gedecodeerde data
            % INPUT
             % data : data die gedecodeerd moet worden
             % alfabet : cell array of rijvector met alle mogelijke symbolen
             % codewoorden : cell array met de codewoorden
             % boom_info : (M+M-1)x2 matrix met informatie over boomstructuur
            
            N = length(data);
            M = length(codewoorden);
                
            output = [];
            idx = 1;
            
            indx_tree = size(boom_info,1);
            
            while idx <= N
                next = data(idx)+1;
                indx_tree = boom_info(indx_tree,next);
                if(indx_tree <= M)
                    output = [output indx_tree];
                    indx_tree = size(boom_info,1);
                end
                idx = idx +1;
            end
                         
            % zet de output om naar letters uit het alfabet
            data_gedecodeerd = alfabet(output);
            
        end
        
    end
    
end