function y = pulse(t,T,alpha)
% Functie: De square root raised cosine pulse
% input:
% t sample waarden
% T tijdsinterval van 1 symbool: standaard gelijk aan 1
% alpha: rolloff factor
% vb van gebruik
%
% alpha = 0.5;
% t = [-5:0.1:5];
% s = PHY_Jelle.pulse(t, 1, alpha);
% plot(t, s)
% xlabel('tijd t/T');
% ylabel('y(t)');
% title(['Square root raised cosine pulse met rollofffactor ' num2str(alpha)]);

een=(1-alpha)*sinc(t*(1-alpha)/T);
twee=(alpha)*cos(pi*(t/T-0.25)).*sinc(alpha*t/T-0.25);
drie=(alpha)*cos(pi*(t/T+0.25)).*sinc(alpha*t/T+0.25);
y=1/(sqrt(T))*(een+twee+drie);

end