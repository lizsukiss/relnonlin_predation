function dxdt = RC2PEquations(t,x,params)

dxdt = zeros(size(x));
% params = {a2, aP, h2, hP, d2, dP};
a2 = params.a2;
h2 = params.h2;
d2 = params.d2;
aP = params.aP;
hP = params.hP;
dP = params.dP;

%------- State variables ------------------------------------------------;

R3  = x(1);
C4  = x(2);
P3  = x(3);

% ------ Rates of change -----------------------------------------------;

N3 = 1-R3;
dxdt(1) = (N3-(a2*C4/(1+h2*a2*R3)))*R3; 
dxdt(2) = (a2*R3/(1+h2*a2*R3)-d2-aP*P3/(1+hP*aP*C4))*C4;
dxdt(3) = ((aP*C4/(1+hP*aP*C4))-dP)*P3;  

end
