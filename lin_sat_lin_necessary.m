function lin_sat_lin_necessary(a1,a2,h2,dP)
    hold on

    % PARAMS without d1 and d2!
    PARAMS = [a1, a2, 1, 0, 0, h2, 0, dP, 0, 0, 0, 0];
    %%%%%%%%%%%%%%%%%%
    aP = 1;
    % Boundary of plot determined by possible invasion as first consumer
    Rs=1;
    d1max=(a1*Rs)    % C1 can invade R 
    d2max=a2*Rs/(1+h2*a2*Rs) % C2 can invade R
    d1=0:0.001:d1max;
    
    % Max for d2 in terms of d1 determined by the
    % possibility of invasion in the R-C1-P system by C2
    Rstar = RC1P_Rstar(PARAMS, d1); % in R-C1-P
    Pstar = RC1P_Pstar(PARAMS, d1, Rstar); % 
    
    d2Boundary1=a2*Rstar./(1+h2*a2*Rstar)-aP*Pstar; 
    plot(d1,d2Boundary1,"--", "LineWidth",5)
    
    % Min for d2 in terms of d1 determined by the competitive exclusion 
    % of C1 in the R-C2-P system by C2/impossible invasion by C1
    if a2/(1+h2*a2*Rs)-a1 > 0
        d2Boundary2 = d1; % 
        "positive, taking lower limit for R average (with C2), i.e. 0"
    else % NOT YET AUTOMATIC
        RstarC2 = 0; % in this case there is no coexistence
        d2Boundary2 = (a2/(1+h2*a2*Rs)-a1)*RstarC2 + d1;
        "negative, taking upper limit for R average (with C2), i.e. Rstar in C2"
    end
    hold on
    plot(d1,d2Boundary2,"-.","LineWidth",2)
end

function Pstar = RC1P_Pstar(params,mortRateC1, Rstar)

    params=num2cell(params);
    [a1, a2, aP, hR, h1, h2, hP, dP, eta1, eta2, etaP, delta] = params{:};
    
    i = 1; %index for Pstar elements
    Pstar = (a1.*Rstar - mortRateC1 - delta)/aP;
        
    for i=1:length(Pstar)
        if isnan(Pstar(i))
            Pstar(i) = 0;
        end
    end
    
end


function Rstar = RC1P_Rstar(params, mortRateC1)

    params=num2cell(params);
    [a1, a2, aP, hR, h1, h2, hP, dP, eta1, eta2, etaP, delta] = params{:};

    C = (dP+delta)/aP;

    Rstar = zeros(1,length(mortRateC1));

    i = 1; %index for RstarToni elements
    for d1=mortRateC1
        Rstar(i) = ( (a1*C+delta) + ((a1*C+delta)*hR - 1) * ( 1-eta1*C+etaP*(d1+delta)/aP)) / ( ((a1*C+delta)*hR-1) * (1+etaP*a1/aP));
        if isnan(Rstar(i)) || Rstar(i)<0
           Rstar(i) = 0;
        end
        i = i + 1;
    end
end



















































