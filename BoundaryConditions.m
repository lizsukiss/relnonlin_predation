function BoundaryConditions(fighandle, params, model)

    % aC1 = 1; % aC2 = 4; % hC2 = 1; % aP  =  1.5; %1 %0.25; % dP  = 0.4; %0.1;
    % species: all or C2 (for R-C2-P)
    
    % Parameters
    params=num2cell(params); 
    [aC1, aC2, hC2, aP, hP, dP] = params{:};

    % Axis limit, colours, fig
    dC1max = aC1;
    dC2max = aC2/(1+hC2*aC2);
    colorC1 = [0.8500, 0.3250, 0.0980];
    colorC2 = [0.9290, 0.6940, 0.1250];
    colorP = 'r';% [0.5 0.5 0.5];
    figure(fighandle); %clf
    hold on;

    % % Persistence/invasion boundary of P in R-C1(-P)  ---> will be done
    % numerically
    dC1per = (1-(dP/aP)*aC1/(1-dP*hP))*aC1;
    d2 = 0:0.001:dC2max;
    d1 = d2*0 + dC1per;
    plot(d1,d2,'-','Linewidth',3,"Color", colorP);
    
    % Persistence boundary of P in R-C2-P
    p = (1-hC2*aC2)/(hC2*aC2);
    C2star = dP/(aP*(1-dP*hP));
    q = (aC2*C2star-1)/(hC2*aC2);
    RS = -p/2 + ((p/2)^2 - q)^(1/2);

    P = ( aC2*RS/(1+hC2*aC2*RS)-d2 )*(1+aP*hP*C2star)/aP;

    if imag(RS) == 0
        
        dC2per = aC2*RS/(1+hC2*aC2*RS)
        d1 = 0:0.001:dC1max;
        d2 = d1*0 + dC2per;
        plot(d1,d2,'--','Linewidth',3,"Color",colorP);
        
        % Invasion boundary of P in R-C2 (numerically calculated)
        %d2Interval = [0.001:0.01:dC2max];
        %[Rmean Rstd C2mean C2std]= getmeans_RC2(aC2,hC2,d2Interval);
        %d2pinv = aP*C2mean-dP>0;
        
        %for jj = 1:length(d2pinv)
        %    d2pinvValue = d2Interval(jj);
        %    "checking"
        %    if d2pinv(jj)
        %        "can invade"
        %        p1 = plot(d1,d2pinvValue*ones(size(d1)),"-","Color",...
        %            colorP,"LineWidth",2);
        %        p1.Color(4)=0.3;
        %    end
        %end
        if strcmp(model,'linsatlin')
            % Invasion boundary of C1 in R-C2-P system if its stable
            Rmax = 1;
            endPoint = dC2per-(-aC1*RS+aC2*RS/(1+hC2*aC2*RS)); % ?
            
            d1 = 0:0.001:endPoint;
            
            plot(d1,-aC1*RS+aC2*RS/(1+hC2*aC2*RS)+d1,':',"Color", 'w',... %colorC2,
                'Linewidth',3);
        end
    else
        dC2per = 0;
        sprintf("RS complex, RS = %s", num2str(RS))
    end

    % C1 outcompeted in R-C1/C2-P
    Rmax = 1; % RS in the system without consumers
    endPoint = dC2per-(aC2*RS/(1+hC2*aC2*RS)-aC1*(1+hC2*aC2*Rmax)*RS/(1+hC2*aC2*RS));
    if imag(endPoint)~=0
        endPoint = 1;
    end
    d1 = 0:0.001:endPoint;
    C1boundary = aC2*RS/(1+hC2*aC2*RS)-aC1*(1+hC2*aC2*Rmax)*RS/(1+hC2*aC2*RS) + d1;
    %plot(d1,C1boundary,"LineStyle","--","LineWidth",3,"Color",[0.6350, 0.0780, 0.1840]);%colorC1);
    
    % C1 outcompeted in R-C1/C2 (for assembly)
    d1 = 0:0.001:dC1max;
    C1boundaryRC12 = d1 .* aC2/(aC1*(1+hC2*aC2*Rmax));
    plot(d1,C1boundaryRC12,"LineStyle",":","Color",[.7 .7 .7], ...
        "LineWidth",1.5);
    
    if strcmp(model,'linsat') || strcmp(model,'linsatlin') || strcmp(model,'linsatsat')
        % Hopf in R-C2
        d2hopf = (aC2*hC2-1)/(hC2*(aC2*hC2+1));
        plot(d1,ones(size(d1))*d2hopf,"-.k")
    end

    if strcmp(model,'linsatlin')
        % Invasion boundary of C2 in R-C1-P system if P is able to persist
        RS = 1 - aC1*(dP/aP);
        d1 = 0:0.001:dC1per;
        d2 = aC2*RS./(1+hC2*aC2*RS)-aC1*RS+d1;
        plot(d1,d2,'-',"Color",'w',...%colorC2,
            'Linewidth',3);
    end
    
    % lin lin lin
    if strcmp(model,'linlinlin')
        % Invasion boundary of C1 in R-C2-P system if P is able to
        % persist and we have the lin lin lin model
        d2line = 0:0.001:dC2per;
        alin = (1 - d2line*hC2) * aC2;

        RS = 1 - alin*(dP/aP);
        d1line = aC1*RS-alin.*RS+d2line;
        plot(d1line,d2line,'--',"Color",'w',...%colorC2,
            'Linewidth',3);

        % Invasion boundary of C2 when lin lin lin 
        RS = 1 - aC1*(dP/aP);
        d2line2 = 0:0.001:dC2max; % basically until dC1per, filter later
        alin2 = (1 - d2line2*hC2) * aC2;
        d1line2 = -alin2*RS+aC1*RS + d2line2;
        d2line2 = d2line2(d1line2<dC1per);
        d1line2 = d1line2(d1line2<dC1per);
        
        plot(d1line2,d2line2,'-',"Color",'w',...%colorC2,
            'Linewidth',3);
    end
    %d1 = 0:0.001:dC1max;
    %d2 = aC2*RS./(1+hC2*aC2*RS)-aC1*RS+d1;
    %plot(d1,d2,'--k','Linewidth',3);

    % Invasion boundary of C2 in R-C1 system if P is not present
    % d1 = dC1per:0.001:dC1max;
    d1 = 0:0.001:dC1max;
    RS = d1/aC1;
    d2 = aC2*RS./(1+hC2*aC2*RS);
    plot(d1,d2,'-',"Color",[.7 .7 .7],'Linewidth',3);
    %d1 = 0:0.001:dC1max;
    %RS = d1/aC1;
    %d2 = aC2*RS./(1+hC2*aC2*RS);
    %plot(d1,d2,'-b','Linewidth',3);

    % Plot where the R* of the 4-species-system is complex
    d1 = 0:0.01:dC1max;
    d2 = 0:0.01:dC2max;
    G1_R = zeros(length(d1),length(d2));
    G2_R = zeros(length(d1),length(d2));
    G1_C1 = zeros(length(d1),length(d2));
    G2_C1 = zeros(length(d1),length(d2));
    G1_C2 = zeros(length(d1),length(d2));
    G2_C2 = zeros(length(d1),length(d2));
    G1_P = zeros(length(d1),length(d2));
    G2_P = zeros(length(d1),length(d2));
    for ii=1:length(d1)
        for jj=1:length(d2)
            i = d1(ii);
            j = d2(jj);
            G_p = (aC1-aC2-(i-j)*hC2*aC2)/(aC1*hC2*aC2); G_q = (i-j)/(aC1*hC2*aC2); % R-C1,C2-P
            G1_Rstar = -G_p/2 + sqrt(G_p^2/4-G_q);
            G2_Rstar = -G_p/2 - sqrt(G_p^2/4-G_q);
            G1_C2 = (G1_Rstar+aC1*dP/aP-1)/(aC1-aC2/(1+hC2*aC2*G1_Rstar));
            G2_C2 = (G2_Rstar+aC1*dP/aP-1)/(aC1-aC2/(1+hC2*aC2*G2_Rstar));
            G1_R(ii,jj) = G1_Rstar;
            G2_R(ii,jj) = G2_Rstar;
            G1_C1(ii,jj) = dP/aP-G1_C2;
            G2_C1(ii,jj) = dP/aP-G2_C2;
            G1_C2(ii,jj) = G1_C2;
            G2_C2(ii,jj) = G2_C2;
            G1_P(ii,jj) = (aC1*G1_Rstar-i)/aP;
            G2_P(ii,jj) = (aC1*G2_Rstar-i)/aP;
            
        end
    end
    
   
    % Cosmetics
    xlim([0,dC1max])
    ylim([0,dC2max])
    box('on')

end